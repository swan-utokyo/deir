import gym
from typing import Dict, Any
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.common_func import normalize_rewards
from src.utils.enum_types import NormType
from src.utils.running_mean_std import RunningMeanStd


class RNDModel(IntrinsicRewardBaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
        # Method-specific params
        rnd_err_norm: int = 0,
        rnd_err_momentum: float = -1.0,
        rnd_use_policy_emb: int = 1,
        policy_cnn: Type[nn.Module] = None,
        policy_rnns: Type[nn.Module] = None,
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)

        self.policy_cnn = policy_cnn
        self.policy_rnns = policy_rnns
        self.rnd_use_policy_emb = rnd_use_policy_emb
        self.rnd_err_norm = rnd_err_norm
        self.rnd_err_momentum = rnd_err_momentum
        self.rnd_err_running_stats = RunningMeanStd(momentum=self.rnd_err_momentum)

        self._build()
        self._init_modules()
        self._init_optimizers()


    def _build(self) -> None:
        # Build CNN and RNN
        super()._build()

        # Build MLP
        self.model_mlp = RNDOutputHeads(
            features_dim = self.model_features_dim,
            latents_dim = self.model_latents_dim,
            outputs_dim = self.model_latents_dim,
            activation_fn = self.activation_fn,
            mlp_norm = self.model_mlp_norm,
            mlp_layers = self.model_mlp_layers,
        )


    def forward(self, curr_obs: Tensor, last_mems: Tensor, curr_dones: Optional[Tensor]):
        curr_mlp_inputs, curr_mems = self._get_rnd_embeddings(curr_obs, last_mems)

        tgt, prd = self.model_mlp(curr_mlp_inputs)
        rnd_losses = F.mse_loss(prd, tgt.detach(), reduction='none').mean(-1)

        curr_dones = curr_dones.view(-1)
        n_samples = (1 - curr_dones).sum()
        rnd_loss = rnd_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)
        return rnd_loss, rnd_losses, curr_mems


    def get_intrinsic_rewards(self, curr_obs, last_mems, curr_dones, stats_logger):
        with th.no_grad():
            rnd_loss, rnd_losses, model_mems = \
                self.forward(curr_obs, last_mems, curr_dones)
        rnd_rewards = rnd_losses.clone().cpu().numpy()

        if self.rnd_err_norm > 0:
            # Normalize RND error per step
            self.rnd_err_running_stats.update(rnd_rewards)
            rnd_rewards = normalize_rewards(
                norm_type=self.rnd_err_norm,
                rewards=rnd_rewards,
                mean=self.rnd_err_running_stats.mean,
                std=self.rnd_err_running_stats.std,
            )

        stats_logger.add(rnd_loss=rnd_loss)
        return rnd_rewards, model_mems


    def optimize(self, rollout_data, stats_logger):
        rnd_loss, _, _ = \
            self.forward(
                rollout_data.observations,
                rollout_data.last_model_mems,
                rollout_data.episode_dones,
            )

        stats_logger.add(rnd_loss=rnd_loss)

        # Optimization
        self.model_optimizer.zero_grad()
        rnd_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()