import gym
from typing import Dict, Any

from gym import spaces
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType
from src.utils.loggers import StatisticsLogger


class ICMModel(IntrinsicRewardBaseModel):
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
        # Model-specific params
        icm_forward_loss_coef: float = 0.2,
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)

        self.icm_forward_loss_coef = icm_forward_loss_coef

        self._build()
        self._init_modules()
        self._init_optimizers()


    def _build(self) -> None:
        # Build CNN and RNN
        super()._build()

        # Build MLP
        self.model_mlp = ICMOutputHeads(
            features_dim= self.model_features_dim,
            latents_dim= self.model_latents_dim,
            activation_fn = self.activation_fn,
            action_num = self.action_num,
            mlp_norm = self.model_mlp_norm,
            mlp_layers = self.model_mlp_layers,
        )


    # Based on: https://github.com/pathak22/noreward-rl/blob/master/src/model.py
    def forward(self,
        curr_obs: Tensor, next_obs: Tensor, last_mems: Tensor,
        curr_act: Tensor, curr_dones: Tensor,
    ):
        curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
        next_cnn_embs = self._get_cnn_embeddings(next_obs)

        # Get RNN memories
        if self.use_model_rnn:
            curr_mems = self._get_rnn_embeddings(last_mems, curr_cnn_embs, self.model_rnns)
            next_mems = self._get_rnn_embeddings(curr_mems, next_cnn_embs, self.model_rnns)
            curr_rnn_embs = th.squeeze(curr_mems[:, -1, :])
            next_rnn_embs = th.squeeze(next_mems[:, -1, :])
            curr_embs = curr_rnn_embs
            next_embs = next_rnn_embs
        else:
            curr_embs = curr_cnn_embs
            next_embs = next_cnn_embs
            curr_mems = None

        curr_dones = curr_dones.view(-1)
        n_samples = (1 - curr_dones).sum()

        pred_embs, pred_act = self.model_mlp(curr_embs, next_embs, curr_act)

        # Forward model
        fwd_losses = 0.5 * F.mse_loss(pred_embs, next_embs, reduction='none') \
                         * self.model_features_dim  # eta (scaling factor)
        fwd_losses = fwd_losses.mean(dim=1) * (1 - curr_dones)
        fwd_loss = fwd_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)

        # Inverse model
        inv_losses = F.cross_entropy(pred_act, curr_act, reduction='none') * (1 - curr_dones)
        inv_loss = inv_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)

        return fwd_losses, fwd_loss, inv_loss, curr_mems


    def get_intrinsic_rewards(self, curr_obs, next_obs, last_mems, curr_act, curr_dones, stats_logger):
        with th.no_grad():
            icm_fwd_losses, fwd_loss, inv_loss, _ = \
                self.forward(curr_obs, next_obs, last_mems, curr_act, curr_dones)
        stats_logger.add(
            inv_loss=inv_loss,
            fwd_loss=fwd_loss,
        )
        int_rews = icm_fwd_losses.clone().cpu().numpy()
        return int_rews, last_mems


    def optimize(self, rollout_data, stats_logger):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = rollout_data.actions.long().flatten()

        _, fwd_loss, inv_loss, _ = \
            self.forward(
                rollout_data.observations,
                rollout_data.new_observations,
                rollout_data.last_model_mems,
                actions,
                rollout_data.episode_dones,
            )

        icm_loss = inv_loss * (1 - self.icm_forward_loss_coef) + \
                   fwd_loss * self.icm_forward_loss_coef

        stats_logger.add(
            icm_loss=icm_loss,
            inv_loss=inv_loss,
            fwd_loss=fwd_loss,
        )

        # Optimization
        self.model_optimizer.zero_grad()
        icm_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()