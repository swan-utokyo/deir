import gym
from typing import Dict, Any

import numpy as np
from gym import spaces
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType


class PlainForwardModel(IntrinsicRewardBaseModel):
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
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)
        self._build()
        self._init_modules()
        self._init_optimizers()


    def _build(self) -> None:
        # Build CNN and RNN
        super()._build()

        # Build MLP
        self.model_mlp = ForwardModelOutputHeads(
            feature_dim=self.model_features_dim,
            latent_dim=self.model_latents_dim,
            activation_fn=self.activation_fn,
            action_num=self.action_num,
            mlp_norm=self.model_mlp_norm,
            mlp_layers=self.model_mlp_layers,
        )


    def forward(self,
        curr_obs: Tensor, next_obs: Tensor, last_mems: Tensor,
        curr_act: Tensor, curr_dones: Tensor,
        curr_key_status: Optional[Tensor],
        curr_door_status: Optional[Tensor],
        curr_target_dists: Optional[Tensor],
    ):
        # CNN Extractor
        curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
        next_cnn_embs = self._get_cnn_embeddings(next_obs)

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

        # Forward model
        pred_embs = self.model_mlp(curr_embs, curr_act)

        # Intrinsic reward
        curr_dones = curr_dones.view(-1)
        n_samples = (1 - curr_dones).sum()

        fwd_losses = F.mse_loss(pred_embs, next_embs, reduction='none')
        fwd_losses = fwd_losses.mean(-1) * (1 - curr_dones)
        fwd_loss = fwd_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)

        key_loss, door_loss, pos_loss, \
        key_dist, door_dist, goal_dist = \
            self._get_status_prediction_losses(
                curr_embs, curr_key_status, curr_door_status, curr_target_dists
            )
        return fwd_loss, \
            key_loss, door_loss, pos_loss, \
            key_dist, door_dist, goal_dist, \
            curr_cnn_embs, next_cnn_embs, \
            curr_embs, next_embs, curr_mems


    def get_intrinsic_rewards(self,
        curr_obs, next_obs, last_mems, curr_act, curr_dones, obs_history,
        key_status, door_status, target_dists, stats_logger
    ):
        with th.no_grad():
            fwd_loss, \
            key_loss, door_loss, pos_loss, \
            key_dist, door_dist, goal_dist, \
            fwd_curr_cnn_embs, fwd_next_cnn_embs, \
            _, _, model_mems = \
                self.forward(
                    curr_obs, next_obs, last_mems,
                    curr_act, curr_dones,
                    key_status, door_status, target_dists
                )

        batch_size = curr_obs.shape[0]
        int_rews = np.zeros(batch_size, dtype=np.float32)
        for env_id in range(batch_size):
            # Update historical observation embeddings
            curr_obs_emb = fwd_curr_cnn_embs[env_id].view(1, -1)
            next_obs_emb = fwd_next_cnn_embs[env_id].view(1, -1)
            obs_embs = obs_history[env_id]
            new_embs = [curr_obs_emb, next_obs_emb] if obs_embs is None else [obs_embs, next_obs_emb]
            obs_embs = th.cat(new_embs, dim=0)
            obs_history[env_id] = obs_embs

            # Generate intrinsic reward
            obs_dists = self.calc_euclidean_dists(obs_embs[:-1], obs_embs[-1])
            int_rews[env_id] += obs_dists.min().item()

        # Logging
        stats_logger.add(
            fwd_loss=fwd_loss,
            key_loss=key_loss,
            door_loss=door_loss,
            pos_loss=pos_loss,
            key_dist=key_dist,
            door_dist=door_dist,
            goal_dist=goal_dist,
        )
        return int_rews, model_mems


    def optimize(self, rollout_data, stats_logger):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = rollout_data.actions.long().flatten()

        if self.use_status_predictor:
            curr_key_status = rollout_data.curr_key_status
            curr_door_status = rollout_data.curr_door_status
            curr_target_dists = rollout_data.curr_target_dists
        else:
            curr_key_status = None
            curr_door_status = None
            curr_target_dists = None

        fwd_loss, \
        key_loss, door_loss, pos_loss, \
        key_dist, door_dist, goal_dist, \
        _, _, _, _, _ = \
            self.forward(
                rollout_data.observations,
                rollout_data.new_observations,
                rollout_data.last_model_mems,
                actions,
                rollout_data.episode_dones,
                curr_key_status,
                curr_door_status,
                curr_target_dists,
            )

        forward_loss = fwd_loss
        self.model_optimizer.zero_grad()
        forward_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()

        if self.use_status_predictor:
            predictor_loss = key_loss + door_loss + pos_loss
            self.predictor_optimizer.zero_grad()
            predictor_loss.backward()
            self.predictor_optimizer.step()

        stats_logger.add(
            fwd_loss=fwd_loss,
            key_loss=key_loss,
            door_loss=door_loss,
            pos_loss=pos_loss,
            key_dist=key_dist,
            door_dist=door_dist,
            goal_dist=goal_dist,
        )