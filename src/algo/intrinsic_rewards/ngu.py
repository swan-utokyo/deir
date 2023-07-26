"""
[NOTICE] As specified in Appendix A.5 of our arXiv preprint, this implementation of 
NGU is NOT COMPLETELY IDENTICAL to the original implementation. It is because the 
learning framework we adopted for training and comparing all exploration methods is 
more lightweight than the originally proposed one. However, we confirmed that every 
step of the original algorithm for intrinsic reward generation was followed in 
our reproduction (according to the original definition given in NGU’s paper).
"""
import gym
from typing import Dict, Any

import numpy as np
from gym import spaces
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.common_func import normalize_rewards
from src.utils.enum_types import NormType
from src.utils.running_mean_std import RunningMeanStd


class NGUModel(IntrinsicRewardBaseModel):
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
        ngu_knn_k: int = 10,
        ngu_dst_momentum: float = 0.997,
        ngu_use_rnd: int = 1,
        rnd_err_norm: int = 0,
        rnd_err_momentum: float = -1,
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
        self.ngu_knn_k = ngu_knn_k
        self.ngu_use_rnd = ngu_use_rnd
        self.ngu_dst_momentum = ngu_dst_momentum
        self.ngu_moving_avg_dists = RunningMeanStd(momentum=self.ngu_dst_momentum)
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
        self.model_mlp = NGUOutputHeads(
            features_dim = self.model_features_dim,
            latents_dim = self.model_latents_dim,
            activation_fn = self.activation_fn,
            action_num = self.action_num,
            mlp_norm = self.model_mlp_norm,
            mlp_layers = self.model_mlp_layers,
            use_rnd=self.ngu_use_rnd,
        )

    def forward(self,
        curr_obs: Tensor, next_obs: Tensor, last_mems: Tensor,
        curr_act: Tensor, curr_dones: Tensor
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

        pred_act = self.model_mlp.inverse_forward(curr_embs, next_embs)

        # Inverse model
        curr_dones = curr_dones.view(-1)
        n_samples = (1 - curr_dones).sum()
        inv_losses = F.cross_entropy(pred_act, curr_act, reduction='none') * (1 - curr_dones)
        inv_loss = inv_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)

        # RND
        rnd_losses, rnd_loss, _ = None, None, None
        if self.ngu_use_rnd:
            curr_rnd_embs, _ = self._get_rnd_embeddings(curr_obs, last_mems)
            tgt_out, prd_out = self.model_mlp.rnd_forward(curr_rnd_embs)

            rnd_losses = F.mse_loss(prd_out, tgt_out.detach(), reduction='none').mean(-1) * (1 - curr_dones)
            rnd_loss = rnd_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)

        return inv_loss, curr_embs, next_embs, rnd_losses, rnd_loss, curr_mems


    def get_intrinsic_rewards(self, curr_obs, next_obs, last_mems, curr_act, curr_dones, obs_history, stats_logger):
        with th.no_grad():
            inv_loss, ngu_curr_embs, ngu_next_embs, ngu_rnd_losses, ngu_rnd_loss, _ = \
                self.forward(curr_obs, next_obs, last_mems, curr_act, curr_dones)
            stats_logger.add(inv_loss=inv_loss)

            if ngu_rnd_losses is not None:
                ngu_rnd_error = ngu_rnd_losses.clone().cpu().numpy()

                if self.rnd_err_norm > 0:
                    # Normalize RND error per step
                    self.rnd_err_running_stats.update(ngu_rnd_error)
                    ngu_rnd_error = normalize_rewards(
                        norm_type=self.rnd_err_norm,
                        rewards=ngu_rnd_error,
                        mean=self.rnd_err_running_stats.mean,
                        std=self.rnd_err_running_stats.std,
                    )

                ngu_lifelong_rewards = ngu_rnd_error + 1
                stats_logger.add(rnd_loss=ngu_rnd_loss)

        # Create IRs
        batch_size = curr_obs.shape[0]
        int_rews = np.zeros(batch_size, dtype=np.float32)
        for env_id in range(batch_size):
            # Update historical observation embeddings
            curr_emb = ngu_curr_embs[env_id].view(1, -1)
            next_emb = ngu_next_embs[env_id].view(1, -1)
            obs_embs = obs_history[env_id]
            new_embs = [curr_emb, next_emb] if obs_embs is None else [obs_embs, next_emb]
            obs_embs = th.cat(new_embs, dim=0)
            obs_history[env_id] = obs_embs

            # Implemented based on the paper of NGU (Algorithm 1)
            episodic_reward = 0.0
            if obs_embs.shape[0] > 1:
                # Compute the k-nearest neighbours of f (x_t) in M and store them in a list N_k
                # - d is the Euclidean distance and
                # - d^2_m is a running average of the squared Euclidean distance of the k-nearest neighbors.
                knn_dists = self.calc_euclidean_dists(obs_embs[:-1], obs_embs[-1]) ** 2
                knn_dists = knn_dists.clone().cpu().numpy()
                knn_dists = np.sort(knn_dists)[:self.ngu_knn_k]
                # Update the moving average d^2_m with the list of distances d_k
                self.ngu_moving_avg_dists.update(knn_dists)
                moving_avg_dist = self.ngu_moving_avg_dists.mean
                # Normalize the distances d_k with the updated moving average d^2_m
                normalized_dists = knn_dists / (moving_avg_dist + 1e-5)
                # Cluster the normalized distances d_n
                # i.e. they become 0 if too small and 0k is a list of k zeros
                normalized_dists = np.maximum(normalized_dists - 0.008, np.zeros_like(knn_dists))
                # Compute the Kernel values between the embedding f (x_t) and its neighbours N_k
                kernel_values = 0.0001 / (normalized_dists + 0.0001)
                # Compute the similarity between the embedding f (x_t) and its neighbours N_k
                simlarity = np.sqrt(kernel_values.sum()) + 0.001
                # Compute the episodic intrinsic reward at time t
                if simlarity <= 8:
                    episodic_reward += 1 / simlarity

            if self.ngu_use_rnd and ngu_lifelong_rewards is not None:
                L = 5.0  # L is a chosen maximum reward scaling (default: 5)
                lifelong_reward = min(max(ngu_lifelong_rewards[env_id], 1.0), L)
                int_rews[env_id] += episodic_reward * lifelong_reward
            else:
                int_rews[env_id] += episodic_reward

        return int_rews, last_mems


    def optimize(self, rollout_data, stats_logger):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = rollout_data.actions.long().flatten()

        inv_loss, _, _, _, rnd_loss, _ = \
            self.forward(
                rollout_data.observations,
                rollout_data.new_observations,
                rollout_data.last_model_mems,
                actions,
                rollout_data.episode_dones,
            )

        ngu_loss = inv_loss
        stats_logger.add(inv_loss=inv_loss)

        if self.ngu_use_rnd:
            ngu_loss = ngu_loss + rnd_loss
            stats_logger.add(rnd_loss=rnd_loss)

        # Optimization
        self.model_optimizer.zero_grad()
        ngu_loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()
