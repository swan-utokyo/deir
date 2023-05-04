import torch as th
import wandb
import warnings

from gym import spaces
from torch import Tensor

from src.utils.loggers import StatisticsLogger, LocalLogger
from src.algo.ppo_rollout import PPORollout
from src.utils.enum_types import ModelType

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from torch.nn import functional as F

from typing import Any, Dict, Optional, Type, Union


class PPOTrainer(PPORollout):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        run_id: int,
        learning_rate: Union[float, Schedule] = 3e-4,
        model_learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 4,
        model_n_epochs: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        pg_coef: float = 1.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        int_rew_source: ModelType = ModelType.DEIR,
        int_rew_norm: int = 0,
        int_rew_coef: float = 1e-3,
        int_rew_momentum: float = 1,
        int_rew_eps: float = 0.0,
        adv_momentum: float = 0.0,
        image_noise_scale: float = 0.0,
        int_rew_clip: float = 0.0,
        enable_plotting: int = 0,
        can_see_walls: int = 1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        ext_rew_coef: float = 1.0,
        adv_norm: int = 1,
        adv_eps: float = 1e-8,
        env_source: Optional[str] = None,
        env_render: Optional[int] = None,
        fixed_seed: Optional[int] = None,
        plot_interval: int = 10,
        plot_colormap: str = 'Blues',
        log_explored_states: Optional[int] = None,
        local_logger: Optional[LocalLogger] = None,
        use_wandb: bool = False,
    ):
        super(PPOTrainer, self).__init__(
            policy,
            env,
            run_id,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            pg_coef=pg_coef,
            vf_coef=vf_coef,
            int_rew_coef=int_rew_coef,
            int_rew_norm=int_rew_norm,
            int_rew_momentum=int_rew_momentum,
            int_rew_eps=int_rew_eps,
            int_rew_clip=int_rew_clip,
            adv_momentum=adv_momentum,
            image_noise_scale=image_noise_scale,
            enable_plotting=enable_plotting,
            can_see_walls=can_see_walls,
            ext_rew_coef=ext_rew_coef,
            adv_norm=adv_norm,
            adv_eps=adv_eps,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            batch_size=batch_size,
            int_rew_source=int_rew_source,
            _init_setup_model=False,
            env_source=env_source,
            env_render=env_render,
            fixed_seed=fixed_seed,
            plot_interval=plot_interval,
            plot_colormap=plot_colormap,
            log_explored_states=log_explored_states,
            local_logger=local_logger,
            use_wandb=use_wandb,
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_n_epochs = model_n_epochs
        self.model_learning_rate = model_learning_rate
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.adv_norm = adv_norm
        self.adv_eps = adv_eps
        self.pg_loss_avg = None
        self.ent_loss_avg = None
        self.ent_coef_init = ent_coef
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPOTrainer, self)._setup_model()
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive or `None`"
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)


    def train_policy_and_models(self, clip_range, clip_range_vf) -> Tensor:
        loss = None
        continue_training = True

        for epoch in range(max(self.n_epochs, self.model_n_epochs)):
            for rollout_data in self.ppo_rollout_buffer.get(self.batch_size):
                # Train intrinsic reward models
                if epoch < self.model_n_epochs:
                    if self.policy.int_rew_source != ModelType.NoModel:
                        self.policy.int_rew_model.optimize(
                            rollout_data=rollout_data,
                            stats_logger=self.training_stats
                        )

                # Training for policy and value nets
                if epoch < self.n_epochs:
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    # if that line is commented (as in SAC)
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy, memories = \
                        self.policy.evaluate_policy(
                            rollout_data.observations,
                            actions,
                            rollout_data.last_policy_mems,
                        )
                    values = values.flatten()

                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalize Advangages per mini-batch
                    if self.adv_norm == 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + self.adv_eps)
                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    # Logging
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    # Policy & Value Losses
                    loss = self.pg_coef * policy_loss + \
                           self.ent_coef * entropy_loss + \
                           self.vf_coef * value_loss

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Logging
                    self.training_stats.add(
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        entropy_loss=entropy_loss,
                        adv=advantages.mean(),
                        adv_std=advantages.std(),
                        clip_fraction=clip_fraction,
                        approx_kl_div=approx_kl_div,
                    )

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                    # END OF A TRAINING BATCH

                    if not continue_training:
                        break
        return loss


    def train(self) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Log training stats per each iteration
        self.training_stats = StatisticsLogger(mode='train')

        # Train PPO policy (+value function) and intrinsic reward models
        ppo_loss = self.train_policy_and_models(clip_range, clip_range_vf)

        # Update stats
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.ppo_rollout_buffer.values.flatten(), self.ppo_rollout_buffer.returns.flatten())

        # Logging
        log_data = {
            "time/total_timesteps": self.num_timesteps,
            "train/loss": ppo_loss.item(),
            "train/explained_variance": explained_var,
            "train/n_updates": self._n_updates,
        }
        if hasattr(self.policy, "log_std"):
            log_data.update({"train/std": th.exp(self.policy.log_std).mean().item()})
        # Update with other stats
        log_data.update(self.training_stats.to_dict())
        # Logging with wandb
        if self.use_wandb:
            wandb.log(log_data)
        # Logging with local logger
        if self.local_logger is not None:
            self.local_logger.write(log_data, log_type='train')

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "CustomPPOAlgo",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> BaseAlgorithm:

        return super(PPOTrainer, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
