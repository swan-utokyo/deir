import click
import warnings
import torch as th

# noinspection PyUnresolvedReferences
from src.env.minigrid_envs import *
from src.algo.ppo_model import PPOModel
from src.algo.ppo_trainer import PPOTrainer
from src.utils.configs import TrainingConfig
from stable_baselines3.common.utils import set_random_seed

warnings.filterwarnings("ignore", category=DeprecationWarning)


def train(config):
    th.autograd.set_detect_anomaly(False)
    th.set_default_dtype(th.float32)
    th.backends.cudnn.benchmark = False

    wrapper_class = config.get_wrapper_class()
    venv = config.get_venv(wrapper_class)
    callbacks = config.get_callbacks()
    optimizer_class, optimizer_kwargs = config.get_optimizer()
    activation_fn, cnn_activation_fn = config.get_activation_fn()
    config.cast_enum_values()
    policy_features_extractor_class, \
        features_extractor_common_kwargs, \
        model_cnn_features_extractor_class, \
        model_features_extractor_common_kwargs = \
        config.get_cnn_kwargs(cnn_activation_fn)

    policy_kwargs = dict(
        run_id=config.run_id,
        n_envs=config.num_processes,
        activation_fn=activation_fn,
        learning_rate=config.learning_rate,
        model_learning_rate=config.model_learning_rate,
        policy_features_extractor_class=policy_features_extractor_class,
        policy_features_extractor_kwargs=features_extractor_common_kwargs,
        model_cnn_features_extractor_class=model_cnn_features_extractor_class,
        model_cnn_features_extractor_kwargs=model_features_extractor_common_kwargs,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        max_grad_norm=config.max_grad_norm,
        model_features_dim=config.model_features_dim,
        latents_dim=config.latents_dim,
        model_latents_dim=config.model_latents_dim,
        policy_mlp_norm=config.policy_mlp_norm,
        model_mlp_norm=config.model_mlp_norm,
        model_cnn_norm=config.model_cnn_norm,
        model_mlp_layers=config.model_mlp_layers,
        use_status_predictor=config.use_status_predictor,
        gru_layers=config.gru_layers,
        policy_mlp_layers=config.policy_mlp_layers,
        policy_gru_norm=config.policy_gru_norm,
        use_model_rnn=config.use_model_rnn,
        model_gru_norm=config.model_gru_norm,
        total_timesteps=config.total_steps,
        n_steps=config.n_steps,
        int_rew_source=config.int_rew_source,
        icm_forward_loss_coef=config.icm_forward_loss_coef,
        ngu_knn_k=config.ngu_knn_k,
        ngu_dst_momentum=config.ngu_dst_momentum,
        ngu_use_rnd=config.ngu_use_rnd,
        rnd_err_norm=config.rnd_err_norm,
        rnd_err_momentum=config.rnd_err_momentum,
        rnd_use_policy_emb=config.rnd_use_policy_emb,
        dsc_obs_queue_len=config.dsc_obs_queue_len,
        log_dsc_verbose=config.log_dsc_verbose,
    )

    model = PPOTrainer(
        policy=PPOModel,
        env=venv,
        seed=config.run_id,
        run_id=config.run_id,
        can_see_walls=config.can_see_walls,
        image_noise_scale=config.image_noise_scale,
        n_steps=config.n_steps,
        n_epochs=config.n_epochs,
        model_n_epochs=config.model_n_epochs,
        learning_rate=config.learning_rate,
        model_learning_rate=config.model_learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        batch_size=config.batch_size,
        pg_coef=config.pg_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        ext_rew_coef=config.ext_rew_coef,
        int_rew_source=config.int_rew_source,
        int_rew_coef=config.int_rew_coef,
        int_rew_norm=config.int_rew_norm,
        int_rew_momentum=config.int_rew_momentum,
        int_rew_eps=config.int_rew_eps,
        int_rew_clip=config.int_rew_clip,
        adv_momentum=config.adv_momentum,
        adv_norm=config.adv_norm,
        adv_eps=config.adv_eps,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        policy_kwargs=policy_kwargs,
        env_source=config.env_source,
        env_render=config.env_render,
        fixed_seed=config.fixed_seed,
        use_wandb=config.use_wandb,
        local_logger=config.local_logger,
        enable_plotting=config.enable_plotting,
        plot_interval=config.plot_interval,
        plot_colormap=config.plot_colormap,
        log_explored_states=config.log_explored_states,
        verbose=0,
    )

    if config.run_id == 0:
        print('model.policy:', model.policy)

    model.learn(
        total_timesteps=config.total_steps,
        callback=callbacks)


@click.command()
# Training params
@click.option('--run_id', default=0, type=int, help='Index (and seed) of the current run')
@click.option('--group_name', type=str, help='Group name (wandb option), leave blank if not logging with wandb')
@click.option('--log_dir', default='./logs', type=str, help='Directory for saving training logs')
@click.option('--total_steps', default=int(1e6), type=int, help='Total number of frames to run for training')
@click.option('--features_dim', default=64, type=int, help='Number of neurons of a learned embedding (PPO)')
@click.option('--model_features_dim', default=128, type=int,
              help='Number of neurons of a learned embedding (dynamics model)')
@click.option('--learning_rate', default=3e-4, type=float, help='Learning rate of PPO')
@click.option('--model_learning_rate', default=3e-4, type=float, help='Learning rate of the dynamics model')
@click.option('--num_processes', default=16, type=int, help='Number of training processes (workers)')
@click.option('--batch_size', default=512, type=int, help='Batch size')
@click.option('--n_steps', default=512, type=int, help='Number of steps to run for each process per update')
# Env params
@click.option('--env_source', default='minigrid', type=str, help='minigrid or procgen')
@click.option('--game_name', default="DoorKey-8x8", type=str, help='e.g. DoorKey-8x8, ninja, jumper')
@click.option('--project_name', required=False, type=str, help='Where to store training logs (wandb option)')
@click.option('--map_size', default=5, type=int, help='Size of the minigrid room')
@click.option('--can_see_walls', default=1, type=int, help='Whether walls are visible to the agent')
@click.option('--fully_obs', default=0, type=int, help='Whether the agent can receive full observations')
@click.option('--image_noise_scale', default=0.0, type=float, help='Standard deviation of the Gaussian noise')
@click.option('--procgen_mode', default='hard', type=str, help='Mode of ProcGen games (easy or hard)')
@click.option('--procgen_num_threads', default=4, type=int, help='Number of parallel ProcGen threads')
@click.option('--log_explored_states', default=0, type=int, help='Whether to log the number of explored states')
@click.option('--fixed_seed', default=-1, type=int, help='Whether to use a fixed env seed (MiniGrid)')
# Algo params
@click.option('--n_epochs', default=4, type=int, help='Number of epochs to train policy and value nets')
@click.option('--model_n_epochs', default=4, type=int, help='Number of epochs to train common_models')
@click.option('--gamma', default=0.99, type=float, help='Discount factor')
@click.option('--gae_lambda', default=0.95, type=float, help='GAE lambda')
@click.option('--pg_coef', default=1.0, type=float, help='Coefficient of policy gradients')
@click.option('--vf_coef', default=0.5, type=float, help='Coefficient of value function loss')
@click.option('--ent_coef', default=0.01, type=float, help='Coefficient of policy entropy')
@click.option('--max_grad_norm', default=0.5, type=float, help='Maximum norm of gradient')
@click.option('--clip_range', default=0.2, type=float, help='PPO clip range of the policy network')
@click.option('--clip_range_vf', default=-1, type=float,
              help='PPO clip range of the value function (-1: disabled, >0: enabled)')
@click.option('--adv_norm', default=2, type=int,
              help='Normalized advantages by: [0] No normalization [1] Standardization per mini-batch [2] Standardization per rollout buffer [3] Standardization w.o. subtracting the mean per rollout buffer')
@click.option('--adv_eps', default=1e-5, type=float, help='Epsilon for advantage normalization')
@click.option('--adv_momentum', default=0.9, type=float, help='EMA smoothing factor for advantage normalization')
# Reward params
@click.option('--ext_rew_coef', default=1.0, type=float, help='Coefficient of extrinsic rewards')
@click.option('--int_rew_coef', default=1e-2, type=float, help='Coefficient of intrinsic rewards (IRs)')
@click.option('--int_rew_source', default='DEIR', type=str,
              help='Source of IRs: [NoModel|DEIR|ICM|RND|NGU|NovelD|PlainDiscriminator|PlainInverse|PlainForward]')
@click.option('--int_rew_norm', default=1, type=int,
              help='Normalized IRs by: [0] No normalization [1] Standardization [2] Min-max normalization [3] Standardization w.o. subtracting the mean')
@click.option('--int_rew_momentum', default=0.9, type=float,
              help='EMA smoothing factor for IR normalization (-1: total average)')
@click.option('--int_rew_eps', default=1e-5, type=float, help='Epsilon for IR normalization')
@click.option('--int_rew_clip', default=-1, type=float, help='Clip IRs into [-X, X] when X>0')
@click.option('--dsc_obs_queue_len', default=100000, type=int, help='Maximum length of observation queue (DEIR)')
@click.option('--icm_forward_loss_coef', default=0.2, type=float, help='Coefficient of forward model losses (ICM)')
@click.option('--ngu_knn_k', default=10, type=int, help='Search for K nearest neighbors (NGU)')
@click.option('--ngu_use_rnd', default=1, type=int, help='Whether to enable lifelong IRs generated by RND (NGU)')
@click.option('--ngu_dst_momentum', default=0.997, type=float,
              help='EMA smoothing factor for averaging embedding distances (NGU)')
@click.option('--rnd_use_policy_emb', default=1, type=int,
              help='Whether to use the embeddings learned by policy/value nets as inputs (RND)')
@click.option('--rnd_err_norm', default=1, type=int,
              help='Normalized RND errors by: [0] No normalization [1] Standardization [2] Min-max normalization [3] Standardization w.o. subtracting the mean')
@click.option('--rnd_err_momentum', default=-1, type=float,
              help='EMA smoothing factor for RND error normalization (-1: total average)')
# Network params
@click.option('--use_model_rnn', default=1, type=int, help='Whether to enable RNNs for the dynamics model')
@click.option('--latents_dim', default=256, type=int, help='Dimensions of latent features in policy/value nets\' MLPs')
@click.option('--model_latents_dim', default=256, type=int,
              help='Dimensions of latent features in the dynamics model\'s MLP')
@click.option('--policy_cnn_type', default=0, type=int, help='CNN Structure ([0-2] from small to large)')
@click.option('--policy_mlp_layers', default=1, type=int, help='Number of latent layers used in the policy\'s MLP')
@click.option('--policy_cnn_norm', default='BatchNorm', type=str, help='Normalization type for policy/value nets\' CNN')
@click.option('--policy_mlp_norm', default='BatchNorm', type=str, help='Normalization type for policy/value nets\' MLP')
@click.option('--policy_gru_norm', default='NoNorm', type=str, help='Normalization type for policy/value nets\' GRU')
@click.option('--model_cnn_type', default=0, type=int, help='CNN Structure ([0-2] from small to large)')
@click.option('--model_mlp_layers', default=1, type=int, help='Number of latent layers used in the model\'s MLP')
@click.option('--model_cnn_norm', default='BatchNorm', type=str,
              help='Normalization type for the dynamics model\'s CNN')
@click.option('--model_mlp_norm', default='BatchNorm', type=str,
              help='Normalization type for the dynamics model\'s MLP')
@click.option('--model_gru_norm', default='NoNorm', type=str, help='Normalization type for the dynamics model\'s GRU')
@click.option('--activation_fn', default='relu', type=str, help='Activation function for non-CNN layers')
@click.option('--cnn_activation_fn', default='relu', type=str, help='Activation function for CNN layers')
@click.option('--gru_layers', default=1, type=int, help='Number of GRU layers in both the policy and the model')
# Optimizer params
@click.option('--optimizer', default='adam', type=str, help='Optimizer, adam or rmsprop')
@click.option('--optim_eps', default=1e-5, type=float, help='Epsilon for optimizers')
@click.option('--adam_beta1', default=0.9, type=float, help='Adam optimizer option')
@click.option('--adam_beta2', default=0.999, type=float, help='Adam optimizer option')
@click.option('--rmsprop_alpha', default=0.99, type=float, help='RMSProp optimizer option')
@click.option('--rmsprop_momentum', default=0.0, type=float, help='RMSProp optimizer option')
# Logging & Analysis options
@click.option('--write_local_logs', default=1, type=int, help='Whether to output training logs locally')
@click.option('--enable_plotting', default=0, type=int, help='Whether to generate plots for analysis')
@click.option('--plot_interval', default=10, type=int, help='Interval of generating plots (iterations)')
@click.option('--plot_colormap', default='Blues', type=str, help='Colormap of plots to generate')
@click.option('--record_video', default=0, type=int, help='Whether to record video')
@click.option('--rec_interval', default=10, type=int, help='Interval of two videos (iterations)')
@click.option('--video_length', default=512, type=int, help='Length of the video (frames)')
@click.option('--log_dsc_verbose', default=0, type=int, help='Whether to record the discriminator loss for each action')
@click.option('--env_render', default=0, type=int, help='Whether to render games in human mode')
@click.option('--use_status_predictor', default=0, type=int,
              help='Whether to train status predictors for analysis (MiniGrid only)')
def main(
    run_id, group_name, log_dir, total_steps, features_dim, model_features_dim, learning_rate, model_learning_rate,
    num_processes, batch_size, n_steps, env_source, game_name, project_name, map_size, can_see_walls, fully_obs,
    image_noise_scale, procgen_mode, procgen_num_threads, log_explored_states, fixed_seed, n_epochs, model_n_epochs,
    gamma, gae_lambda, pg_coef, vf_coef, ent_coef, max_grad_norm, clip_range, clip_range_vf, adv_norm, adv_eps,
    adv_momentum, ext_rew_coef, int_rew_coef, int_rew_source, int_rew_norm, int_rew_momentum, int_rew_eps, int_rew_clip,
    dsc_obs_queue_len, icm_forward_loss_coef, ngu_knn_k, ngu_use_rnd, ngu_dst_momentum, rnd_use_policy_emb,
    rnd_err_norm, rnd_err_momentum, use_model_rnn, latents_dim, model_latents_dim, policy_cnn_type, policy_mlp_layers,
    policy_cnn_norm, policy_mlp_norm, policy_gru_norm, model_cnn_type, model_mlp_layers, model_cnn_norm, model_mlp_norm,
    model_gru_norm, activation_fn, cnn_activation_fn, gru_layers, optimizer, optim_eps, adam_beta1, adam_beta2,
    rmsprop_alpha, rmsprop_momentum, write_local_logs, enable_plotting, plot_interval, plot_colormap, record_video,
    rec_interval, video_length, log_dsc_verbose, env_render, use_status_predictor
):
    set_random_seed(run_id, using_cuda=True)
    args = locals().items()
    config = TrainingConfig()
    for k, v in args: setattr(config, k, v)

    config.init_env_name(game_name, project_name)
    config.init_meta_info()
    config.init_logger()
    config.init_values()

    train(config)

    config.close()


if __name__ == '__main__':
    main()
