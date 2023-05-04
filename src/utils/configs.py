import os
import time

import torch as th
import wandb
from torch import nn
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from procgen import ProcgenEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from datetime import datetime

from src.algo.common_models.cnns import BatchNormCnnFeaturesExtractor, LayerNormCnnFeaturesExtractor, \
    CnnFeaturesExtractor
from src.env.subproc_vec_env import CustomSubprocVecEnv
from src.utils.enum_types import EnvSrc, NormType, ModelType
from wandb.integration.sb3 import WandbCallback

from src.utils.loggers import LocalLogger
from src.utils.video_recorder import VecVideoRecorder


class TrainingConfig():
    def __init__(self):
        self.dtype = th.float32
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    def init_meta_info(self):
        self.file_path = __file__
        self.model_name = os.path.basename(__file__)
        self.start_time = time.time()
        self.start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def init_env_name(self, game_name, project_name):
        env_name = game_name
        self.env_source = EnvSrc.get_enum_env_src(self.env_source)
        if self.env_source == EnvSrc.MiniGrid and not game_name.startswith('MiniGrid-'):
            env_name = f'MiniGrid-{game_name}'
            env_name += '-v0'
        self.env_name = env_name
        self.project_name = env_name if project_name is None else project_name

    def init_logger(self):
        if self.group_name is not None:
            self.wandb_run = wandb.init(
                name=f'run-id-{self.run_id}',
                entity='abcde-project',  # your project name on wandb
                project=self.project_name,
                group=self.group_name,
                settings=wandb.Settings(start_method="fork"),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=True,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
            self.wandb_run = None

        self.log_dir = os.path.join(self.log_dir, self.env_name, self.start_datetime, str(self.run_id))
        os.makedirs(self.log_dir, exist_ok=True)
        if self.write_local_logs:
            self.local_logger = LocalLogger(self.log_dir)
            print(f'Writing local logs at {self.log_dir}')
        else:
            self.local_logger = None

        print(f'Starting run {self.run_id}')

    def init_values(self):
        if self.clip_range_vf <= 0:
            self.clip_range_vf = None

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def get_wrapper_class(self):
        if self.env_source == EnvSrc.MiniGrid:
            if self.fully_obs:
                wrapper_class = lambda x: ImgObsWrapper(FullyObsWrapper(x))
            else:
                wrapper_class = lambda x: ImgObsWrapper(x)

            if self.fixed_seed >= 0 and self.env_source == EnvSrc.MiniGrid:
                assert not self.fully_obs
                _seeds = [self.fixed_seed]
                wrapper_class = lambda x: ImgObsWrapper(ReseedWrapper(x, seeds=_seeds))
            return wrapper_class
        return None

    def get_venv(self, wrapper_class=None):
        if self.env_source == EnvSrc.MiniGrid:
            venv = make_vec_env(
                self.env_name,
                wrapper_class=wrapper_class,
                vec_env_cls=CustomSubprocVecEnv,
                n_envs=self.num_processes,
                monitor_dir=self.log_dir,
            )
        elif self.env_source == EnvSrc.ProcGen:
            venv = ProcgenEnv(
                num_envs=self.num_processes,
                env_name=self.env_name,
                rand_seed=self.run_id,
                num_threads=self.procgen_num_threads,
                distribution_mode=self.procgen_mode,
            )
            venv = VecMonitor(venv=venv)
        else:
            raise NotImplementedError

        if (self.record_video == 2) or \
                (self.record_video == 1 and self.run_id == 0):
            _trigger = lambda x: x > 0 and x % (self.n_steps * self.rec_interval) == 0
            venv = VecVideoRecorder(
                venv,
                os.path.join(self.log_dir, 'videos'),
                record_video_trigger=_trigger,
                video_length=self.video_length,
            )
        return venv

    def get_callbacks(self):
        if self.group_name is not None:
            callbacks = CallbackList([
                WandbCallback(
                    gradient_save_freq=50,
                    verbose=1,
                )])
        else:
            callbacks = CallbackList([])
        return callbacks

    def get_optimizer(self):
        if self.optimizer.lower() == 'adam':
            optimizer_class = th.optim.Adam
            optimizer_kwargs = dict(
                eps=self.optim_eps,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        elif self.optimizer.lower() == 'rmsprop':
            optimizer_class = th.optim.RMSprop
            optimizer_kwargs = dict(
                eps=self.optim_eps,
                alpha=self.rmsprop_alpha,
                momentum=self.rmsprop_momentum,
            )
        else:
            raise NotImplementedError
        return optimizer_class, optimizer_kwargs

    def get_activation_fn(self):
        if self.activation_fn.lower() == 'relu':
            activation_fn = nn.ReLU
        elif self.activation_fn.lower() == 'gelu':
            activation_fn = nn.GELU
        elif self.activation_fn.lower() == 'elu':
            activation_fn = nn.ELU
        else:
            raise NotImplementedError

        if self.cnn_activation_fn.lower() == 'relu':
            cnn_activation_fn = nn.ReLU
        elif self.cnn_activation_fn.lower() == 'gelu':
            cnn_activation_fn = nn.GELU
        elif self.cnn_activation_fn.lower() == 'elu':
            cnn_activation_fn = nn.ELU
        else:
            raise NotImplementedError
        return activation_fn, cnn_activation_fn

    def cast_enum_values(self):
        self.policy_cnn_norm = NormType.get_enum_norm_type(self.policy_cnn_norm)
        self.policy_mlp_norm = NormType.get_enum_norm_type(self.policy_mlp_norm)
        self.policy_gru_norm = NormType.get_enum_norm_type(self.policy_gru_norm)

        self.model_cnn_norm = NormType.get_enum_norm_type(self.model_cnn_norm)
        self.model_mlp_norm = NormType.get_enum_norm_type(self.model_mlp_norm)
        self.model_gru_norm = NormType.get_enum_norm_type(self.model_gru_norm)

        self.int_rew_source = ModelType.get_enum_model_type(self.int_rew_source)
        if self.int_rew_source == ModelType.DEIR and not self.use_model_rnn:
            print('\nWARNING: Running DEIR without RNNs\n')
        if self.int_rew_source in [ModelType.DEIR, ModelType.PlainDiscriminator]:
            assert self.n_steps * self.num_processes >= self.batch_size

    def get_cnn_kwargs(self, cnn_activation_fn=nn.ReLU):
        features_extractor_common_kwargs = dict(
            features_dim=self.features_dim,
            activation_fn=cnn_activation_fn,
            model_type=self.policy_cnn_type,
        )

        model_features_extractor_common_kwargs = dict(
            features_dim=self.model_features_dim,
            activation_fn=cnn_activation_fn,
            model_type=self.model_cnn_type,
        )

        if self.policy_cnn_norm == NormType.BatchNorm:
            policy_features_extractor_class = BatchNormCnnFeaturesExtractor
        elif self.policy_cnn_norm == NormType.LayerNorm:
            policy_features_extractor_class = LayerNormCnnFeaturesExtractor
        elif self.policy_cnn_norm == NormType.NoNorm:
            policy_features_extractor_class = CnnFeaturesExtractor
        else:
            raise ValueError

        if self.model_cnn_norm == NormType.BatchNorm:
            model_cnn_features_extractor_class = BatchNormCnnFeaturesExtractor
        elif self.model_cnn_norm == NormType.LayerNorm:
            model_cnn_features_extractor_class = LayerNormCnnFeaturesExtractor
        elif self.model_cnn_norm == NormType.NoNorm:
            model_cnn_features_extractor_class = CnnFeaturesExtractor
        else:
            raise ValueError

        return policy_features_extractor_class, \
            features_extractor_common_kwargs, \
            model_cnn_features_extractor_class, \
            model_features_extractor_common_kwargs





