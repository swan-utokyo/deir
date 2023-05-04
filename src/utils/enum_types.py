from enum import Enum

from torch import nn


class ModelType(Enum):
    NoModel = 0
    ICM = 1  # Forward + Inverse
    RND = 2
    NGU = 3
    NovelD = 4  # Inverse
    DEIR = 5
    PlainForward = 6
    PlainInverse = 7
    PlainDiscriminator = 8

    @staticmethod
    def get_enum_model_type(model_type):
        if isinstance(model_type, ModelType):
            return model_type
        if isinstance(model_type, str):
            model_type = model_type.strip().lower()
            if model_type == "icm":
                return ModelType.ICM
            elif model_type == "rnd":
                return ModelType.RND
            elif model_type == "ngu":
                return ModelType.NGU
            elif model_type == "noveld":
                return ModelType.NovelD
            elif model_type == "deir":
                return ModelType.DEIR
            elif model_type == "plainforward":
                return ModelType.PlainForward
            elif model_type == "plaininverse":
                return ModelType.PlainInverse
            elif model_type == "plaindiscriminator":
                return ModelType.PlainDiscriminator
            else:
                return ModelType.NoModel
        raise ValueError


class NormType(Enum):
    NoNorm = 0
    BatchNorm = 1
    LayerNorm = 2

    @staticmethod
    def get_enum_norm_type(norm_type):
        if isinstance(norm_type, NormType):
            return norm_type
        if isinstance(norm_type, str):
            norm_type = norm_type.strip().lower()
            if norm_type == 'batchnorm':
                return NormType.BatchNorm
            if norm_type == 'layernorm':
                return NormType.LayerNorm
            if norm_type == 'nonorm':
                return NormType.NoNorm
        raise ValueError

    @staticmethod
    def get_norm_layer_1d(norm_type, fetures_dim, momentum=0.1):
        norm_type = NormType.get_enum_norm_type(norm_type)
        if norm_type == NormType.BatchNorm:
            return nn.BatchNorm1d(fetures_dim, momentum=momentum)
        if norm_type == NormType.LayerNorm:
            return nn.LayerNorm(fetures_dim)
        if norm_type == NormType.NoNorm:
            return nn.Identity()
        raise NotImplementedError

    @staticmethod
    def get_norm_layer_2d(norm_type, n_channels, n_size, momentum=0.1):
        norm_type = NormType.get_enum_norm_type(norm_type)
        if norm_type == NormType.BatchNorm:
            return nn.BatchNorm2d(n_channels, momentum=momentum)
        if norm_type == NormType.LayerNorm:
            return nn.LayerNorm([n_channels, n_size, n_size])
        if norm_type == NormType.NoNorm:
            return nn.Identity()
        raise NotImplementedError


class EnvSrc(Enum):
    MiniGrid = 0
    ProcGen = 1

    @staticmethod
    def get_enum_env_src(env_src):
        if isinstance(env_src, EnvSrc):
            return env_src
        if isinstance(env_src, str):
            env_src = env_src.strip().lower()
            if env_src == 'minigrid':
                return EnvSrc.MiniGrid
            if env_src == 'procgen':
                return EnvSrc.ProcGen
        raise ValueError
