import numpy as np
import torch as th

from torch import nn
from typing import Optional, Callable
from src.utils.enum_types import NormType, EnvSrc, ModelType


def bkdr_hash(inputs, seed=131, mask=0x7fffffff):
    """
    A simple deterministic hashing algorithm by
    Kernighan, Brian W., and Dennis M. Ritchie.
    "The C programming language." (2002).
    """
    if isinstance(inputs, np.ndarray):
        data = inputs.reshape(-1)
    else:
        data = inputs
    res = 0
    is_str = isinstance(data, str)
    get_val = lambda x: ord(val) if is_str else val
    for val in data:
        int_val = get_val(val)
        res = (res * seed) & mask
        res = (res + int_val) & mask
    return res


def normalize_rewards(norm_type, rewards, mean, std, eps=1e-5):
    """
    Normalize the input rewards using a specified normalization method (norm_type).
    [0] No normalization
    [1] Standardization per mini-batch
    [2] Standardization per rollout buffer
    [3] Standardization without subtracting the average reward
    """
    if norm_type <= 0:
        return rewards

    if norm_type == 1:
        # Standardization
        return (rewards - mean) / (std + eps)

    if norm_type == 2:
        # Min-max normalization
        min_int_rew = np.min(rewards)
        max_int_rew = np.max(rewards)
        mean_int_rew = (max_int_rew + min_int_rew) / 2
        return (rewards - mean_int_rew) / (max_int_rew - min_int_rew + eps)

    if norm_type == 3:
        # Standardization without subtracting the mean
        return rewards / (std + eps)


def set_random_seed(self, seed: Optional[int] = None) -> None:
    """
    From Stable Baslines 3.
    Set the seed of the pseudo-random generators
    (python, numpy, pytorch, gym, action_space)
    """
    if seed is None:
        return
    set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
    self.action_space.seed(seed)

    if self.env is not None:
        seed_method = getattr(self.env, "seed_method", None)
        if seed_method is not None:
            self.env.seed(seed)
    if self.eval_env is not None:
        seed_method = getattr(self.eval_env, "seed_method", None)
        if seed_method is not None:
            self.eval_env.seed(seed)


def init_module_with_name(n: str, m: nn.Module, fn: Callable[['Module'], None] = None) -> nn.Module:
    """
    Initialize the parameters of a neural network module using the hash of the module name
    as a random seed. The purpose of this feature is to ensure that experimentally adding or
    removing submodules does not affect the initialization of other modules.
    """
    def _reset_parameters(module: nn.Module) -> None:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()

    has_child = False
    for name, module in m.named_children():
        has_child = True
        init_module_with_name(n + '.' + name, module, fn)
    if not has_child:
        run_id = th.initial_seed()
        hash_val = bkdr_hash(n)
        hash_val = bkdr_hash([hash_val, run_id])
        th.manual_seed(hash_val)
        if fn is None:
            _reset_parameters(m)
        else:
            fn(m)
        th.manual_seed(run_id)
    return m
