from typing import NamedTuple, Optional
import torch as th

class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    new_observations: th.Tensor
    last_policy_mems: th.Tensor
    last_model_mems: th.Tensor
    episode_starts: th.Tensor
    episode_dones: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    curr_key_status: Optional[th.Tensor]
    curr_door_status: Optional[th.Tensor]
    curr_target_dists: Optional[th.Tensor]
