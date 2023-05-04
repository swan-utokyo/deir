import warnings
from typing import Callable, List, Optional, Union, Sequence, Dict

import gym
import numpy as np
from numpy import ndarray
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import tile_images, VecEnvStepReturn
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs


class CustomSubprocVecEnv(SubprocVecEnv):

    def __init__(self, 
                 env_fns: List[Callable[[], gym.Env]], 
                 start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        self.can_see_walls = True
        self.image_noise_scale = 0.0
        self.image_rng = None  # to be initialized with run id in ppo_rollout.py

    def set_seeds(self, seeds: List[int] = None) -> List[Union[None, int]]:
        self.seeds = seeds
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", int(seeds[idx])))
        return [remote.recv() for remote in self.remotes]

    def get_seeds(self) -> List[Union[None, int]]:
        return self.seeds

    def send_reset(self, env_id: int) -> None:
        self.remotes[env_id].send(("reset", None))

    def invisibilize_obstacles(self, obs):
        # Algorithm A5 in the Technical Appendix
        # For MiniGrid envs only
        obs = np.copy(obs)
        for r in range(len(obs[0])):
            for c in range(len(obs[0][r])):
                # The color of Walls is grey
                # See https://github.com/Farama-Foundation/gym-minigrid/blob/20384cfa59d7edb058e8dbd02e1e107afd1e245d/gym_minigrid/minigrid.py#L215-L223
                # COLOR_TO_IDX['grey']: 5
                if obs[1][r][c] == 5 and 0 <= obs[0][r][c] <= 2:
                    obs[1][r][c] = 0
                # OBJECT_TO_IDX[0,1,2]: 'unseen', 'empty', 'wall'
                if 0 <= obs[0][r][c] <= 2:
                    obs[0][r][c] = 0
        return obs

    def add_noise(self, obs):
        # Algorithm A4 in the Technical Appendix
        # Add noise to observations
        obs = obs.astype(np.float64)
        obs_noise = self.image_rng.normal(loc=0.0, scale=self.image_noise_scale, size=obs.shape)
        return obs + obs_noise

    def recv_obs(self, env_id: int) -> ndarray:
        obs = VecTransposeImage.transpose_image(self.remotes[env_id].recv())
        if not self.can_see_walls:
            obs = self.invisibilize_obstacles(obs)
        if self.image_noise_scale > 0:
            obs = self.add_noise(obs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_arr, rews, dones, infos = zip(*results)
        obs_arr = _flatten_obs(obs_arr, self.observation_space).astype(np.float64)
        for idx in range(len(obs_arr)):
            if not self.can_see_walls:
                obs_arr[idx] = self.invisibilize_obstacles(obs_arr[idx])
            if self.image_noise_scale > 0:
                obs_arr[idx] = self.add_noise(obs_arr[idx])
        return obs_arr, np.stack(rews), np.stack(dones), infos

    def get_first_image(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes[:1]:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes[:1]]
        return imgs

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        try:
            # imgs = self.get_images()
            imgs = self.get_first_image()
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs[:1])
        if mode == "human":
            import cv2  # pytype:disable=import-error
            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")