# About
This repository implements __Discriminative-model-based Episodic Intrinsic Reward (DEIR)__, an exploration method for reinforcement learning that has been found to be particularly effective in environments with stochasticity and partial observability.
More details can be found in the original paper "_DEIR: Efficient and Robust Exploration through Discriminative-Model-Based Episodic Intrinsic Rewards_" ([arXiv preprint](https://arxiv.org/abs/2304.10770), to appear at [IJCAI 2023](https://ijcai-23.org/)).

Our PPO implementation is based on [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3/tree/master). In case you are mainly interested in the implementation of DEIR, its major components can be found at [src/algo/intrinsic_rewards/deir.py](https://github.com/swan-utokyo/deir/blob/main/src/algo/intrinsic_rewards/deir.py).

### Demos
Video demos of DEIR: (1) [Introduction & MiniGrid demo](https://youtu.be/VDnkFuGKZC0), (2) [ProcGen demo](https://youtu.be/ljmiXwKq1u8).

<img src="https://github.com/swan-utokyo/deir/assets/132561960/79235738-7a41-4a66-afb8-dc15325f40f9" width="480">

# Usage
### Installation
```commandline
conda create -n deir python=3.9
conda activate deir 
git clone https://github.com/swan-utokyo/deir.git
cd deir
python3 -m pip install -r requirements.txt
```

### Train DEIR on MiniGrid
Run the below command in the root directory of this repository to train a DEIR agent in the standard _DoorKey-8x8_ (MiniGrid) environment.
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=DoorKey-8x8
```

### Train DEIR on MiniGrid with advanced settings
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=DoorKey-8x8-ViewSize-3x3 \
  --can_see_walls=0 \
  --image_noise_scale=0.1
```

### Sample output in DoorKey-8x8
```
run:  0  iters: 1  frames: 8192  rew: 0.389687  rollout: 9.281 sec  train: 1.919 sec
run:  0  iters: 2  frames: 16384  rew: 0.024355  rollout: 9.426 sec  train: 1.803 sec
run:  0  iters: 3  frames: 24576  rew: 0.032737  rollout: 8.622 sec  train: 1.766 sec
run:  0  iters: 4  frames: 32768  rew: 0.036805  rollout: 8.309 sec  train: 1.776 sec
run:  0  iters: 5  frames: 40960  rew: 0.043546  rollout: 8.370 sec  train: 1.768 sec
run:  0  iters: 6  frames: 49152  rew: 0.068045  rollout: 8.337 sec  train: 1.772 sec
run:  0  iters: 7  frames: 57344  rew: 0.112299  rollout: 8.441 sec  train: 1.754 sec
run:  0  iters: 8  frames: 65536  rew: 0.188911  rollout: 8.328 sec  train: 1.732 sec
run:  0  iters: 9  frames: 73728  rew: 0.303772  rollout: 8.354 sec  train: 1.741 sec
run:  0  iters: 10  frames: 81920  rew: 0.519742  rollout: 8.239 sec  train: 1.749 sec
run:  0  iters: 11  frames: 90112  rew: 0.659334  rollout: 8.324 sec  train: 1.777 sec
run:  0  iters: 12  frames: 98304  rew: 0.784067  rollout: 8.869 sec  train: 1.833 sec
run:  0  iters: 13  frames: 106496  rew: 0.844819  rollout: 9.068 sec  train: 1.740 sec
run:  0  iters: 14  frames: 114688  rew: 0.892450  rollout: 8.077 sec  train: 1.745 sec
run:  0  iters: 15  frames: 122880  rew: 0.908270  rollout: 7.873 sec  train: 1.738 sec
```

### Train DEIR on ProcGen
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=procgen --game_name=ninja --total_steps=100_000_000 \
  --num_processes=64 --n_steps=256 --batch_size=2048 \
  --n_epochs=3 --model_n_epochs=3 \
  --learning_rate=1e-4 --model_learning_rate=1e-4 \
  --policy_cnn_type=2 --features_dim=256 --latents_dim=256 \
  --model_cnn_type=1 --model_features_dim=64 --model_latents_dim=256 \
  --policy_cnn_norm=LayerNorm --policy_mlp_norm=NoNorm \
  --model_cnn_norm=LayerNorm --model_mlp_norm=NoNorm \
  --adv_norm=0 --adv_eps=1e-5 --adv_momentum=0.9
```

### Example of Training Baselines
Please note that the default value of each option in `src/train.py` is optimized for DEIR. For now, when training other methods, please use the corresponding hyperparameter values specified in Table A1 (in our [arXiv preprint](https://arxiv.org/abs/2304.10770)). An example is `--int_rew_coef=3e-2` and `--rnd_err_norm=0` in the below command.
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=NovelD \
  --env_source=minigrid \
  --game_name=DoorKey-8x8 \
  --int_rew_coef=3e-2 \
  --rnd_err_norm=0
```
