import os
os.environ['MUJOCO_GL']='egl'
os.environ['MUJOCO_GL']='osmesa'
if 'SLURM_STEP_GPUS' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from collections import namedtuple
import argparse
from datetime import datetime

from awr.awr_original import RAWR
# import rlbase # get the debugger

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_envs', type=int, default=32)
parser.add_argument('--relabel_ratio', type=float, default=0.5)
parser.add_argument('--total_timesteps', type=int, default=1_000_000)
parser.add_argument('--learning_starts', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--ent_coef', type=str, default='5e-2')
parser.add_argument('--action_noise', type=float, default=0)
parser.add_argument('--temperature', type=float, default=0.2)
parser.add_argument('--max_grad_norm', type=float, default=0)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--n_steps', type=int, default=1024*32)
parser.add_argument('--buffer_size', type=int, default=1_000_000)
parser.add_argument('--gradient_steps', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024*8)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--env_name', type=str, default='FetchReach-v2')
parser.add_argument('--log_std_init', type=float, default=0.0)
parser.add_argument('--wandb_name', type=str, default='relabel')
parser.add_argument('--wandb_group', type=str, default='none')
args = parser.parse_args()

np.random.seed(args.seed)
import torch
torch.manual_seed(args.seed)
env_kwargs = dict(render_mode='rgb_array', width=128, height=128, max_episode_steps=50)
n_envs = args.n_envs
if n_envs > 1:
    env = SubprocVecEnv([lambda: Monitor(gym.make(args.env_name, **env_kwargs))] * n_envs, 'fork')

policy_config = args.__dict__.copy()
policy_config['n_steps'] = args.n_steps // n_envs
# policy_config['train_freq'] = 1
policy_config['action_noise'] = NormalActionNoise(mean=np.zeros_like(env.action_space.low), sigma=args.action_noise * (env.action_space.high - env.action_space.low))
policy_config['policy_kwargs'] = dict(log_std_init=args.log_std_init)
policy_config['policy_kwargs']['net_arch'] = [256, 256, 256]

wandb_config = policy_config.copy()
run = wandb.init(
    project="taskmaster_sb3",
    entity="oleh-rybkin",
    config=wandb_config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    name=args.wandb_name,
    group=args.wandb_group,
)
run_name = datetime.now().strftime("%Y%m%d-%H%M%S-") + run.name
callback = WandbCallback(model_save_path=f"models/{run_name}",verbose=2,)

model = RAWR("MultiInputPolicy", env, verbose=1, run_name=run_name, tensorboard_log=f"logs/{run_name}", **policy_config)
model.learn(total_timesteps=args.total_timesteps, callback=callback)
run.finish()