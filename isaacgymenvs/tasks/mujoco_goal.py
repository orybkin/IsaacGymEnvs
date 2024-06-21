# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
from gym.spaces import Box
import gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium_robotics.envs.fetch import reach

class MujocoGoal:

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.max_episode_length = 50
        self.n_envs = n_envs = cfg['env']['numEnvs']
        import gymnasium
        self.env = SubprocVecEnv([lambda: Monitor(gymnasium.make(cfg['task_name'], max_episode_steps=50))] * n_envs, 'fork')
        # self.env = DummyVecEnv([lambda: Monitor(gymnasium.make(cfg['task_name'], max_episode_steps=50))] * n_envs)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        obs = self.observation_space
        total_space = obs['observation'].shape[0] + obs['achieved_goal'].shape[0] + obs['desired_goal'].shape[0]
        self.observation_space = Box(self.observation_space['observation'].low[0], self.observation_space['observation'].high[0], (total_space,))
        self.action_space = Box(self.action_space.low, self.action_space.high, self.action_space.shape)
        self.num_states = 0
        self.render_every_episodes = 10000000
        self.max_pix = 1
        self.achieved_idx = range(self.observation_space.shape[0] - self.env.observation_space['desired_goal'].shape[0] - self.env.observation_space['achieved_goal'].shape[0], self.observation_space.shape[0] - self.env.observation_space['desired_goal'].shape[0])
        self.desired_idx = range(self.observation_space.shape[0] - self.env.observation_space['desired_goal'].shape[0], self.observation_space.shape[0])
        self.target_name = 'achieved_goal'
        self.curr_step = 1

    def cat(self, obs):
        # arr = list(np.concatenate([i['observation'], i['achieved_goal'], i['desired_goal']], axis=0) for i in obs)
        return np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']], axis=1)
    
    def reset(self):
        self.curr_step = 1
        return self.cat(self.env.reset())
    
    def reset_idx(self):
        return self.reset()

    def step(self, actions):
        self.curr_step += 1
        obs, r, truncated, info = self.env.step(actions.cpu().numpy())
        terminated = np.zeros_like(truncated)
        truncated = np.ones_like(truncated) * (self.curr_step % self.max_episode_length == 0)
        # if truncated.all() or terminated.all():
        #     self.reset()
        success = [i['is_success'] for i in info]
        return self.cat(obs), r, np.array(terminated), np.array(truncated), {'episodic': {'success': torch.Tensor(success)}}
    
    def compute_reward_stateless(self, d):
        rewards = torch.Tensor(self.env.env_method("compute_reward", d['goal_pos'].cpu(), d['achieved_goal'].cpu(), {}, indices=[0]))[0].cuda()
        return rewards
