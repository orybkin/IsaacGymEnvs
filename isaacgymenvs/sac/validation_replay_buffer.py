import copy
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from rl_games.common import experience
from isaacgymenvs.sac.her_replay_buffer import HERReplayBuffer

class ValidationHERReplayBuffer(HERReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.

    """
    def __init__(self, obs_shape, action_shape, capacity, n_envs, device, env, rewards_shaper=None, her_ratio=0.8, random_ratio=0.0, validation_ratio=0.0):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device
        self.n_steps = n_steps = capacity // n_envs
        self.n_envs = n_envs
        self.val_envs = int(n_envs * validation_ratio)
        self.train_envs = n_envs - self.val_envs
        assert capacity == self.n_steps * n_envs
        self.dtype = torch.float32

        self.obses = torch.empty((n_steps, n_envs, *obs_shape), dtype=self.dtype, device=self.device)
        self.next_obses = torch.empty((n_steps, n_envs, *obs_shape), dtype=self.dtype, device=self.device)
        self.actions = torch.empty((n_steps, n_envs, *action_shape), dtype=self.dtype, device=self.device)
        self.rewards = torch.empty((n_steps, n_envs, 1), dtype=self.dtype, device=self.device)
        self.dones = torch.empty((n_steps, n_envs, 1), dtype=torch.bool, device=self.device)
        self.steps_to_go = torch.empty((n_steps, n_envs), dtype=torch.int64, device=self.device)
        self.data = [self.obses, self.actions, self.rewards, self.next_obses, self.dones, self.steps_to_go]
        # I am going to track episode length as follows. Log current episode start. Every step, increment
        # steps_to_go starting from current episode start
        self.current_ep_start = torch.zeros((n_envs,), dtype=torch.long, device=self.device)

        self.idx = 0
        self.full = False
        self.her_ratio = her_ratio
        self.random_ratio = random_ratio
        self.env = env
        self.rewards_shaper = rewards_shaper

    @property
    def train_data(self):
        return [l[:, :self.train_envs] for l in self.data]
    
    @property
    def val_data(self):
        return [l[:, self.train_envs:] for l in self.data]

    def add(self, obs, action, reward, next_obs, terminated, done):
        steps_to_go = torch.zeros_like(done)[:, 0]
        for x, y in zip(self.data, [obs, action, reward, next_obs, terminated, steps_to_go]):
            if y.dtype == torch.float32: y = y.to(self.dtype)
            x[self.idx] = y

        # Get steps-to-go
        mask = torch.zeros_like(self.steps_to_go)
        ids = torch.arange(self.n_steps, device=self.device)[:, None]
        mask[(ids >= self.current_ep_start[None]) * (ids < self.idx)] = 1
        # overflow
        mask[(self.current_ep_start[None] > self.idx) * (ids < self.idx)] = 1
        mask[(self.current_ep_start[None] > self.idx) * (ids >= self.current_ep_start[None])] = 1
        self.steps_to_go += mask
        self.current_ep_start = torch.where(done[:, 0] == 1, (self.idx + 1) % self.n_steps, self.current_ep_start)

        self.idx = (self.idx + 1) % self.n_steps
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, validation=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        """
        if not validation:
            data = self.train_data
            n_envs = self.train_envs
        else:
            data = self.val_data
            n_envs = self.val_envs

        current_size = self.n_steps * n_envs if self.full else self.idx * n_envs
        idxs = torch.randint(0, current_size, (batch_size,), device=self.device)
        n_virtual = int(self.her_ratio * batch_size)
        virtual_idxs, real_idxs = np.split(idxs, [n_virtual])

        real_data = list(l.flatten(0,1)[real_idxs] for l in data)
        virtual_data = self.sample_virtual(data, virtual_idxs, n_envs)
        data = [torch.cat([real, virtual], dim=0) for real, virtual in zip(real_data, virtual_data)]
        for i in range(4): data[i] = data[i].to(torch.float32)

        return data
    
    def sample_virtual(self, data, idxs, n_envs):
        obs, action, reward, next_obs, done, steps_to_go = list(l.flatten(0,1)[idxs] for l in data)
        n_virtual = obs.shape[0]
        n_random = int(n_virtual * self.random_ratio / self.her_ratio)
        n_future = n_virtual - n_random

        # Future goals
        add_steps = randint(steps_to_go[:n_future] + 1)
        future_idxs = (idxs[:n_future] + add_steps * n_envs) % (self.n_steps * n_envs)

        # Random goals
        current_size = self.n_steps * n_envs if self.full else self.idx * n_envs
        random_idxs = torch.randint(0, current_size, (n_random,), device=self.device)

        # Relabel
        goal_idxs = torch.cat([future_idxs, random_idxs], 0)
        goal = data[3].flatten(0,1)[goal_idxs][:, self.env.target_idx] 
        target_pos = next_obs[:, self.env.target_idx]
        obs[:, 7:10] = goal
        next_obs[:, 7:10] = goal
        reward = self.env.compute_franka_reward({'goal_pos': goal, self.env.target_name: target_pos})[:, None]
        reward = self.rewards_shaper(reward)

        return obs, action, reward, next_obs, done
    
def randint(high): return torch.randint(2**63 - 1, size=high.shape, device=high.device) % high


if __name__ == "__main__":
    ## Test sampling
    buffer = ValidationHERReplayBuffer(obs_shape=(3,),
                                    action_shape=(1,),
                                    capacity=200,
                                    n_envs=10, 
                                    device='cpu', 
                                    env=None, 
                                    rewards_shaper=None, 
                                    her_ratio=0.8,
                                    random_ratio=0.2,
                                    validation_ratio=0.0)

    for i in range(20):
        # add 10 random tensor observations of batches of 10 
        buffer.add(torch.rand(10, 3, dtype=torch.float32), 
                    torch.rand(10, 1, dtype=torch.float32), 
                    torch.rand(10, 1, dtype=torch.float32), 
                    torch.rand(10, 3, dtype=torch.float32), 
                    (((i + 1) % 4) == 0) * torch.ones(10, 1, dtype=torch.float32))
        print(buffer.steps_to_go[:, 0])
        
    # buffer.sample(10)

