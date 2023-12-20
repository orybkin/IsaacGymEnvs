import copy
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from rl_games.common import experience


class HERReplayBuffer(experience.VectorizedReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.

    """
    def __init__(self, obs_shape, action_shape, capacity, n_envs, device, env, her_ratio=0.8):
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
        assert capacity == self.n_steps * n_envs

        self.obses = torch.empty((n_steps, n_envs, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_obses = torch.empty((n_steps, n_envs, *obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((n_steps, n_envs, *action_shape), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((n_steps, n_envs, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((n_steps, n_envs, 1), dtype=torch.bool, device=self.device)
        self.steps_to_go = torch.empty((n_steps, n_envs), dtype=torch.int64, device=self.device)
        self.data = [self.obses, self.actions, self.rewards, self.next_obses, self.dones]
        # I am going to track episode length as follows. Log current episode start. Every step, increment
        # steps_to_go starting from current episode start
        self.current_ep_start = torch.zeros((n_envs,), dtype=torch.long, device=self.device)

        self.idx = 0
        self.full = False
        self.her_ratio = her_ratio
        self.env = env

    def add(self, obs, action, reward, next_obs, done):
        steps_to_go = 1 - done
        for x, y in zip(self.data, [obs, action, reward, next_obs, done, steps_to_go]):
            x[self.idx] = y

        self.current_ep_start = torch.where(done[:, 0] == 1, self.idx + 1, self.current_ep_start)
        # Increment steps to go starting from current_ep_start
        # create a mask with 1s in the range between self.current_ep_start and self.idx
        mask = torch.ones_like(self.steps_to_go)
        ids = torch.arange(self.n_steps, device=self.device)[:, None]
        mask[ids < self.current_ep_start[None]] = 0
        mask[self.idx + 1:] = 0  
        self.steps_to_go += mask

        self.idx += 1 % self.n_steps
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
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
        current_size = self.n_steps * self.n_envs if self.full else self.idx * self.n_envs
        idxs = torch.randint(0, current_size, (batch_size,), device=self.device)
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_idxs, real_idxs = np.split(idxs, [nb_virtual])

        real_data = list(l.flatten(0,1)[real_idxs] for l in self.data)
        virtual_data = self.sample_virtual(virtual_idxs)
        data = [torch.cat([real, virtual], dim=0) for real, virtual in zip(real_data, virtual_data)]
        return data
    
    def sample_virtual(self, idxs):

        obs, action, reward, next_obs, done = list(l.flatten(0,1)[idxs] for l in self.data)

        max_steps_to_go = self.steps_to_go.flatten(0,1)[idxs]
        steps_to_go = randint(max_steps_to_go + 1)
        goal_idxs = idxs + steps_to_go * self.n_envs
        goal = self.next_obses.flatten(0,1)[goal_idxs][:, self.env.target_idx] 

        target_pos = next_obs[:, self.env.target_idx]
        obs[:, 7:10] = goal
        next_obs[:, 7:10] = goal
        reward = self.env.compute_franka_reward({'goal_pos': goal, self.env.target_name: target_pos})[:, None]

        return obs, action, reward, next_obs, done
    
def randint(high): return torch.randint(2**63 - 1, size=high.shape, device=high.device) % high

