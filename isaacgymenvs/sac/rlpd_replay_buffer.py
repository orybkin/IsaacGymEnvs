import copy
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from rl_games.common import experience
from isaacgymenvs.sac.validation_replay_buffer import ValidationHERReplayBuffer

class RLPDReplayBuffer:
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.

    """
    def __init__(self, obs_shape, action_shape, capacity, n_envs, device, env, rewards_shaper=None, her_ratio=0.8, random_ratio=0.0, validation_ratio=0.0, precision='float32'):
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
        
        self.online_buffer = ValidationHERReplayBuffer(obs_shape, action_shape, capacity // 2, n_envs, device, env, rewards_shaper, her_ratio, random_ratio, validation_ratio, precision)
        self.offline_buffer = ValidationHERReplayBuffer(obs_shape, action_shape, capacity // 2, n_envs, device, env, rewards_shaper, her_ratio, random_ratio, validation_ratio, precision)

    def add(self, *args):
        self.online_buffer.add(*args)

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
        
        online_data = self.online_buffer.sample(batch_size // 2, validation)
        offline_data = self.offline_buffer.sample(batch_size // 2, validation)
        data = [torch.cat([on, off], dim=0) for on, off in zip(online_data, offline_data)]
        return data
    
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

