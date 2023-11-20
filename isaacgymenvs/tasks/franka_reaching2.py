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

import numpy as np
import os
import torch
import math

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.franka_pushing import FrankaPushing



class FrankaReaching2(FrankaPushing):

    def __init__(self, *args, **kwargs):
        self.pushing_like = kwargs['cfg']['env'].get('pushing_like', True)
        super().__init__(*args, **kwargs)
        self.target_idx = []
        self.target_idx = [0, 1, 2]
        self.target_name = 'eef_pos'


    def compute_franka_reward(self, states):
        ## type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, float, float, float) -> Tuple[Tensor, Tensor]

        # distance from cube to the goal
        d = torch.norm(states["goal_pos"] - states["eef_pos"], dim=-1)
        if self.dist_reward_threshold:
            dist_reward = torch.where(d < self.dist_reward_threshold, torch.ones_like(d), torch.zeros_like(d))
        else:
            dist_reward = 1 - torch.tanh(self.dist_reward_dropoff * d)

        return self.dist_reward_scale * dist_reward

    
    def _reset_goal_state(self, env_ids):
        """

        Args:
            env_ids (tensor or None): Specific environments to reset cube for
        """
        # If env_ids is None, we reset all the envs
        # TODO randomize Z
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_goal_state = torch.zeros(num_resets, 13, device=self.device)

        if self.pushing_like:
            # Sampling is "centered" around middle of table
            centered_goal_xyz_state = torch.tensor(self._table_surface_pos[:3], device=self.device, dtype=torch.float32)
            # Set z value, which is fixed height
            centered_goal_xyz_state[2] = centered_goal_xyz_state[2] + 0.05
        else:
            # Initial arm position
            centered_goal_xyz_state = torch.tensor([0, 0, 1.21], device=self.device, dtype=torch.float32)

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_goal_state[:, 6] = 1.0

        sampled_goal_state[:, :2] = centered_goal_xyz_state[:2].unsqueeze(0) + \
                                            2.0 * self.goal_position_noise * (
                                                    torch.rand(num_resets, 2, device=self.device) - 0.5)
        sampled_goal_state[:, 2] = centered_goal_xyz_state[2]

        if self.test:
            sampled_goal_state[:, 0] = centered_goal_xyz_state[0] + 0.08
            sampled_goal_state[:, 1] = centered_goal_xyz_state[1] - 0.08
            sampled_goal_state[:, 2] = centered_goal_xyz_state[2] 

        # Lastly, set these sampled values as the new init state
        self._goal_state[env_ids, :] = sampled_goal_state

