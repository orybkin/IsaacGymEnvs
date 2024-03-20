# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

import os
import sys

# noinspection PyUnresolvedReferences
import isaacgym  # pylint: disable=unused-import
import torch
from isaacgym import gymapi

sys.path.append(os.path.join(os.path.dirname(__file__), "IsaacGymEnvs"))
import isaacgymenvs
from isaacgymenvs.tasks.base import vec_task

# Create the environment and step the simulation as normal
device = "cuda"  # or "cpu"
num_env = 64
rl_device = device
envs = isaacgymenvs.make(
    seed=0,
    task="FrankaPushing",
    num_envs=num_env,
    sim_device=device,
    rl_device=device,
    graphics_device_id=0,
    headless=False,  # Need to be False if even you are using a headless server.
)

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)

try:
    while True:
        envs.reset()
        for _ in range(10):
            a = torch.rand((num_env,) + envs.action_space.shape, device=device)
            a = torch.zeros_like(a)
            a[:, 0] = 1 # forward 
            # a[:, 1] = 1 # left
            # a[:, 2] = 1 # up
            # a[:, 3] = 1 # up
            # import pdb; pdb.set_trace()
            obs, reward, terminated, truncated, info = envs.step(a)
            print(obs['obs'][0, :3])
        import pdb; pdb.set_trace()
except KeyboardInterrupt:
    print("Keyboard interrupt, shutting down.\n")
