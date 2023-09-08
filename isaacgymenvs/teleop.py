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

import numpy as np
import pygame
from pygame.locals import KEYDOWN, QUIT

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

pygame.init()
screen = pygame.display.set_mode((400, 300))

char_to_action = {
    "a": torch.Tensor([0, -1, 0, 0]),
    "s": torch.Tensor([1, 0, 0, 0]),
    "d": torch.Tensor([0, 1, 0, 0]),
    "w": torch.Tensor([-1, 0, 0, 0]),
    "q": torch.Tensor([1, -1, 0, 0]),
    "e": torch.Tensor([-1, -1, 0, 0]),
    "z": torch.Tensor([1, 1, 0, 0]),
    "c": torch.Tensor([-1, 1, 0, 0]),
    "k": torch.Tensor([0, 0, 1, 0]),
    "j": torch.Tensor([0, 0, -1, 0]),
    "h": "close",
    "l": "open",
    "x": "toggle",
    "r": "reset",
    "p": "put obj in hand",
}

# transform = gymapi.Transform()
# transform.p = (x,y,z)
# transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
# gym.set_camera_transform(camera_handle, env, transform)


envs.gym.viewer_camera_look_at(envs.viewer, envs.envs[0], gymapi.Vec3(0.2,0,1.3), gymapi.Vec3(0, 0, 1))

try:
    while True:
        envs.reset()
        for _ in range(5000):
            action = torch.zeros(7)
            event_happened = False
            while not event_happened:
                for event in pygame.event.get():
                    event_happened = True
                    if event.type == QUIT:
                        sys.exit()
                    if event.type == KEYDOWN:
                        char = event.dict["key"]
                        new_action = char_to_action.get(chr(char), None)
                        if new_action == "toggle":
                            lock_action = not lock_action
                        elif new_action == "reset":
                            done = True
                        elif new_action == "close":
                            action[3] = 1
                        elif new_action == "open":
                            action[3] = -1
                        elif new_action is not None:
                            action[:3] = new_action[:3]
                        else:
                            action = torch.zeros(7)
                        print(action)
            # print(envs.gym.get_camera_transform())
            # t = envs.gym.get_viewer_camera_transform(envs.viewer, envs.envs[0])
            # print(t.p, t.r)
            obs, reward, done, info = envs.step(action[None])
except KeyboardInterrupt:
    print("Keyboard interrupt, shutting down.\n")






