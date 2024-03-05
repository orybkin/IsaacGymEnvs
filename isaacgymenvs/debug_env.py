""" 
Debug headless camera rendering
Author: Oleg 
"""


import imageio
from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np

from tasks.cubes_debug import CubesDebug

def mat33_to_np(mat):
    return np.array([
        [mat.x.x, mat.x.y, mat.x.z],
        [mat.y.x, mat.y.y, mat.y.z],
        [mat.z.x, mat.z.y, mat.z.z]
    ])

env = CubesDebug(1)
env.reset_idx()
env.refresh()

# breakpoint()

env.render("render_og1.png")
og1_states = env.states['cube0_pos'].cpu().numpy()
print(og1_states)

env.freeze_cubes()
env.refresh()
env.render("render_new1.png")

for _ in range(10):
    env.step()

env.freeze_cubes()
env.refresh()
env.render("render_og2.png")
og2_states = env.states['cube0_pos'].cpu().numpy()
print(og2_states)

for _ in range(10):
    env.step()
    env.refresh()
env.render("render_og3.png")
og3_states = env.states['cube0_pos'].cpu().numpy()
print(og3_states)

# random refresh and physics simulation steps should do something
assert not np.all(np.isclose(og1_states, og2_states))
assert not np.all(np.isclose(og3_states, og2_states))