import numpy as np
import matplotlib.pyplot as plt
import torch

import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from isaacgymenvs.clean_awr.viz.utils import add_cubes_to_viz

@torch.no_grad()
def viz(agent, grid_res: int):
    device = agent.device
    env = agent.vec_env.env
    obs = agent.experience_buffer.tensor_dict['obses']
    
    step_num = np.random.randint(obs.shape[0])
    env_num = np.random.randint(obs.shape[1])
    traj = obs[:, env_num, env.achieved_idx[:2]].cpu().numpy()
    goal_pos = obs[step_num, env_num, env.desired_idx]
    goal = goal_pos[None, :].repeat(grid_res**2, 1)
    grid_obs = obs[step_num, env_num, :][None, None, :].repeat(grid_res, grid_res, 1)
    coords = torch.linspace(-env.goal_position_noise, env.goal_position_noise, steps=grid_res+1, device=device)
    coords = (coords[:-1] + coords[1:]) / 2
    grid_obs[:, :, env.achieved_idx[0]], grid_obs[:, :, env.achieved_idx[1]] = torch.meshgrid(coords, coords, indexing='ij')
    distances = agent.run_td_in_slices(grid_obs[:, :, agent.td_idx].flatten(0, 1), goal)
    distances = distances.reshape(grid_res, grid_res).cpu().numpy()
    
    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    kwargs = dict(
        cmap='Greys', 
        interpolation='none',
        extent=[-env.goal_position_noise, env.goal_position_noise, 
                -env.goal_position_noise, env.goal_position_noise]
    )
    im = ax.imshow(distances, vmin=0, vmax=agent.td_output_size-1, **kwargs)
    fig.colorbar(im, ax=ax)
    plt.plot(goal_pos[0].item(), goal_pos[1].item(), 'go', markersize=10)
    # add_cubes_to_viz(obs, env, ax)
    
    traj = traj[:, None, :]
    segments = np.concatenate([traj[:-1], traj[1:]], axis=1)
    norm = Normalize(vmin=0, vmax=len(traj)-1)
    colors = cm.plasma(norm(np.arange(len(traj)-1)))
    lc = LineCollection(segments, colors=colors, zorder=10)
    ax.add_collection(lc)
    
    return fig