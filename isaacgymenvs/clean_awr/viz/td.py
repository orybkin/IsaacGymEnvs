import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from isaacgymenvs.clean_awr.viz.utils import add_cubes_to_viz

class TemporalDistanceVisualizer:
    def __init__(self, agent):
        self.agent = agent
        self.config = self.agent.config
        self.device = agent.device
        self.env = agent.vec_env.env
    
    @torch.no_grad()
    def _td_probs_sliced(self, x, y, n_slices=None, classifier_selection=None):
        """From awr.py"""
        obs = torch.cat([x, y], dim=1)
        if n_slices is None: n_slices = min(16, obs.shape[0])
        obs = obs.reshape([n_slices, -1, obs.shape[1]])
        logits = [self.agent.temporal_distance(obs[i], classifier_selection=classifier_selection)[0] for i in range(n_slices)]
        logits = torch.cat(logits, 0)
        probs = torch.softmax(logits, dim=1)
        return probs
    
    def _get_obs(self):
        return self.agent.experience_buffer.tensor_dict['obses']
    
    def hist(self, pred, target=None):
        if target is None:
            sns.histplot(data=pred.detach().cpu().numpy(), bins=self.agent.td_output_size)
            plt.gca().set_xlabel('pred')
            plt.xlim(-1, self.agent.td_output_size)
        else:
            target = target.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            df = pd.DataFrame({'target': target, 'pred': pred})
            df['counts'] = df.groupby(['target', 'pred'])['target'].transform('count')
            j = sns.jointplot(x=target, y=pred, marginal_kws=dict(bins=np.arange(self.agent.td_output_size+1)-0.5))
            j.ax_joint.cla()
            sns.scatterplot(x='target', y='pred', data=df, hue='counts', palette='Blues', ax=j.ax_joint, legend=False)
            j.ax_marg_x.set_xlim(-1, self.agent.td_output_size)
            j.ax_marg_y.set_ylim(-1, self.agent.td_output_size)
            plt.gca().set(xlabel='target', ylabel='pred')
        res = wandb.Image(plt.gcf())
        plt.close('all')
        return res
    
    @torch.no_grad()
    def successful_traj(self, success_buffer):
        obs = self._get_obs()
        env_num = success_buffer.float().argmax()
        traj = obs[:, env_num, :]
        initial_state = traj[0, self.agent.td_idx].repeat(obs.shape[0], 1)
        achieved = traj[:, self.env.achieved_idx]
        distances = self.agent.run_td_in_slices(initial_state, achieved)
        distances = distances.flatten().cpu().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.scatter(np.arange(obs.shape[0]), distances, zorder=10)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        start = max(xlim[0], ylim[0])
        end = min(xlim[1], ylim[1])
        plt.plot([start, end], [start, end], color='red', linestyle='--')
        
        res = wandb.Image(fig)
        plt.close('all')
        return res
    
    @torch.no_grad()
    def _goal_heatmap(self, traj, goal_pos, grid_obs, grid_res, bounds, plot_kw, classifier_selection):
        goal = goal_pos[None, :].repeat(grid_res**2, 1)
        distances = self.agent.run_td_in_slices(
            grid_obs[:, :, self.agent.td_idx].flatten(0, 1), 
            goal, 
            classifier_selection=classifier_selection)
        distances = distances.reshape(grid_res, grid_res).cpu().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(np.flip(distances, axis=0), vmin=0, vmax=self.agent.td_output_size-1, **plot_kw)
        fig.colorbar(im, ax=ax)
        plt.plot(goal_pos[0].item(), goal_pos[1].item(), 'go', markersize=10)
        
        traj = traj[:, None, :]
        segments = np.concatenate([traj[:-1], traj[1:]], axis=1)
        norm = Normalize(vmin=0, vmax=len(traj)-1)
        colors = cm.plasma(norm(np.arange(len(traj)-1)))
        lc = LineCollection(segments, colors=colors, zorder=10)
        ax.add_collection(lc)
        ax.set_xlim(*bounds)
        ax.set_ylim(*bounds)
        plt.scatter(*traj[1:].squeeze(1).T, color=colors, s=5, zorder=20)
        
        img = wandb.Image(fig)
        plt.close('all')
        return img
        
    @torch.no_grad()
    def _neg_prob_heatmap(self, obs, step_num, grid_obs, grid_res, plot_kw):
        env_num = np.random.randint(obs.shape[1])  # hopefully not same as env_num
        goal_pos = obs[step_num, env_num, self.env.desired_idx]
        goal = goal_pos[None, :].repeat(grid_res**2, 1)
        
        probs = self._td_probs_sliced(grid_obs[:, :, self.agent.td_idx].flatten(0, 1), goal)
        probs = probs[:, -1].reshape(grid_res, grid_res).cpu().numpy()
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        if not self.config['temporal_distance']['regression']:
            im = ax.imshow(np.flip(probs, axis=0), **plot_kw)
            fig.colorbar(im, ax=ax)
            plt.plot(goal_pos[0].item(), goal_pos[1].item(), 'go', markersize=10)
        
        img = wandb.Image(fig)
        plt.close('all')
        return img
    
    def heatmap(self, grid_res: int):
        """
        Returns a dictionary {'goal': Image, 'neg': Image}
        
        goal: Samples a step and env from obs as a goal. Modifying x and y components
              (but leaving all else equal), make a heatmap with resulting distances.
        neg:  Samples a negative goal and makes heatmap of corresponding logits.
        """
        obs = self._get_obs()
        step_num = np.random.randint(obs.shape[0])
        env_num = np.random.randint(obs.shape[1])
        traj = obs[:, env_num, self.env.achieved_idx[:2]].cpu().numpy()
        
        goal_pos = obs[step_num, env_num, self.env.desired_idx]
        grid_obs = obs[step_num, env_num, :][None, None, :].repeat(grid_res, grid_res, 1)
        coords = torch.linspace(-self.env.goal_position_noise, self.env.goal_position_noise, steps=grid_res+1, device=self.device)
        coords = (coords[:-1] + coords[1:]) / 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        grid_obs[:, :, self.env.achieved_idx[0]] = x
        grid_obs[:, :, self.env.achieved_idx[1]] = y
        
        bounds = [-self.env.goal_position_noise, self.env.goal_position_noise]
        plot_kw = dict(cmap='Greys', interpolation='none', extent=[*bounds, *bounds])
        img_goal_mode = self._goal_heatmap(traj, goal_pos, grid_obs, grid_res, bounds, plot_kw, classifier_selection='mode')
        img_goal_mean = self._goal_heatmap(traj, goal_pos, grid_obs, grid_res, bounds, plot_kw, classifier_selection='mean')
        img_neg = self._neg_prob_heatmap(obs, step_num, grid_obs, grid_res, plot_kw)
        
        res_dict = {'goal_mode': img_goal_mode, 'goal_mean': img_goal_mean, 'neg': img_neg}
        plt.close('all')
        return res_dict
    
    def viz_all(self, success_buffer):
        obs = self._get_obs()
        pos = obs[:, :, self.env.achieved_idx]
        did_move = ~torch.all(torch.all(torch.isclose(pos, pos.flip(0)), dim=2), dim=0)
        success_criterion = success_buffer & did_move
        heatmap_dict = self.heatmap(grid_res=16)
        for k, img in heatmap_dict.items():
            wandb.log({f'temporal_distance/{k}_viz': [img]})
        if torch.any(success_criterion):
            img = self.successful_traj(success_criterion)
            wandb.log({'temporal_distance/traj_viz': [img]})
