import isaacgym

import gym.spaces
from rl_games.common import env_configurations, vecenv
from rl_games.algos_torch import torch_ext
import os
import torch 
from torch import nn
from torch import optim
import numpy as np
import time
import gym
import math
import copy
from collections import defaultdict
from tensorboardX import SummaryWriter
from functools import partial

import isaacgymenvs
import wandb
from datetime import datetime
from pathlib import Path

from isaacgymenvs.clean_awr.awr_networks import AWRNetwork, _neglogp
from isaacgymenvs.clean_awr.temporal_distance import TemporalDistanceNetwork, TemporalDistanceDataset
from isaacgymenvs.clean_awr.awr_utils import AWRDataset, Diagnostics, ExperienceBuffer
from isaacgymenvs.clean_awr.viz.td import TemporalDistanceVisualizer
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, Every
import ml_collections
from ml_collections import config_flags
from absl import app, flags

# This wraps the package in a debugging wrapper.
import sys
import os
import pdb
import traceback
def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    # if keyboardinterrup, just quite.
    if not issubclass(etype, KeyboardInterrupt):
        print() # make a new line before launching post-mortem
        pdb.pm() # post-mortem debugger
    os._exit(0)
sys.excepthook = debughook


FLAGS = flags.FLAGS

# =======================
# Training Code.
# =======================

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

class AWRAgent():

    def __init__(self, config, algo_observer):
        self.config = config

        # Make env.
        self.vec_env = vecenv.create_vec_env('rlgpu', self.config['num_actors'])
        self.env_info = self.vec_env.get_env_info()

        # Assign some configuration things.
        self.device = config['device']
        self.observation_space = self.env_info['observation_space']
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env_info['action_space']
        assert isinstance(self.observation_space, gym.spaces.Box)
        self.batch_size = self.config['horizon_length'] * self.config['num_actors']
        self.batch_size_envs = self.batch_size
        self.num_minibatches = self.config['num_minibatches']
        self.minibatch_size = self.batch_size // self.num_minibatches
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)

        # Initialize loggers.
        self.diagnostics = Diagnostics()
        self.games_to_track = 100
        print('current training device:', self.device)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
        self.game_shaped_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)
        self.obs = None
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        self.train_dir = 'runs'
        self.experiment_dir = os.path.join(self.train_dir, self.config['run_name'])
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')
        self.save_td_dir = os.path.join('data', 'temporal_distance', self.config['run_name'])
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        self.writer = SummaryWriter(self.summaries_dir)
        self.algo_observer = algo_observer
        self.algo_observer.before_init('run', config, self.config['run_name'])
        self.algo_observer.after_init(self)

        # Initialize model.
        self.model = AWRNetwork(
            actions_num=self.actions_num,
            input_shape = self.obs_shape,
            num_seqs = self.config['num_actors'],
            units = self.config['hidden_dims'],
            fixed_sigma = True,
            normalize_value = self.config['normalize_value'],
            normalize_input = self.config['normalize_input'],
            separate=self.config['separate']
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.config['lr'], eps=1e-08, weight_decay=0)
        
        if not self.config['temporal_distance']['full_state']:
            self.td_idx = self.vec_env.env.achieved_idx
        else:
            self.td_idx = [i for i in range(self.vec_env.env.cfg['env']['numObservations']) if i not in self.vec_env.env.desired_idx]
        self.max_pred = max(self.config['relabel_every'], self.config['horizon_length'])  # relabel_every should always be bigger
        self.td_output_size = self.max_pred + 1 if not self.config['temporal_distance']['regression'] else 1
        kw = dict(
            units=self.config['hidden_dims'], 
            input_size=len(self.td_idx) + len(self.vec_env.env.achieved_idx),
            output_size=self.td_output_size
        )
        if not self.config['temporal_distance']['regression']:
            kw['classifier_selection'] = self.config['temporal_distance']['classifier_selection']
        self.temporal_distance = TemporalDistanceNetwork(**kw).to(self.device)
        self.temporal_distance_optimizer = optim.Adam(self.temporal_distance.parameters(), self.config['temporal_distance']['lr'], eps=1e-08, weight_decay=0)
        self.temporal_distance_viz = TemporalDistanceVisualizer(self)
        
        if self.config['normalize_value']:
            self.value_mean_std = self.model.value_mean_std
        self.dataset = AWRDataset(self.batch_size, self.minibatch_size, self.device)

        if self.config['relabel']:
            self.relabeled_dataset = AWRDataset(self.batch_size, self.minibatch_size, self.device)

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, metrics, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames, phase=''):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)

        self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
        self.writer.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer.add_scalar('info/e_clip', self.config['e_clip'] * lr_mul, frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)

        for k, v in metrics.items():
            if len(v) > 0:
                self.writer.add_scalar(k, torch_ext.mean_list(v).item(), frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time, phase)

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def run_model(self, obs):
        processed_obs = obs['obs']
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        return res_dict
    
    def get_values(self, obs):
        return self.run_model(obs)['values']
    
    def run_td_in_slices(self, x, y, n_slices=None, classifier_selection=None):
        # Conserves memory
        obs = torch.cat([x,y], dim=1)
        if n_slices is None: n_slices = min(16, obs.shape[0])
        obs = obs.reshape([n_slices, -1, obs.shape[1]])
        if not self.config['temporal_distance']['regression']:
            distances = [
                self.temporal_distance(obs[i], classifier_selection=classifier_selection)[1] 
                for i in range(n_slices)
            ]
        else:
            distances = [self.temporal_distance(obs[i]) for i in range(n_slices)]
        distances = torch.stack(distances, 0)
        return distances

    def cast(self, x):
        if isinstance(x, torch.Tensor): return x
        return torch.from_numpy(x).to(self.device)

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor): return obs
        return torch.FloatTensor(obs).to(self.device)

    def obs_to_tensors(self, obs):
        return {'obs' : self.cast_obs(obs)}

    def env_step(self, actions):
        if self.config['clip_actions']:
            actions = torch.clamp(actions, -1.0, 1.0)
            actions = rescale_actions(self.actions_low, self.actions_high, actions)
        # actions = actions.cpu().numpy()

        obs, rewards, terminated, truncated, infos = self.vec_env.step(actions)
        if isinstance(obs, dict): obs = obs['obs']
        dones = terminated + truncated
        return self.obs_to_tensors(obs), self.cast(rewards[:, None]).float(), self.cast(dones), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        if isinstance(obs, dict): obs = obs['obs']
        obs = self.obs_to_tensors(obs)
        return obs

    def discount_values(self, fdones, last_values, mb_fdones, mb_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.config['horizon_length'])):
            if t == self.config['horizon_length'] - 1:
                nextvalues = last_values
            else:
                nextvalues = mb_values[t+1]
            nonterminal = 1.0 - mb_fdones[t]
            nonterminal = nonterminal.unsqueeze(1)

            delta = (mb_rewards[t] + self.config['gamma'] * nextvalues * nonterminal - mb_values[t]) * nonterminal
            mb_advs[t] = lastgaelam = delta + self.config['gamma'] * self.config['tau'] * nonterminal * lastgaelam
        return mb_advs
    
    def bound_loss(self, mu):
        if self.config['bounds_loss_coef'] is not None:
            soft_bound = self.config['action_bound'] = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_shaped_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        self.algo_observer.after_clear_stats()

    def train_awr(self, original_dict):
        loss = [0, 0]
        loss_coef = [self.config.get('onpolicy_coef', 1.0), 1.0]
        rcc = self.config.get('relabeled_critic_coef', 1.0)
        rac = self.config.get('relabeled_actor_coef', 1.0)
        loss_critic_coef = [1.0, rcc]
        loss_actor_coef = [1.0, rac]
        losses_dict = {}
        diagnostics = {}
        for i, input_dict in enumerate([original_dict]):
            if not self.config['relabel'] and i == 1: continue
            identifier = '' if i == 0 else '_relabeled'

            value_preds_batch = input_dict['old_values']
            old_action_log_probs_batch = input_dict['old_logp_actions']
            advantage = input_dict['advantages']
            old_mu_batch = input_dict['mu']
            old_sigma_batch = input_dict['sigma']
            return_batch = input_dict['returns']
            batch_dict = {
                'is_train': True,
                'prev_actions': input_dict['actions'], 
                'obs' : input_dict['obs'],
            }
            lr_mul = 1.0

            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # AWR loss.
            a_loss = action_log_probs * torch.clamp((advantage / self.config['temperature']).exp(), max=20)

            # Value function loss.
            if self.config['norm_by_return']:
                norm = self.return_std[identifier]
            else:
                norm = 1
            if self.config['clip_value']:
                value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.config['e_clip'], self.config['e_clip'])
                value_losses = (values - return_batch)**2
                value_losses_clipped = (value_pred_clipped - return_batch)**2
                c_loss = torch.max(value_losses, value_losses_clipped) / norm**2
                clip_value_frac = (value_losses < value_losses_clipped).sum() / np.prod(value_losses.shape)
            else:
                c_loss = (return_batch - values)**2 / norm**2
                clip_value_frac = torch.Tensor([0])

            b_loss = self.bound_loss(mu)
            a_loss, c_loss, entropy, b_loss = a_loss.mean(), c_loss.mean(), entropy.mean(), b_loss.mean()

            loss[i] = loss_coef[i] * (a_loss * loss_actor_coef[i]
                                        + c_loss * loss_critic_coef[i] * self.config['critic_coef']
                                        - entropy * self.config['entropy_coef']
                                        + b_loss * self.config['bounds_loss_coef'])
            for param in self.model.parameters():
                param.grad = None

            with torch.no_grad():
                kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, True)

            identifier = '' if i == 0 else '_relabeled'
            losses_dict.update({f'a_loss{identifier}': a_loss, f'c_loss{identifier}': c_loss, f'entropy{identifier}': entropy})
            if self.config['bounds_loss_coef'] is not None:
                losses_dict[f'bounds_loss{identifier}'] = b_loss
            diagnostics.update({
                f'explained_variance{identifier}': torch_ext.explained_variance(value_preds_batch, return_batch, None).detach(),
                f'clipped_fraction{identifier}': torch_ext.policy_clip_fraction(action_log_probs, old_action_log_probs_batch, self.config['e_clip'], None).detach(),
                f'clipped_value_fraction{identifier}': clip_value_frac,
            })

            # print(f'values train {identifier}: {values.mean().item():.3f}')

        loss = loss[0] + loss[1]
        loss.backward()
        if self.config['truncate_grads']:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.diagnostics.mini_batch(self, diagnostics)      

        return losses_dict, kl_dist, self.config['lr'], lr_mul, mu.detach(), sigma.detach()

    # Normalize advantages, returns, etc.
    def prepare_dataset(self, batch_dict, dataset, update_mov_avg=True, fixed_advantage_normalizer=None, identifier=''):
        returns = batch_dict['returns']
        values = batch_dict['values']
        advantages = returns - values
        self.diagnostics.add(f'diagnostics/return_mean{identifier}', returns.mean())
        self.diagnostics.add(f'diagnostics/value_mean{identifier}', values.mean())

        if self.config['normalize_value']:
            if update_mov_avg:
                self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            if update_mov_avg:
                self.value_mean_std.eval()
        self.return_std[identifier] = returns.std()

        advantages = torch.sum(advantages, axis=1)
        if self.config['normalize_advantage']:
            # if update_mov_avg:
            if identifier == '':
                self.advantage_mean_std = dict(mean=advantages.mean(), std=advantages.std())
            self.diagnostics.add(f'diagnostics/advantage_mean{identifier}', advantages.mean())
            self.diagnostics.add(f'diagnostics/advantage_std{identifier}', advantages.std())

            if fixed_advantage_normalizer is not None:
                advantages = (advantages - fixed_advantage_normalizer[0]) / (fixed_advantage_normalizer[1] + 1e-8)
            else:
                advantages = (advantages - self.advantage_mean_std['mean']) / (self.advantage_mean_std['std'] + 1e-8)
        
            if self.config['normalize_value']:
                self.diagnostics.add(f'diagnostics/rms_value_mean{identifier}', self.value_mean_std.running_mean)
                self.diagnostics.add(f'diagnostics/rms_value_std{identifier}', math.sqrt(self.value_mean_std.running_var))

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = batch_dict['neglogpacs']
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = batch_dict['actions']
        dataset_dict['obs'] = batch_dict['obses']
        dataset_dict['dones'] = batch_dict['dones']
        dataset_dict['mu'] = batch_dict['mus']
        dataset_dict['sigma'] = batch_dict['sigmas']
        dataset.update_values_dict(dataset_dict)

    # Evaluate the policy.
    def test(self, render):
        self.set_eval()
        self.vec_env.env.test = True
        if render:
            self.vec_env.env.override_render = True
        self.env_reset()
        cut = self.vec_env.env.max_pix

        update_list = self.update_list
        obs = self.obs
        dones = self.dones
        for n in range(self.vec_env.env.max_episode_length - 1):
            res_dict = self.run_model(obs)
            self.test_buffer.update_data('obses', n, obs['obs'][:cut])
            self.test_buffer.update_data('dones', n, dones[:cut])
            for k in update_list:
                self.test_buffer.update_data(k, n, res_dict[k][:cut]) 
            obs, rewards, dones, infos = self.env_step(res_dict['actions'])
            # Save images
            all_done_indices = dones.nonzero(as_tuple=False)[::1]
            self.algo_observer.process_infos(infos, all_done_indices)

        self.vec_env.env.test = False
        if render:
            self.vec_env.env.override_render = False
        self.env_reset()

    def train(self):
        batch_size = self.config['num_actors']

        # Buffers to hold rollouts in.
        self.experience_buffer = ExperienceBuffer(self.observation_space, self.action_space, self.config['horizon_length'], self.config['num_actors'], self.device)
        self.test_buffer = ExperienceBuffer(self.observation_space, self.action_space, self.vec_env.env.max_episode_length, self.vec_env.env.max_pix, self.device)

        # Stats
        val_shape = (self.config['horizon_length'], batch_size, 1)
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        self.vec_env.env.start_epoch_num = self.epoch_num
        test_check = Every(self.config['test_every_episodes'] * self.vec_env.env.max_episode_length)
        test_render_check = Every(math.ceil(self.vec_env.env.render_every_episodes / self.config['test_every_episodes']))
        test_counter = 0
        self.start_frame = self.frame

        while True:
            self.epoch_num += 1
            epoch_num = self.epoch_num

            # ===============================
            # Collect data by doing a rollout in the environment. Store in self.experience_buffer.
            # ===============================
            self.vec_env.set_train_info(self.frame, self)
            self.set_eval()
            play_time_start = time.time()
            success_buffer = None
            with torch.no_grad():
                update_list = self.update_list
                step_time = 0.0

                for n in range(self.config['horizon_length']):
                    res_dict = self.run_model(self.obs)
                    self.experience_buffer.update_data('obses', n, self.obs['obs'])
                    self.experience_buffer.update_data('dones', n, self.dones)

                    for k in update_list:
                        self.experience_buffer.update_data(k, n, res_dict[k]) 

                    step_time_start = time.time()
                    self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
                    step_time_end = time.time()

                    step_time += (step_time_end - step_time_start)
                    shaped_rewards = rewards
                    if self.config['value_bootstrap'] and 'time_outs' in infos:
                        shaped_rewards += self.config['gamma'] * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

                    self.experience_buffer.update_data('rewards', n, shaped_rewards)

                    self.current_rewards += rewards
                    self.current_shaped_rewards += shaped_rewards
                    self.current_lengths += 1
                    all_done_indices = self.dones.nonzero(as_tuple=False)
                    env_done_indices = all_done_indices[::1]
            
                    self.game_rewards.update(self.current_rewards[env_done_indices])
                    self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
                    self.game_lengths.update(self.current_lengths[env_done_indices])
                    self.algo_observer.process_infos(infos, env_done_indices)

                    not_dones = 1.0 - self.dones.float()

                    self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
                    self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
                    self.current_lengths = self.current_lengths * not_dones
                
                    if n == self.config['horizon_length'] - 1:
                        success_buffer = infos['episodic'][self.vec_env.env.success_key]

                last_values = self.get_values(self.obs)

                fdones = self.dones.float()
                mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
                mb_values = self.experience_buffer.tensor_dict['values']
                mb_rewards = self.experience_buffer.tensor_dict['rewards']

            self.curr_frames = self.batch_size
            play_time_end = time.time()
            update_time_start = time.time()
            fixed_advantage_normalizer = None
            metrics = defaultdict(list)
            td_hist_buffer = {}
            
            temporal_distance_dataset = TemporalDistanceDataset(self.experience_buffer, self.config, self.vec_env.env)

            if self.config['temporal_distance']['save_data']:
                if self.epoch_num == 1: print('Saving TD data to:', self.save_td_dir)
                td_save_path = Path(self.save_td_dir) / f'{self.epoch_num}.npy'
                td_save_path.parent.mkdir(parents=True, exist_ok=True)
                td_save_dict = {k: v.detach().cpu().numpy() for k, v in temporal_distance_dataset.pairs.items()}
                np.save(td_save_path, td_save_dict)
                
            temporal_distance_dataset.pairs['goal'] = temporal_distance_dataset.pairs['goal'][:, self.td_idx]

            # ===============================
            # Train distance function
            # ===============================

            for mini_ep in range(0, self.config['temporal_distance']['mini_epochs']):
                for i in range(len(self.dataset)):
                    losses = self.temporal_distance.loss(temporal_distance_dataset[i])
                    loss_key = 'ce' if not self.config['temporal_distance']['regression'] else 'mse'
                    losses[loss_key].backward()
                    self.temporal_distance_optimizer.step()
                    self.temporal_distance_optimizer.zero_grad()
                    if mini_ep == 0:
                        for k in [loss_key, 'accuracy']:
                            metrics[f'temporal_distance/val_{k}'].append(losses[k])
                    if mini_ep == self.config['temporal_distance']['mini_epochs'] - 1:
                        for k in [loss_key, 'accuracy']:
                            metrics[f'temporal_distance/{k}'].append(losses[k])
                        if i == 0:  # selected arbitrarily
                            td_hist_buffer = {'pred': losses['pred'], 'target': temporal_distance_dataset[i]['distance']}
                # print('epoch', self.epoch_num, {k: round(v.item(), 4) for k, v in losses.items()})
            
            if self.epoch_num % self.config['temporal_distance']['plot_every'] == 0:
                self.temporal_distance_viz.viz_success(success_buffer)
                self.temporal_distance_viz.hist(
                    'temporal_distance/state_desired_scatter',
                    td_hist_buffer['pred'].flatten(), 
                    td_hist_buffer['target'].flatten())

            if self.config['relabel']:
                # Relabel rewards with temporal distance
                with torch.no_grad():
                    obs = self.experience_buffer.tensor_dict['obses']
                    distances = self.run_td_in_slices(obs[:, :, self.td_idx].flatten(0,1),
                                                      obs[:, :, self.vec_env.env.desired_idx].flatten(0,1))
                    distances = distances.reshape(self.config['horizon_length'], self.config['num_actors'], -1)
                    self.temporal_distance_viz.hist('temporal_distance/state_desired_hist', distances.flatten())
                        
                    td_rewards = - distances / self.config['relabel_every']
                    mb_rewards = td_rewards
            
            mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
            mb_returns = mb_advs + mb_values
            batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
            batch_dict['returns'] = swap_and_flatten01(mb_returns)
            batch_dict['step_time'] = step_time

            self.return_std = dict()
            self.prepare_dataset(batch_dict, self.dataset, fixed_advantage_normalizer=fixed_advantage_normalizer, identifier='')

            kl_dataset = self.dataset
            self.algo_observer.after_steps()
            ep_kls = []

            # ===============================
            # Train policy and value function.
            # ===============================

            for mini_ep in range(0, self.config['mini_epochs']):
                ep_kls = []

                # self.dataset.shuffle()
                # if self.relabel:
                #     self.relabeled_dataset.shuffle()
                for i in range(len(self.dataset)):
                    losses, kl, last_lr, lr_mul, cmu, csigma = self.train_awr(self.dataset[i])
                    for k, v in losses.items(): 
                        metrics[f'losses/{k}'].append(v)
                    ep_kls.append(kl)
                    kl_dataset.update_mu_sigma(cmu, csigma)

                av_kls = torch_ext.mean_list(ep_kls)
                metrics['info/kl'].append(av_kls)
                self.diagnostics.mini_epoch(self, mini_ep)
                if self.config['normalize_input']:
                    self.model.running_mean_std.eval()

            update_time_end = time.time()
            play_time = play_time_end - play_time_start
            update_time = update_time_end - update_time_start
            sum_time = update_time_end - play_time_start

            # ===============================
            # Log Stats.
            # ===============================
            total_time += sum_time
            frame = self.frame
            sigma = self.dataset.values_dict['sigma'][self.dataset.last_range[0]:self.dataset.last_range[1]].mean()
            self.writer.add_scalar('info/sigma', sigma, frame)
            self.diagnostics.epoch(self, current_epoch = epoch_num)
            scaled_time = sum_time
            scaled_play_time = play_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                            metrics, last_lr, lr_mul, frame,
                            scaled_time, scaled_play_time, curr_frames)
            # If an episode has ended, log episode-specific stats.
            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()
                self.mean_rewards = mean_rewards[0]

                step_time = max(step_time, 1e-9)
                fps_total = self.frame / total_time
                max_epochs = self.config['max_epochs']
                ep_str = f'/{max_epochs:.0f}' if max_epochs != -1 else ''
                print(f'rewards: {mean_rewards[0]:.3f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}{ep_str} frames:', self.frame)

                for i in range(1):
                    rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                    self.writer.add_scalar(rewards_name.format(i), mean_rewards[i], frame)
                    self.writer.add_scalar('shaped_' + rewards_name.format(i), mean_shaped_rewards[i], frame)

                self.writer.add_scalar('episode_lengths', mean_lengths, frame)
            update_time = 0
            
            # ===============================
            # Evaluate policy in test mode.
            # ===============================
            # iteration = (self.frame - self.start_frame) / self.config['num_actors']
            # if test_check.check(iteration):
            #     print("Testing...")
            #     test_counter += 1
            #     self.test(render=test_render_check.check(test_counter))
            #     self.algo_observer.after_print_stats(frame, epoch_num, total_time, '_test')
            #     print("Done Testing.")


# =======================
# Run Script.
# =======================
def main(_):
    FLAGS.agent['num_actors'] = FLAGS.agent['num_envs']
    FLAGS.env['env']['numEnvs'] = FLAGS.agent['num_envs']

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            -1, # seed 
            FLAGS.env['name'], 
            FLAGS.agent['num_envs'], 
            FLAGS.agent['device'],
            FLAGS.agent['device'],
            FLAGS.agent['graphics_device_id'],
            True, # headless
            False, # multi_gpu
            False, # capture_video
            True, # force_render
            FLAGS.env,
            **kwargs,
        )
        return envs
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })
    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    time_str = datetime.now().strftime("%y%m%d-%H%M%S-%f")
    if FLAGS.slurm_job_id != -1:
        run_name = f"{FLAGS.slurm_job_id}_{FLAGS.agent['experiment']}"
    else:
        run_name = f"{time_str}_{FLAGS.agent['experiment']}"
    FLAGS.agent['run_name'] = run_name

    env = dict(FLAGS.env)
    env['env'] = dict(env['env'])
    if 'sim' in env:
        env['sim'] = dict(env['sim'])
    wandb.init(
        project='taskmaster',
        entity=FLAGS.agent['wandb_entity'],
        group='Default',
        sync_tensorboard=True,
        id=run_name,
        name=run_name,
        config={'agent': dict(FLAGS.agent), 'env': env, 'experiment': FLAGS.agent['experiment']},
        # settings=wandb.Settings(start_method='fork'),
    )

    agent = AWRAgent(FLAGS.agent, RLGPUAlgoObserver(run_name))
    agent.train()

if __name__ == '__main__':
    flags.DEFINE_integer('slurm_job_id', -1, '')
    config_flags.DEFINE_config_file('agent', 'clean_awr/agent_config/ig_push.py', 'Agent configuration.', lock_config=False)
    config_flags.DEFINE_config_file('env', 'clean_awr/env_config/ig_push.py', 'Env configuration.', lock_config=False)

    app.run(main)