
import copy
import os
from collections import defaultdict

from rl_games.algos_torch import torch_ext
from isaacgymenvs.ppo import a2c_common
from utils.rlgames_utils import Every
from isaacgymenvs.ppo.a2c_common import A2CBase, rescale_actions, swap_and_flatten01, print_statistics
from isaacgymenvs.ppo import torch_ext
from isaacgymenvs.ppo import datasets
import numpy as np
import time
import math

import torch 
from torch import nn
import torch.distributed as dist
from torch import optim



class AWRsbBase(a2c_common.A2CBase):

    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)

        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)

        self.clip_actions = self.config.get('clip_actions', True)

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)

        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.relabel:
            self.relabeled_dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def relabel_batch(self, buffer):
        env = self.vec_env.env
        relabeled_buffer = copy.deepcopy(buffer)

        # Relabel states
        obs = relabeled_buffer.tensor_dict['obses']
        # compute episode idx
        idx = self.get_relabel_idx(env, relabeled_buffer.tensor_dict['dones'])
        desired = torch.gather(obs[:, :, env.achieved_idx], 0, idx)
        relabeled_buffer.tensor_dict['obses'][:, :, env.desired_idx] = desired
        achieved = obs[..., env.achieved_idx]

        # Rewards should be shifted by one
        last_obs = dict(obs=self.obs['obs'].clone())
        last_obs['obs'][:, env.desired_idx] = relabeled_buffer.tensor_dict['obses'][-1, :, env.desired_idx]
        desired = torch.cat([desired[1:], last_obs['obs'][None, :, env.desired_idx]], 0)
        achieved = torch.cat([achieved[1:], last_obs['obs'][None, :, env.achieved_idx]], 0)

        res_dict = self.run_model_in_slices(obs, relabeled_buffer.tensor_dict['actions'], relabeled_buffer.tensor_dict.keys())
        relabeled_buffer.tensor_dict.update(res_dict)
        rewards = env.compute_franka_reward({'goal_pos': desired, env.target_name: achieved})[:, :, None]

        # TODO there is something funny about this - why the multiply by gamma?
        if self.value_bootstrap:
            rewards += self.gamma * relabeled_buffer.tensor_dict['values'] * relabeled_buffer.tensor_dict['dones'].unsqueeze(2).float()
        relabeled_buffer.tensor_dict['rewards'] = rewards

        # Compute returns
        last_values = self.get_values(last_obs)
        fdones = self.dones.float()
        mb_fdones = relabeled_buffer.tensor_dict['dones'].float()
        mb_values = relabeled_buffer.tensor_dict['values']
        mb_rewards = relabeled_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        relabeled_batch = relabeled_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        relabeled_batch['returns'] = swap_and_flatten01(mb_returns)
        relabeled_batch['played_frames'] = self.batch_size

        return relabeled_buffer, relabeled_batch

    def run_model_in_slices(self, obs, actions, out_list, n_slices=None):
        # Conserves memory
        if n_slices is None: n_slices = min(16, obs.shape[0])

        obs = obs.reshape([n_slices, -1] + list(obs.shape[1:]))
        res_dicts = [self.run_model(dict(obs=obs[i].flatten(0, 1))) for i in range(n_slices)]
        res_dict = {}
        for k in res_dicts[0]:
            if k in out_list:
                res_dict[k] = torch.stack([d[k] for d in res_dicts], 0).reshape(self.horizon_length, self.num_actors, -1)
        res_dict.pop('actions')
        res_dict['neglogpacs'] = self.model.neglogp(actions, res_dict['mus'], res_dict['sigmas'], torch.log(res_dict['sigmas']))
                                                     
        return res_dict

    def get_relabel_idx(self, env, dones):
        idx = dones
        present, first_idx = idx.max(0)
        first_idx[present == 0] = self.horizon_length - 1
        idx = idx.flip(0).cumsum(0).flip(0)
        idx = idx[[0]] - idx
        # Compute last frame idx
        ep_len = env.max_episode_length
        idx = idx * ep_len + first_idx[None, :]
        idx = torch.minimum(idx, (dones.shape[0] - 1) * torch.ones([1], dtype=torch.int32, device=idx.device))[:, :, None]
        idx = idx.repeat(1, 1, len(env.achieved_idx))
        return idx

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        if self.relabel:
            self.set_eval()
            relabeled_buffer, relabeled_batch = self.relabel_batch(self.experience_buffer)
        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict, self.dataset)

        kl_dataset = self.dataset
        if self.relabel:
            self.set_eval()
            self.prepare_dataset(relabeled_batch, self.relabeled_dataset, update_mov_avg=self.joint_value_norm, identifier='_relabeled')
            kl_dataset = self.relabeled_dataset
            self.set_train()
        relabeled_minibatch = None

        self.algo_observer.after_steps()
        metrics = defaultdict(list)
        ep_kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []

            for i in range(len(self.dataset)):
                relabeled_minibatch = self.relabeled_dataset[i] if self.relabel else None
                losses, kl, last_lr, lr_mul, cmu, csigma = self.train_actor_critic(self.dataset[i], relabeled_minibatch)
                for k, v in losses.items(): metrics[f'losses/{k}'].append(v)
                ep_kls.append(kl)

                kl_dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            metrics['info/kl'].append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, metrics, last_lr, lr_mul

    def prepare_dataset(self, batch_dict, dataset, update_mov_avg=True, identifier=''):
        returns = batch_dict['returns']
        values = batch_dict['values']
        rnn_masks = batch_dict.get('rnn_masks', None)
        advantages = returns - values

        self.diagnostics.add(f'diagnostics/return_mean{identifier}', returns.mean())
        self.diagnostics.add(f'diagnostics/value_mean{identifier}', values.mean())

        if self.normalize_value:
            if update_mov_avg:
                self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            if update_mov_avg:
                self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                    self.diagnostics.add(f'diagnostics/rms_advantage_mean{identifier}', self.advantage_mean_std.moving_mean)
                    self.diagnostics.add(f'diagnostics/rms_advantage_var{identifier}', self.advantage_mean_std.moving_var)
                else:
                    # if update_mov_avg:
                    if identifier == '':
                        self.advantage_mean_std = dict(mean=advantages.mean(), std=advantages.std())
                    self.diagnostics.add(f'diagnostics/advantage_mean{identifier}', advantages.mean())
                    self.diagnostics.add(f'diagnostics/advantage_std{identifier}', advantages.std())
                    advantages = (advantages - self.advantage_mean_std['mean']) / (self.advantage_mean_std['std'] + 1e-8)
            
                if self.normalize_value:
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
        dataset_dict['rnn_states'] = batch_dict.get('rnn_states', None)
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = batch_dict['mus']
        dataset_dict['sigma'] = batch_dict['sigmas']

        dataset.update_values_dict(dataset_dict)

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
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(obs, masks)
            else:
                res_dict = self.run_model(obs)
            
            self.test_buffer.update_data('obses', n, obs['obs'][:cut])
            self.test_buffer.update_data('dones', n, dones[:cut])

            for k in update_list:
                self.test_buffer.update_data(k, n, res_dict[k][:cut]) 

            obs, rewards, dones, infos = self.env_step(res_dict['actions'])

            # Save images
            all_done_indices = dones.nonzero(as_tuple=False)[::self.num_agents]
            self.algo_observer.process_infos(infos, all_done_indices)

        self.vec_env.env.test = False
        if render:
            self.vec_env.env.override_render = False
        self.env_reset()

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        self.vec_env.env.start_epoch_num = self.epoch_num
        test_check = Every(self.test_every_episodes * self.vec_env.env.max_episode_length)
        test_render_check = Every(math.ceil(self.vec_env.env.render_every_episodes / self.test_every_episodes))
        test_counter = 0
        self.start_frame = self.frame

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, metrics, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents
            if self.global_rank == 0:
                sigma = self.dataset.values_dict['sigma'][self.dataset.last_range[0]:self.dataset.last_range[1]].mean()
                self.writer.add_scalar('info/sigma', sigma, frame)

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            if self.relabel:
                self.relabeled_dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                metrics, last_lr, lr_mul, frame,
                                scaled_time, scaled_play_time, curr_frames)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)


                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                    epoch_num, self.max_epochs, frame, self.max_frames, mean_rewards[0])

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num
            
            # Test
            iteration = (self.frame - self.start_frame) / self.num_actors
            if test_check.check(iteration):
                print("Testing...")
                test_counter += 1
                self.test(render=test_render_check.check(test_counter))
                self.algo_observer.after_print_stats(frame, epoch_num, total_time, '_test')
                print("Done Testing.")

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn, self.device)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def train_actor_critic(self, input_dict, relabeled_dict=None):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss, clip_value_frac = a2c_common.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        self.truncate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
        
        self.diagnostics.mini_batch(self, 
        {
            'explained_variance': torch_ext.explained_variance(value_preds_batch, return_batch, rnn_masks).detach(),
            'clipped_fraction': torch_ext.policy_clip_fraction(action_log_probs, old_action_log_probs_batch, self.e_clip, rnn_masks).detach(),
            'clipped_value_fraction': clip_value_frac.detach(),
        })  

        losses_dict = {'a_loss': a_loss, 'c_loss': c_loss, 'entropy': entropy}
        if self.bounds_loss_coef is not None:
            losses_dict['bounds_loss'] = b_loss
        return losses_dict, kl_dist, self.last_lr, lr_mul, mu.detach(), sigma.detach()

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss


