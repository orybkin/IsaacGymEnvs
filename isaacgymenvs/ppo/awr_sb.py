import copy
import os
from collections import defaultdict

from rl_games.algos_torch import torch_ext
from isaacgymenvs.ppo import a2c_common
from utils.rlgames_utils import Every
from isaacgymenvs.ppo.a2c_common import A2CBase, rescale_actions, swap_and_flatten01, print_statistics
from isaacgymenvs.ppo.awr_sb_base import AWRsbBase
from isaacgymenvs.ppo import torch_ext
from isaacgymenvs.ppo import datasets
import numpy as np
import time
import math

import torch 
from torch import nn
import torch.distributed as dist
from torch import optim


import warnings
import copy
import sys
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import time
from collections import defaultdict

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, BaseBuffer
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from isaacgymenvs.awr.awr_original import RAWR
from stable_baselines3.common.noise import NormalActionNoise

    

class AWRsb(AWRsbBase):
    def __init__(self, base_name, params):
        AWRsbBase.__init__(self, base_name, params)

        policy_config = dict()
        policy_config['n_envs'] = self.config['horizon_length']
        policy_config['relabel_ratio'] = 0.5 if self.config['relabel'] else 0.0
        policy_config['learning_rate'] = self.config['learning_rate']
        policy_config['ent_coef'] = self.config['entropy_coef']
        policy_config['temperature'] = self.config['awr_temperature']
        policy_config['max_grad_norm'] = self.config['grad_norm']
        policy_config['gamma'] = self.config['tau']
        policy_config['gamma'] = self.config['gamma']
        policy_config['batch_size'] = self.config['horizon_length'] * self.config['num_actors'] // self.config['num_minibatches']
        policy_config['n_epochs'] = self.config['mini_epochs']
        policy_config['n_envs'] = self.config['num_actors']
        policy_config['n_steps'] = self.config['horizon_length']
        policy_config['policy_kwargs'] = dict(log_std_init=0.0, net_arch=[256, 256, 256])

        self.sb_agent = RAWR("MlpPolicy", self.vec_env, verbose=1, run_name=self.experiment_name, tensorboard_log=f"logs/{self.experiment_name}", **policy_config)


    # def train(self):
    #     self.init_tensors()
    #     self.last_mean_rewards = -100500
    #     start_time = time.time()
    #     total_time = 0
    #     rep_count = 0
    #     self.obs = self.env_reset()
    #     self.curr_frames = self.batch_size_envs
    #     self.vec_env.env.start_epoch_num = self.epoch_num
    #     test_check = Every(self.test_every_episodes * self.vec_env.env.max_episode_length)
    #     test_render_check = Every(math.ceil(self.vec_env.env.render_every_episodes / self.test_every_episodes))
    #     test_counter = 0
    #     self.start_frame = self.frame

    #     if self.multi_gpu:
    #         print("====================broadcasting parameters")
    #         model_params = [self.model.state_dict()]
    #         dist.broadcast_object_list(model_params, 0)
    #         self.model.load_state_dict(model_params[0])

    #     while True:
    #         epoch_num = self.update_epoch()
    #         step_time, play_time, update_time, sum_time, metrics, last_lr, lr_mul = self.train_epoch()
    #         total_time += sum_time
    #         frame = self.frame // self.num_agents
    #         if self.global_rank == 0:
    #             sigma = self.dataset.values_dict['sigma'][self.dataset.last_range[0]:self.dataset.last_range[1]].mean()
    #             self.writer.add_scalar('info/sigma', sigma, frame)

    #         # cleaning memory to optimize space
    #         self.dataset.update_values_dict(None)
    #         if self.relabel:
    #             self.relabeled_dataset.update_values_dict(None)
    #         should_exit = False

    #         if self.global_rank == 0:
    #             self.diagnostics.epoch(self, current_epoch = epoch_num)
    #             # do we need scaled_time?
    #             scaled_time = self.num_agents * sum_time
    #             scaled_play_time = self.num_agents * play_time
    #             curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
    #             self.frame += curr_frames

    #             self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
    #                             metrics, last_lr, lr_mul, frame,
    #                             scaled_time, scaled_play_time, curr_frames)

    #             if self.has_soft_aug:
    #                 self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)


    #             if self.game_rewards.current_size > 0:
    #                 mean_rewards = self.game_rewards.get_mean()
    #                 mean_shaped_rewards = self.game_shaped_rewards.get_mean()
    #                 mean_lengths = self.game_lengths.get_mean()
    #                 self.mean_rewards = mean_rewards[0]

    #                 print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
    #                                 epoch_num, self.max_epochs, frame, self.max_frames, mean_rewards[0])

    #                 for i in range(self.value_size):
    #                     rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
    #                     self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
    #                     self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
    #                     self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
    #                     self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
    #                     self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
    #                     self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

    #                 self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
    #                 self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
    #                 self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

    #                 checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

    #                 if self.save_freq > 0:
    #                     if epoch_num % self.save_freq == 0:
    #                         self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

    #                 if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
    #                     print('saving next best rewards: ', mean_rewards)
    #                     self.last_mean_rewards = mean_rewards[0]
    #                     self.save(os.path.join(self.nn_dir, self.config['name']))

    #                     if 'score_to_win' in self.config:
    #                         if self.last_mean_rewards > self.config['score_to_win']:
    #                             print('Maximum reward achieved. Network won!')
    #                             self.save(os.path.join(self.nn_dir, checkpoint_name))
    #                             should_exit = True

    #             if epoch_num >= self.max_epochs and self.max_epochs != -1:
    #                 if self.game_rewards.current_size == 0:
    #                     print('WARNING: Max epochs reached before any env terminated at least once')
    #                     mean_rewards = -np.inf

    #                 self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
    #                     + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
    #                 print('MAX EPOCHS NUM!')
    #                 should_exit = True

    #             if self.frame >= self.max_frames and self.max_frames != -1:
    #                 if self.game_rewards.current_size == 0:
    #                     print('WARNING: Max frames reached before any env terminated at least once')
    #                     mean_rewards = -np.inf

    #                 self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
    #                     + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
    #                 print('MAX FRAMES NUM!')
    #                 should_exit = True

    #             update_time = 0

    #         if self.multi_gpu:
    #             should_exit_t = torch.tensor(should_exit, device=self.device).float()
    #             dist.broadcast(should_exit_t, 0)
    #             should_exit = should_exit_t.float().item()
    #         if should_exit:
    #             return self.last_mean_rewards, epoch_num

    #         if should_exit:
    #             return self.last_mean_rewards, epoch_num
            
    #         # Test
    #         iteration = (self.frame - self.start_frame) / self.num_actors
    #         if test_check.check(iteration):
    #             print("Testing...")
    #             test_counter += 1
    #             self.test(render=test_render_check.check(test_counter))
    #             self.algo_observer.after_print_stats(frame, epoch_num, total_time, '_test')
    #             print("Done Testing.")

    def train(self):
        import pdb; pdb.set_trace()
        log_interval = 1
        tb_log_name = "AWR"
        total_timesteps = np.inf
        iteration = 0
        total_timesteps, callback = self._setup_learn(total_timesteps, callback, tb_log_name)
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # ======================
            # Collect rollouts
            # ======================
            time_start = time.time()
            self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            time_collected = time.time()
            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            print(self.tensorboard_log)
            if log_interval is not None and iteration % log_interval == 0:
                self.logger.record("time/iterations", iteration)
                self._dump_logs()

            # ======================
            # Create relabelled buffer.
            # ======================
            relabeled_buffer = copy.deepcopy(self.rollout_buffer)

            # Set goal to be final achieved goal of the trajectory.
            obs = self.rollout_buffer.observations
            idx = self.rollout_buffer.episode_starts.astype(int)
            init_idx = idx.argmax(0)
            idx = idx.cumsum(0) * 50 - 1 + init_idx[None, :] # TODO  this is hardcoded for now
            idx = np.minimum(idx, self.n_steps - 1)
            if isinstance(obs, dict):
                last_obs = self._last_obs.copy()
                last_obs['desired_goal'] = relabeled_buffer.observations['achieved_goal'][-1]
                relabeled_buffer.observations['desired_goal'] = np.take_along_axis(obs['achieved_goal'], idx[:, :, None], 0)
                goal = relabeled_buffer.observations['desired_goal']
                pos = relabeled_buffer.observations['achieved_goal']
                goal = np.concatenate([goal[1:], last_obs['desired_goal'][None]], 0)
                pos = np.concatenate([pos[1:], last_obs['achieved_goal'][None]], 0)
                relabeled_buffer.rewards = self.env.env_method("compute_reward", pos, goal, None, indices=[0])[0]
            else:
                raise NotImplementedError
        
            # Compute value/advantages for relabeled buffer.
            with th.no_grad():
                obs_tensor = obs_as_tensor(relabeled_buffer.observations, self.device)
                act_tensor = obs_as_tensor(relabeled_buffer.actions, self.device)
                values = self.policy.predict_values(flatten(obs_tensor))
                relabeled_buffer.values = values.reshape(list(act_tensor.shape[:2])).cpu().numpy()

                # Add value to last reward IF flag is done, but not terminal. TODO check this.
                done = np.concatenate([relabeled_buffer.episode_starts[1:], self._last_episode_starts[None]], 0)
                relabeled_buffer.rewards = relabeled_buffer.rewards + self.gamma * relabeled_buffer.values * done

                # Compute advantages.
                relabeled_buffer.returns = np.zeros_like(relabeled_buffer.returns)
                relabeled_buffer.advantages = np.zeros_like(relabeled_buffer.advantages)
                obs_tensor = obs_as_tensor(last_obs, self.device)
                values = self.policy.predict_values(obs_tensor)  
                relabeled_buffer.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)

            # ======================
            # Update policy.
            # ======================            
            buffer = MixedBuffer(relabeled_buffer, self.rollout_buffer, self.relabel_ratio)

            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)

            entropy_losses = []
            pg_losses, value_losses = [], []
            metrics = defaultdict(list)

            if self.normalize_advantage and len(buffer.advantages) > 1:
                buffer_adv_mean = buffer.advantages.mean()
                buffer_adv_std = buffer.advantages.std()
            bs = self.batch_size

            # train for n_epochs epochs
            for epoch in range(self.n_epochs):
                # Do a complete pass on the rollout buffer
                for batch in buffer.get(self.batch_size):
                    actions = batch.actions

                    policy_distribution = self.policy.get_distribution(batch.observations)
                    log_prob = policy_distribution.log_prob(actions)
                    entropy = policy_distribution.entropy()

                    advantages = batch.advantages
                    metrics['advantages_mean'].append(advantages[:bs].mean().item())
                    metrics['advantages_std'].append(advantages[:bs].std().item())
                    metrics['advantages_max'].append(advantages[:bs].max().item())
                    metrics['advantages_min'].append(advantages[:bs].min().item())
                    if self.relabel_ratio == 0.5:
                        metrics['advantages_mean_relabeled'].append(advantages[-bs:].mean().item())
                        metrics['advantages_std_relabeled'].append(advantages[-bs:].std().item())
                        metrics['advantages_max_relabeled'].append(advantages[-bs:].max().item())
                        metrics['advantages_min_relabeled'].append(advantages[-bs:].min().item())

                    # Normalize advantage
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - buffer_adv_mean) / (buffer_adv_std + 1e-8)

                    ## AWR
                    weighting = th.clamp((advantages / self.temperature).exp(), max=20)
                    policy_loss = -(log_prob * weighting)

                    # Entropy loss favor exploration
                    entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(log_prob)
                    entropy_losses.append(entropy_loss.item())

                    ## Value
                    values = self.policy.predict_values(batch.observations).flatten()
                    value_loss = ((batch.returns - values) ** 2)

                    loss = policy_loss.mean() + self.ent_coef * entropy_loss + self.vf_coef * value_loss.mean()

                    # Logging
                    metrics['advantages_mean_normalized'].append(advantages[:bs].mean().item())
                    metrics['value_pred'].append(values[:bs].mean().item())
                    metrics['return'].append(batch.returns[:bs].mean().item())
                    metrics['log_prob'].append(log_prob[:bs].mean().item())
                    metrics['weighting'].append(weighting[:bs].mean().item())
                    metrics['policy_gradient_loss'].append(policy_loss[:bs].mean().item())
                    metrics['value_loss'].append(value_loss[:bs].mean().item())
                    if self.relabel_ratio == 0.5:
                        metrics['advantages_mean_normalized_relabeled'].append(advantages[-bs:].mean().item())
                        metrics['value_pred_relabeled'].append(values[-bs:].mean().item())
                        metrics['return_relabeled'].append(batch.returns[-bs:].mean().item())
                        metrics['policy_gradient_loss_relabeled'].append(policy_loss[-bs:].mean().item())
                        metrics['value_loss_relabeled'].append(value_loss[-bs:].mean().item())

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    grad = [th.linalg.vector_norm(p.grad.detach()) for p in self.policy.parameters() if p.grad is not None]
                    grad = th.linalg.vector_norm(th.stack(grad))
                    metrics['grad_norm'].append(grad.item())
                    if self.max_grad_norm:
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                self._n_updates += 1

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            # Logs
            for metric in metrics: 
                self.logger.record(f"train/{metric}", np.mean(metrics[metric]))
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train/n_updates", self._n_updates)
            time_trained = time.time()
            self.diagnostics['time_collect'].append(time_collected - time_start)
            self.diagnostics['time_train'].append(time_trained - time_collected)

