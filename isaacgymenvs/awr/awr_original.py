import warnings
import copy
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

def flatten(x):
    if isinstance(x, dict):
        return {k: v.flatten(0, 1) for k, v in x.items()}
    else:
        return x.flatten(0, 1)

class MixedBuffer(BaseBuffer):
    def __init__(self, buffer1, buffer2, ratio):
        self.buffer1 = buffer1
        self.buffer2 = buffer2
        self.ratio = ratio
        self.buffer_size = buffer1.buffer_size = buffer2.buffer_size
        self.n_envs = buffer1.n_envs

    @property
    def advantages(self):
        return np.concatenate([self.buffer1.advantages, self.buffer2.advantages], 0)

    def get(self, batch_size: Optional[int] = None):
        def cat(x, y):
            if isinstance(x, dict):
                return {k: th.cat([x[k], y[k]], 0) for k in x.keys()}
            else:
                return th.cat([x, y], 0)
            
        start_idx = 0
        if self.ratio == 0.5:
            # If we can, multiply the batch size by two so we go over all existing data
            batch1_gen = self.buffer1.get(int(batch_size * self.ratio * 2))
            batch2_gen = self.buffer2.get(int(batch_size * (1 - self.ratio) * 2))
        else:
            # TODO: implement a nice way to use all existing data in this case
            batch1_gen = self.buffer1.get(int(batch_size * self.ratio))
            batch2_gen = self.buffer2.get(int(batch_size * (1 - self.ratio)))
        shuffle = th.randperm(batch_size)
        while start_idx < self.buffer_size * self.n_envs:
            batch1 = next(batch1_gen)
            batch2 = next(batch2_gen)
            # merged_batch = type(batch1)(*[th.cat([batch1[i], batch2[i]], 0)[shuffle] for i in range(len(batch1))])
            merged_batch = type(batch1)(*[cat(batch1[i], batch2[i]) for i in range(len(batch1))])
            yield merged_batch
            start_idx += batch_size

    def _get_samples(self, *args):
        raise NotImplementedError
    
class RAWR(BaseAlgorithm):

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        temperature: float = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        relabel_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = float(ent_coef)
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.relabel_ratio = relabel_ratio
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.temperature = temperature

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.lr_schedule = lambda *args: self.learning_rate
        self.set_random_seed(self.seed)

        if isinstance(self.observation_space, spaces.Dict):
            rollout_buffer_class = DictRolloutBuffer
        else:
            rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(self, env: VecEnv, callback, rollout_buffer: RolloutBuffer, n_rollout_steps: int):
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:    
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())

            self._update_info_buffer(infos)
            n_steps += 1

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs, actions, rewards, 
                self._last_episode_starts, values, log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device)) 
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "AWR",
    ):
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

        # callback.on_training_end()

        return self