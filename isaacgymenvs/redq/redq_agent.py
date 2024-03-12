from rl_games.algos_torch import torch_ext

from rl_games.common import vecenv
from rl_games.common import schedulers
from rl_games.common import experience
from isaacgymenvs.ppo.a2c_common import print_statistics
from isaacgymenvs.ppo import model_builder
from isaacgymenvs.ppo.torch_ext import explained_variance
from isaacgymenvs.sac import her_replay_buffer
from isaacgymenvs.sac import validation_replay_buffer
from isaacgymenvs.utils.rlgames_utils import Every, get_grad_norm, save_cmd

from rl_games.interfaces.base_algorithm import  BaseAlgorithm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch import optim
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import os
from collections import defaultdict

check_for_none = lambda x: None if x == 'None' else x

class REDQSacAgent(BaseAlgorithm):

    def __init__(self, base_name, params):
        self.config = config = params['config']
        print(config)

        self.load_networks(params)
        self.base_init(base_name, config)
        self.num_warmup_steps = config["num_warmup_steps"]
        self.gamma = config["gamma"]
        self.critic_tau = float(config["critic_tau"])
        self.batch_size = config["batch_size"]
        self.init_alpha = config["init_alpha"]
        self.learnable_temperature = config["learnable_temperature"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)
        self.gradient_steps = config.get("gradient_steps", 1)
        self.grad_norm = check_for_none(self.config.get('grad_norm', None))
        self.normalize_input = config.get("normalize_input", False)
        self.relabel_ratio = config.get("relabel_ratio", 0.0)
        self.relabel_ratio_random = config.get("relabel_ratio_random", 0.0)
        self.test_every_episodes = config.get('test_every_episodes', 10) 
        self.reset_every_steps = check_for_none(config.get('reset_every_steps', None))
        self.validation_ratio = config.get('validation_ratio', 0.0)
        self.policy_update_fraction = config.get('policy_update_fraction', 1)
        
        # TODO: double-check! To use bootstrap instead?
        self.max_env_steps = config.get("max_env_steps", 1000) # temporary, in future we will use other approach
        
        print(self.batch_size, self.num_actors, self.num_agents)

        self.num_frames_per_epoch = self.num_actors * self.num_steps_per_episode
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.action_scale = (action_space.high[0].item() - action_space.low[0].item()) / 2

        print("Number of Agents", self.num_actors, "Batch Size", self.batch_size)
        
        # REDQ
        self.policy_update_delay = config.get("policy_update_delay", 20)
        self.num_Q = config.get("num_Q", 2)
        self.num_min = config.get("num_min", 2)
        self.q_target_mode = config.get("q_target_mode", 'min')

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).float().to(self._device)
        self.log_alpha.requires_grad = True

        self.auto_alpha = config.get("auto_alpha", True)
        target_entropy = config.get("target_entropy", 'mbpo')
        if self.auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = -self.actions_num
            if target_entropy == 'mbpo':
                # TODO: fill this in for other environments
                mbpo_target_entropy_dict = {'Reacher-v2':-1,
                    'Hopper-v2':-1, 'HalfCheetah-v2':-3, 'Walker2d-v2':-3, 'Ant-v2':-4, 'Humanoid-v2':-2,
                    'Hopper-v3':-1, 'HalfCheetah-v3':-3, 'Walker2d-v3':-3, 'Ant-v3':-4, 'Humanoid-v3':-2,
                    'Hopper-v4':-1, 'HalfCheetah-v4':-3, 'Walker2d-v4':-3, 'Ant-v4':-4, 'Humanoid-v4':-2}
                self.target_entropy = mbpo_target_entropy_dict.get(config.get("env_name"), )
        else:
            self.target_entropy = None

        self.build_network()

        if self.relabel_ratio > 0.0 or self.relabel_ratio_random > 0.0:
            self.replay_buffer = validation_replay_buffer.ValidationHERReplayBuffer(self.env_info['observation_space'].shape,
                                                            self.env_info['action_space'].shape,
                                                            self.replay_buffer_size,
                                                            self.num_actors,
                                                            self._device,
                                                            self.vec_env.env,
                                                            self.rewards_shaper,
                                                            self.relabel_ratio,
                                                            self.relabel_ratio_random,
                                                            self.validation_ratio)
        else:
            self.replay_buffer = experience.VectorizedReplayBuffer(self.env_info['observation_space'].shape,
                                                                self.env_info['action_space'].shape,
                                                                self.replay_buffer_size,
                                                                self._device)

    def build_network(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        net_config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'normalize_input': self.normalize_input,
        }
    
        self.model = self.network.build(net_config)
        self.model.to(self._device)

        self.actor_optimizer = torch.optim.Adam(self.model.sac_network.actor.parameters(),
                                                lr=float(self.config['actor_lr']),
                                                betas=self.config.get("actor_betas", [0.9, 0.999]))
        
        self.critic_optimizers = []
        for q_net in self.model.sac_network.critic.q_net_list:
            self.critic_optimizers.append(
                torch.optim.Adam(
                    q_net.parameters(),
                    lr=float(self.config["critic_lr"]),
                    betas=self.config.get("critic_betas", [0.9, 0.999])
                )
            )

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=float(self.config["alpha_lr"]),
                                                    betas=self.config.get("alphas_betas", [0.9, 0.999]))

    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)

    def base_init(self, base_name, config):
        self.env_config = config.get('env_config', {})
        self.num_actors = config.get('num_actors', 1)
        self.env_name = config['env_name']
        print("Env name:", self.env_name)

        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()

        self._device = config.get('device', 'cuda:0')

        #temporary for Isaac gym compatibility
        self.ppo_device = self._device
        print('Env info:')
        print(self.env_info)

        self.rewards_shaper = config['reward_shaper']
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        #self.use_action_masks = config.get('use_action_masks', False)
        self.is_train = config.get('is_train', True)

        self.save_best_after = config.get('save_best_after', 500)
        self.print_stats = config.get('print_stats', True)
        self.rnn_states = None
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.save_freq = config.get('save_frequency', 0)

        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.obs_shape = self.observation_space.shape

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self._device)
        self.obs = None

        self.min_alpha = torch.tensor(np.log(1)).float().to(self._device)

        self.frame = 0
        self.epoch_num = 0
        self.update_time = 0
        self.last_mean_rewards = -1000000000
        self.play_time = 0
        self.update_num = 0

        # TODO: put it into the separate class
        pbt_str = ''
        self.population_based_training = config.get('population_based_training', False)
        if self.population_based_training:
            # in PBT, make sure experiment name contains a unique id of the policy within a population
            pbt_str = f'_pbt_{config["pbt_idx"]:02d}'
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + pbt_str + datetime.now().strftime("_%d-%H-%M-%S")
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        save_cmd(self.experiment_dir)

        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.writer = SummaryWriter(self.summaries_dir)
        print("Run Directory:", config['name'] + datetime.now().strftime("_%d-%H-%M-%S"))

        self.is_tensor_obses = False
        self.is_rnn = False
        self.last_rnn_indices = None
        self.last_state_indices = None

    def init_tensors(self):
        if self.observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32
        batch_size = self.num_agents * self.num_actors

        self.current_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.long, device=self._device)

        self.dones = torch.zeros((batch_size,), dtype=torch.uint8, device=self._device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def device(self):
        return self._device

    def get_weights(self):
        state = {
            'actor': self.model.sac_network.actor.state_dict(),
            'critic': self.model.sac_network.critic.state_dict(), 
            'critic_target': self.model.sac_network.critic_target.state_dict()
        }
        return state

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def set_weights(self, weights):
        self.model.sac_network.actor.load_state_dict(weights['actor'])
        self.model.sac_network.critic.load_state_dict(weights['critic'])
        self.model.sac_network.critic_target.load_state_dict(weights['critic_target'])

        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])

    def get_full_state_weights(self):
        state = self.get_weights()

        state['epoch'] = self.epoch_num
        state['frame'] = self.frame
        state['actor_optimizer'] = self.actor_optimizer.state_dict()
        for i in range(self.num_Q):
            state[f'critic_optimizer_{i}'] = self.critic_optimizers[i].state_dict()
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()        

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        for critic_optim in self.critic_optimizers:
            critic_optim.load_state_dict(weights[f'critic_optimizer_{i}'])
        self.log_alpha_optimizer.load_state_dict(weights['log_alpha_optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    def restore(self, fn, set_epoch=True):
        print("SAC restore")
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_param(self, param_name):
        pass

    def set_param(self, param_name, param_value):
        pass

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()
        
    def get_probabilistic_num_min(num_mins):
        # allows the number of min to be a float
        floored_num_mins = np.floor(num_mins)
        if num_mins - floored_num_mins > 0.001:
            prob_for_higher_value = num_mins - floored_num_mins
            if np.random.uniform(0, 1) < prob_for_higher_value:
                return int(floored_num_mins+1)
            else:
                return int(floored_num_mins)
        else:
            return num_mins

    def update_critic(self, obs, action, reward, next_obs, not_done):
        num_mins_to_use = self.get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
        with torch.no_grad():
            _, _, _, log_prob_a_tilda_next, _, _ = self.model.sac_network.actor(next_obs)
            q_targets = self.model.sac_network.critic_target(next_obs, sample_idxs)
            q_targets_cat = torch.cat(q_targets, 1)
            if self.q_target_mode == 'min':
                min_q, min_indices = torch.min(q_targets_cat, dim=1, keepdim=True)
                next_q_with_log_prob = min_q - self.alpha * log_prob_a_tilda_next
                y_q = reward + self.gamma * not_done * next_q_with_log_prob                
            else:
                raise NotImplementedError()
        
        q_current = self.model.sac_network.critic(next_obs)
        q_current = torch.cat(q_current, 1)
        current_Q1 = q_current[0]
        y_q_expanded = y_q.expand_as(q_current)
        
        squared_errors = F.mse_loss(q_current, y_q_expanded, reduction='none')
        critic_losses = squared_errors.mean(dim=tuple(range(1, squared_errors.ndim)))
        critic_loss = critic_losses.sum()
        
        for critic_optim in self.critic_optimizers:
            critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        
        info = {'losses/c_loss': critic_loss.detach(),
                'info/train_reward': reward.mean().detach(),
                'info/c_explained_variance': explained_variance(q_current[0], y_q)}
        for i, loss in enumerate(critic_losses, 1):
            info[f'losses/c{i}_loss'] = loss.detach()

        if self.relabel_ratio > 0:
            bs = q_current[0].shape[0]
            real = int(bs * (1 - self.relabel_ratio))
            info['losses/c_loss_original'] = nn.MSELoss()(current_Q1[:real], y_q[:real]).detach()
            info['losses/c_loss_relabeled'] = nn.MSELoss()(current_Q1[real:], y_q[real:]).detach()

        if self.grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.model.sac_network.critic.parameters(), self.grad_norm)
        else:
            grad_norm = get_grad_norm(self.model.sac_network.critic.parameters())
        info['info/grad_norm'] = grad_norm.detach()

        return critic_loss, info

    def update_actor_and_alpha(self, obs):
        self.model.sac_network.critic.requires_grad_(False)
        a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.model.sac_network.actor(obs)
        
        # actor loss
        q_a_tilda = self.model.sac_network.critic(obs)
        q_a_tilda = torch.cat(q_a_tilda, 1)
        ave_q = torch.mean(q_a_tilda, dim=1, keepdim=True)
        actor_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()

        # alpha loss
        if self.auto_alpha:  # TODO add this
            alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            alpha_loss = torch.Tensor([0])
            
        self.model.sac_network.critic.requires_grad_(True)
        
        info = {
            'losses/a_loss': actor_loss.detach(),
            'info/log_prob': log_prob_a_tilda.mean().detach(),
            'info/alpha': self.alpha.detach(),
            # 'info/actor_q': actor_Q.mean().detach(),
        }
        
        #TODO: figure out what relabeling means
        
        # if self.relabel_ratio > 0:
        #     bs = actor_Q.shape[0]
        #     real = int(bs * (1 - self.relabel_ratio))
        #     info['info/actor_q'] = actor_Q[:real].mean().detach()
        #     info['info/actor_q_relabeled'] = actor_Q[real:].mean().detach()

        return actor_loss, info

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1.0 - tau) * target_param.data)

    def update(self, step):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
        critic_loss, critic_loss_info = self.update_critic(obs, action, reward, next_obs, ~done)
        actor_loss_info = {}
        update_actor = step % self.policy_update_fraction == 0   #TODO: update this
        if update_actor:
            actor_loss, actor_loss_info = self.update_actor_and_alpha(obs)
        for critic_optim in self.critic_optimizers:
            critic_optim.step()
        if update_actor:
            self.actor_optimizer.step()

        self.soft_update_params(self.model.sac_network.critic, self.model.sac_network.critic_target,
                                     self.critic_tau)
        return actor_loss_info, critic_loss_info

    def validate(self):  
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size, validation=True)

        with torch.no_grad():
            critic_loss, critic_loss_info = self.update_critic(obs, action, reward, next_obs, ~done)
            actor_loss, actor_loss_info = self.update_actor_and_alpha(obs)
        
        info = {'val/a_loss': actor_loss_info['losses/a_loss'],
                'val/actor_q': actor_loss_info['info/actor_q'],
                'val/actor_q_relabeled': actor_loss_info['info/actor_q_relabeled'],
                'val/c_loss': critic_loss_info['losses/c_loss'],
                'val/c_loss_original': critic_loss_info['losses/c_loss_original'],
                'val/c_loss_relabeled': critic_loss_info['losses/c_loss_relabeled'],}

        return info


    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']
        # obs = self.model.norm_obs(obs)

        return obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(self.observation_space.dtype != np.int8)
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self._device)
            else:
                obs = torch.FloatTensor(obs).to(self._device)

        return obs

    # TODO: move to common utils
    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}

        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)

        return upd_obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        return actions

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, terminated, truncated, infos = self.vec_env.step(actions) # (obs_space) -> (n, obs_space)

        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self._device), terminated.to(self._device), truncated.to(self._device), infos
        else:
            return torch.from_numpy(obs).to(self._device).float(), torch.from_numpy(rewards).to(self._device), torch.from_numpy(terminated).to(self._device), torch.from_numpy(truncated).to(self._device), infos

    def env_reset(self):
        with torch.no_grad():
            obs = self.vec_env.reset()

        obs = self.obs_to_tensors(obs)

        return obs

    def act(self, obs, action_dim, sample=False):
        obs = self.preproc_obs(obs)
        dist = self.model.sac_network.actor(obs)

        actions = dist.sample() if sample else dist.mean
        actions = actions * self.action_scale
        actions = actions.clamp(*self.action_range)
        assert actions.ndim == 2

        return actions

    def clear_stats(self):
        self.game_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.algo_observer.after_clear_stats()

    def play_steps(self, random_exploration = False):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_metrics = defaultdict(list)
        critic_metrics = defaultdict(list)

        for s in range(self.num_steps_per_episode):
            obs = self.obs
            if isinstance(obs, dict):
                obs = obs['obs']
            self.set_eval()
            if random_exploration:
                action = torch.rand((self.num_actors, *self.env_info["action_space"].shape), device=self._device) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            step_start = time.time()

            with torch.no_grad():
                next_obs, rewards, terminated, truncated, infos = self.env_step(action)

            if isinstance(next_obs, dict):
                next_obs = next_obs['obs']
            step_end = time.time()

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += (step_end - step_start)
            step_time += (step_end - step_start)

            dones = terminated + truncated
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            # if done_indices.numel() > 0:
            #     print(self.current_lengths[done_indices].mean())
            self.game_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()

            self.algo_observer.process_infos(infos, done_indices)

            no_timeouts = self.current_lengths != self.max_env_steps
            dones = dones * no_timeouts

            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            self.obs = next_obs.clone()
            rewards = self.rewards_shaper(rewards)

            self.replay_buffer.add(obs, action, torch.unsqueeze(rewards, 1), next_obs, torch.unsqueeze(terminated, 1), torch.unsqueeze(dones, 1))

            if not random_exploration:
                self.set_train()
                update_time_start = time.time()
                for _ in range(self.gradient_steps):
                    actor_loss_info, critic_loss_info = self.update(self.update_num)
                    for key, value in actor_loss_info.items(): actor_metrics[key].append(value)
                    for key, value in critic_loss_info.items(): critic_metrics[key].append(value)
                    self.update_num += 1
                update_time_end = time.time()
                update_time = update_time_end - update_time_start
            else:
                update_time = 0

            total_update_time += update_time

            if dones.any():
                obs = self.env_reset()
                if isinstance(obs, dict):
                    obs = obs['obs']
                self.obs[dones.bool()] = obs[dones.bool()]

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        play_time = total_time - total_update_time

        return step_time, play_time, total_update_time, total_time, actor_metrics, critic_metrics

    def train_epoch(self):
        random_exploration = self.epoch_num < self.num_warmup_steps / self.num_steps_per_episode
        return self.play_steps(random_exploration)

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        test_check = Every(self.test_every_episodes * (self.vec_env.env.max_episode_length-1))
        render_check = Every(self.vec_env.env.render_every_episodes * (self.vec_env.env.max_episode_length-1))
        reset_check = Every(self.reset_every_steps)
        total_time = 0

        self.obs = self.env_reset()

        while True:
            if reset_check.check(self.frame):
                print('Reset network!')
                self.build_network()
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, actor_metrics, critic_metrics = self.train_epoch()

            total_time += epoch_total_time

            curr_frames = self.num_frames_per_epoch
            self.frame += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time
            
            if self.epoch_num % 1000 == 0:
                self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
                self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
                self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
                self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
                self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
                self.writer.add_scalar('performance/step_time', step_time, self.frame)

                print_statistics(self.print_stats, curr_frames, step_time, play_time, epoch_total_time, 
                    self.epoch_num, self.max_epochs, self.frame, self.max_frames, self.game_rewards.get_mean())
            
                if self.epoch_num >= self.num_warmup_steps:
                    for key, value in critic_metrics.items():
                        if value[0] is not None:
                            self.writer.add_scalar(key, torch_ext.mean_list(value).item(), self.frame)
                    for key, value in actor_metrics.items():
                        if value[0] is not None:
                            self.writer.add_scalar(key, torch_ext.mean_list(value).item(), self.frame)

                self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
                self.writer.add_scalar('info/updates', self.update_num, self.frame)
                self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

                if self.validation_ratio > 0.0:
                    val_info = self.validate()
                    for key, value in val_info.items():
                        self.writer.add_scalar(key, value.item(), self.frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()

                    self.writer.add_scalar('rewards/step', mean_rewards, self.frame)
                    self.writer.add_scalar('rewards/time', mean_rewards, total_time)
                    self.writer.add_scalar('episode_lengths/step', mean_lengths, self.frame)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)
                    checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)

                    should_exit = False

                    if self.save_freq > 0:
                        if self.epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                            print('Maximum reward achieved. Network won!')
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                    if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                        if self.game_rewards.current_size == 0:
                            print('WARNING: Max epochs reached before any env terminated at least once')
                            mean_rewards = -np.inf

                        self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
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

                    if should_exit:
                        return self.last_mean_rewards, self.epoch_num
                
            # Test
            iteration = self.frame / self.num_actors
            if test_check.check(iteration):
                print("Testing...")
                self.test(render=render_check.check(iteration))
                self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time, '_test')
                print("Done Testing.")

    def test(self, render):
        self.set_eval()
        self.vec_env.env.test = True
        if render:
            self.vec_env.env.override_render = True

        obs = self.env_reset()
        if isinstance(obs, dict):
            obs = obs['obs']
        self.obs = obs

        for n in range(self.vec_env.env.max_episode_length - 1):
            with torch.no_grad():
                action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)

            obs, rewards, terminated, truncated, infos = self.env_step(action)
            if isinstance(obs, dict):
                obs = obs['obs']

            # Save images
            dones = terminated + truncated
            all_done_indices = dones.nonzero(as_tuple=False)[::self.num_agents]
            self.algo_observer.process_infos(infos, all_done_indices)

        self.vec_env.env.test = False
        if render:
            self.vec_env.env.override_render = False

        obs = self.env_reset()
        if isinstance(obs, dict):
            obs = obs['obs']
        self.obs = obs