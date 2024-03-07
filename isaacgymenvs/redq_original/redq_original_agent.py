from rl_games.algos_torch import torch_ext

from rl_games.common import vecenv
from rl_games.common import schedulers
from rl_games.common import experience
from isaacgymenvs.ppo.a2c_common import print_statistics
from isaacgymenvs.ppo import model_builder
from isaacgymenvs.sac import her_replay_buffer
from isaacgymenvs.utils.rlgames_utils import Every

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

from torch import Tensor
from isaacgymenvs.redq_original.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer

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


class REDQAgent():

    def __init__(self, base_name, params):

        # IG
        self.config = config = params['config']
        self.load_networks(params)
        self.base_init(base_name, config)
        self.test_every_episodes = config.get('test_every_episodes', 10)      
        self.num_steps_per_episode = config.get("num_steps_per_episode", 1)  


        # IG interface
        self.action_space = self.env_info['action_space']
        obs_dim = self.env_info['observation_space'].shape[0]
        act_dim = self.action_space.shape[0]
        act_limit = self.action_space.high[0].item()
        device = config.get('device', 'cuda:0')
        self.num_frames_per_epoch = self.num_actors
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        # REDQ
        self.replay_size = config["replay_buffer_size"]
        self.batch_size=config["batch_size"]
        self.utd_ratio=config.get("gradient_steps", 1)
        policy_update_delay=config.get("policy_update_delay", 1)    
        self.relabel_ratio = config.get("relabel_ratio", 0.0)
        self.entropy_backup = config.get("entropy_backup", True)
        hidden_sizes=params['network']['mlp']['units']
        lr=config["actor_lr"]
        gamma=0.99
        polyak=0.995
        alpha=0.2
        auto_alpha=True
        start_steps=5000
        delay_update_steps='auto'
        num_Q=2
        # utd_ratio=20
        # num_Q=10
        num_min=2
        # set up networks
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(num_Q):
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = []
        for q_i in range(num_Q):
            self.q_optimizer_list.append(optim.Adam(self.q_net_list[q_i].parameters(), lr=lr))
        # set up adaptive entropy (SAC adaptive)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = - act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
        # set up replay buffer
        # set up other things
        self.mse_criterion = nn.MSELoss()

        if self.relabel_ratio > 0.0:
            self.replay_buffer = her_replay_buffer.HERReplayBuffer(self.env_info['observation_space'].shape,
                                                                self.env_info['action_space'].shape,
                                                                self.replay_size,
                                                                self.num_actors,
                                                                self._device,
                                                                self.vec_env.env,
                                                                self.relabel_ratio)
        else:
            self.replay_buffer = experience.VectorizedReplayBuffer(self.env_info['observation_space'].shape,
                                                                self.env_info['action_space'].shape,
                                                                self.replay_size,
                                                                self._device)

        # store other hyperparameters
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.num_min = num_min
        self.num_Q = num_Q
        self.delay_update_steps = self.start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.policy_update_delay = policy_update_delay
        self.device = device
    

        # interface
        self.num_warmup_steps = self.delay_update_steps

        

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
        self.ep_ret = 0

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

    def get_weights(self):
        state = {'actor': self.model.sac_network.actor.state_dict(),
         'critic': self.model.sac_network.critic.state_dict(), 
         'critic_target': self.model.sac_network.critic_target.state_dict()}
        return state

    def save(self, fn):
        return
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
        state['critic_optimizer'] = self.critic_optimizer.state_dict()
        state['log_alpha_optimizer'] = self.log_alpha_optimizer.state_dict()        

        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)

        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        self.actor_optimizer.load_state_dict(weights['actor_optimizer'])
        self.critic_optimizer.load_state_dict(weights['critic_optimizer'])
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
        self.policy_net.eval()

    def set_train(self):
        self.policy_net.train()

    def preproc_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs['obs']

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
        if not self.is_tensor_obses and isinstance(actions, torch.Tensor):
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
        actions = self.policy_net.forward(obs, deterministic=not sample, return_log_prob=False)[0]
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
        critic1_losses = []
        critic2_losses = []

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
            # if len(done_indices) > 0:
            #     print('return', self.current_rewards[done_indices])
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            self.algo_observer.process_infos(infos, done_indices)

            self.current_rewards = self.current_rewards * (1 - dones.float())
            self.current_lengths = self.current_lengths * (1 - dones.float())

            self.obs = next_obs.clone()

            self.replay_buffer.add(obs, action, torch.unsqueeze(rewards, 1), next_obs, torch.unsqueeze(terminated, 1))

            if not random_exploration:
                self.set_train()
                update_time_start = time.time()
                actor_loss_info, critic1_loss, critic2_loss = self.update(self.epoch_num)
                critic1_losses.append(critic1_loss)
                for key, value in actor_loss_info.items(): actor_metrics[key].append(value)
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
        self.frame += self.num_frames_per_epoch

        return step_time, play_time, total_update_time, total_time, actor_metrics, critic1_losses, critic2_losses

    def get_exploration_action(self, obs, env):
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            if self.replay_buffer.size > self.start_steps:
                if isinstance(obs, np.ndarray):
                    obs = torch.Tensor(obs).to(self.device)
                obs = obs.unsqueeze(0)
                # action_tensor = self.policy_net.forward(obs, deterministic=False,
                #                              return_log_prob=False)[0]
                action_tensor = self.act(obs[0], self.env_info["action_space"].shape, sample=True)
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                # action = self.action_space.sample()
                action = torch.rand(self.env_info["action_space"].shape, device=self._device) * 2.0 - 1.0
        return action
    

    def train_epoch(self):
        random_exploration = self.frame < self.num_warmup_steps
        return self.play_steps(random_exploration)

    def train(self):
        self.init_tensors()
        self.algo_observer.after_init(self)
        test_check = Every(self.test_every_episodes * self.vec_env.env.max_episode_length)
        render_check = Every(self.vec_env.env.render_every_episodes * self.vec_env.env.max_episode_length)
        total_time = 0
        # rep_count = 0

        self.obs = self.env_reset()

        while True:
            self.epoch_num += 1
            step_time, play_time, update_time, epoch_total_time, actor_metrics, critic1_losses, critic2_losses = self.train_epoch()

            total_time += epoch_total_time

            curr_frames = self.num_frames_per_epoch

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time

            self.writer.add_scalar('performance/step_inference_rl_update_fps', fps_total, self.frame)
            self.writer.add_scalar('performance/step_inference_fps', fps_step_inference, self.frame)
            self.writer.add_scalar('performance/step_fps', fps_step, self.frame)
            self.writer.add_scalar('performance/rl_update_time', update_time, self.frame)
            self.writer.add_scalar('performance/step_inference_time', play_time, self.frame)
            self.writer.add_scalar('performance/step_time', step_time, self.frame)
            
            if self.epoch_num % 100 == 0:
                print_statistics(self.print_stats, curr_frames, step_time, play_time, epoch_total_time, 
                    self.epoch_num, self.max_epochs, self.frame, self.max_frames, self.game_rewards.get_mean())

                if self.frame >= self.num_warmup_steps:
                    self.writer.add_scalar('losses/c1_loss', torch_ext.mean_list(critic1_losses).item(), self.frame)
                    for key, value in actor_metrics.items():
                        if value[0] is not None:
                            self.writer.add_scalar(key, torch_ext.mean_list(value).item(), self.frame)

                self.writer.add_scalar('info/epochs', self.epoch_num, self.frame)
                self.writer.add_scalar('info/updates', self.update_num, self.frame)
                self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time)

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
                
            # # Test
            # iteration = self.frame / self.num_actors
            # if test_check.check(iteration):
            #     print("Testing...")
            #     self.test(render=render_check.check(iteration))
            #     self.algo_observer.after_print_stats(self.frame, self.epoch_num, total_time, '_test')
            #     print("Done Testing.")

    def test(self, render):
        self.set_eval()
        self.vec_env.env.test = True
        if render:
            self.vec_env.env.override_render = True
        self.vec_env.env.reset_idx()
        obs = self.obs
        if isinstance(obs, dict):
            obs = self.obs['obs'] 

        for n in range(self.vec_env.env.max_episode_length):
            with torch.no_grad():
                action = self.act(obs.float(), self.env_info["action_space"].shape, sample=True)
        
            obs, rewards, dones, infos = self.env_step(action)

            # Save images
            all_done_indices = dones.nonzero(as_tuple=False)[::self.num_agents]
            self.algo_observer.process_infos(infos, all_done_indices)

        self.vec_env.env.test = False
        if render:
            self.vec_env.env.override_render = False
        self.vec_env.env.reset_idx()

    def get_redq_q_target_no_grad(self, obs_next_tensor, rews_tensor, done_tensor):
        # compute REDQ Q target, depending on the agent's Q target mode
        # allow min as a float:
        num_mins_to_use = get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
        with torch.no_grad():
            """Q target is min of a subset of Q values"""
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
            q_prediction_next_list = []
            for sample_idx in sample_idxs:
                q_prediction_next = self.q_target_net_list[sample_idx](torch.cat([obs_next_tensor, a_tilda_next], 1))
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            next_q_with_log_prob = min_q
            if self.entropy_backup:
                next_q_with_log_prob = min_q - self.alpha * log_prob_a_tilda_next
            y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
        return y_q, sample_idxs


    def sample_data(self, batch_size):
        # sample data from replay buffer
        batch = self.replay_buffer.sample_batch(batch_size)
        return batch['obs1'], batch['obs2'], batch['acts'], batch['rews'].unsqueeze(1), batch['done'].unsqueeze(1)
    
    def update(self, step):
        # this function is called after each datapoint collected.
        # when we only have very limited data, we don't make updates

        actor_loss_info = {}
        num_update = 0 if self.frame < self.delay_update_steps - 1 else self.utd_ratio
        for i_update in range(num_update):
            self.update_num += 1

            obs_tensor, acts_tensor, rews_tensor, obs_next_tensor, done_tensor = self.replay_buffer.sample(self.batch_size)
            done_tensor = done_tensor.float()

            """Q loss"""
            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            """policy and alpha loss"""
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                # get policy loss
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)

                # get alpha loss
                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])
                
                actor_loss_info = {'losses/a_loss': policy_loss.cpu(),
                                   'losses/entropy': (-log_prob_a_tilda.mean()).cpu(),
                                   'losses/alpha_loss': alpha_loss.cpu(),
                                   'info/alpha': torch.ones(1) * self.alpha,
                                   'info/actor_q': q_prediction.mean().detach().cpu(),
                                   'info/target_entropy': torch.ones(1) * self.target_entropy,}

            """update networks"""
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                self.policy_optimizer.step()

            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            # by default only log for the last update out of <num_update> updates
            # if i_update == num_update - 1:
            #     logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
            #                  LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction.detach().cpu().numpy(),
            #                  Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
            #                  PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

        # # if there is no update, log 0 to prevent logging problems
        # if num_update == 0:
        #     logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

        return actor_loss_info, q_loss_all.cpu() / self.num_Q, 0 
