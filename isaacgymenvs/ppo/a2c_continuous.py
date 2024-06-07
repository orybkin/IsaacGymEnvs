from isaacgymenvs.ppo import a2c_common
from isaacgymenvs.ppo import torch_ext
from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from isaacgymenvs.ppo import datasets

from torch import optim
import torch 


class A2CAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
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
        
        optim_kwargs = dict(lr=float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), **optim_kwargs)
        if self.model.a2c_network.num_critics > 1:
            self.base_params = list(self.model.parameters())
            self.extra_critic_params, self.extra_critic_optimizers = [], []
            for i in range(self.model.a2c_network.num_critics - 1):
                critic_mlp = self.model.a2c_network.extra_critic_mlps[i]
                value_fn = self.model.a2c_network.values[i + 1]
                extra_params = list(critic_mlp.parameters()) + list(value_fn.parameters())
                self.extra_critic_params.append(extra_params)
                self.extra_critic_optimizers.append(optim.Adam(extra_params, **optim_kwargs))
                self.base_params = [bp for bp in self.base_params if all(bp is not ep for ep in extra_params)]
            self.base_optimizer = optim.Adam(self.base_params, **optim_kwargs)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.relabel:
            self.relabeled_dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

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
        
    class StatsBuffer:
        def __init__(self, size):
            self.size = size
            self.buffer = {}
            
        def get(self, k, i=None):
            if i is None:
                assert all(self.buffer[k][i] is not None for i in range(self.size))
                return torch.stack(self.buffer[k])
            else:
                assert self.buffer[k][i] is not None
                return self.buffer[k][i]
            
        def update(self, k, v, i):
            if k not in self.buffer:
                self.buffer[k] = [None for _ in range(self.size)]
            self.buffer[k][i] = v

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
                
        # stats = A2CAgent.StatsBuffer(1)
        stats = A2CAgent.StatsBuffer(self.model.a2c_network.num_critics)

        def do_update(value_index):
            # for param in self.model.parameters():
            #     param.requires_grad = False
            if self.model.a2c_network.num_critics == 1:
                params = self.model.parameters()
                optimizer = self.optimizer
            elif value_index == 0:
                params = self.base_params
                optimizer = self.base_optimizer
            else:
                params = self.extra_critic_params[value_index - 1]
                optimizer = self.extra_critic_optimizers[value_index - 1]
            # for param in params:
            #     param.requires_grad = True
            
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                res_dict = self.model(batch_dict, value_index)                
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
                losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
                # for i in range(len(losses)): losses[i] /= stats.size
                a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
                loss = (a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef)
                
                stats.update('action_log_probs', action_log_probs, value_index)
                stats.update('mu', mu, value_index)
                stats.update('sigma', sigma, value_index)
                stats.update('entropy', entropy, value_index)
                stats.update('a_loss', a_loss, value_index)
                stats.update('b_loss', b_loss, value_index)
                stats.update('c_loss', c_loss, value_index)
                stats.update('clip_value_frac', clip_value_frac, value_index)
                
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None
            self.scaler.scale(loss).backward()
            
            # print(value_index, [name for name, param in self.model.named_parameters() if param.grad is not None])
            # breakpoint()
            
            # self.optimizer.step()
            
            #TODO: Refactor this ugliest code of the year
            self.truncate_gradients_and_step(optimizer, params)
            # self.truncate_gradients_and_step(self.optimizer, self.model.parameters())
        
        if not self.model.a2c_network.shared_encoder:
            for i in range(1, self.model.a2c_network.num_critics):
                do_update(i)
        do_update(0)
        
        stats.tensorfy()
        mu = stats.get('mu', 0)
        sigma = stats.get('sigma', 0)
        action_log_probs = stats.get('action_log_probs', 0)
        a_losses = stats.get('a_loss')
        b_losses = stats.get('b_loss')
        c_losses = stats.get('c_loss')
        entropies = stats.get('entropy')
        clip_value_fracs = stats.get('clip_value_frac')
        
        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
        
        diagnostics_batch = {
            'explained_variance': torch_ext.explained_variance(value_preds_batch, return_batch, rnn_masks).detach(),
            'clipped_fraction': torch_ext.policy_clip_fraction(action_log_probs, old_action_log_probs_batch, self.e_clip, rnn_masks).detach(),
            'clipped_value_fraction': clip_value_fracs.mean().detach()
        }
        for i, clip_value_frac in enumerate(clip_value_fracs):
            diagnostics_batch[f'clipped_value_fraction_{i}'] = clip_value_frac.detach()
        self.diagnostics.mini_batch(self, diagnostics_batch)

        losses_dict = {'a_loss': a_loss, 'c_loss': c_loss, 'entropy': entropy}
        for i, ci_loss in enumerate(ci_losses, 1):
            losses_dict[f'c{i}_loss'] = ci_loss
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

