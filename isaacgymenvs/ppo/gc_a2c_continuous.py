from isaacgymenvs.ppo import a2c_continuous
from isaacgymenvs.ppo.a2c_common import awr_loss
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 

from functools import partial

class GCA2CAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        self.algo = self.config.get('algo', 'ppo')
        if self.algo == 'pawr':
            self.actor_loss_func = [None, None]
            self.actor_loss_func[0] = common_losses.actor_loss
            self.actor_loss_func[1] = partial(awr_loss, temperature=self.config['awr_temperature'])
        else:
            self.actor_loss_func = [self.actor_loss_func, self.actor_loss_func]

    def train_actor_critic(self, original_dict, relabeled_dict):
        loss = [0, 0]
        losses_dict = {}
        for i, input_dict in enumerate([original_dict, relabeled_dict]):
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
                res_dict = self.model(batch_dict, value_index=i)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                a_loss = self.actor_loss_func[i](old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

                if self.has_value_loss:
                    c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
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

                loss[i] = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
                from torch.nn import functional
                torch.nn.functional.linear
                if self.multi_gpu:
                    self.optimizer.zero_grad()
                else:
                    for param in self.model.parameters():
                        param.grad = None

            with torch.no_grad():
                reduce_kl = rnn_masks is None
                kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
                if rnn_masks is not None:
                    kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

            identifier = '' if i == 0 else '_relabeled'
            losses_dict.update({f'a_loss{identifier}': a_loss, f'c_loss{identifier}': c_loss, f'entropy{identifier}': entropy})
            if self.bounds_loss_coef is not None:
                losses_dict[f'bounds_loss{identifier}'] = b_loss

        self.scaler.scale(loss[0] + loss[1]).backward()
        #TODO: Refactor this ugliest code of the year
        self.truncate_gradients_and_step()

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        return losses_dict, kl_dist, self.last_lr, lr_mul, mu.detach(), sigma.detach()
