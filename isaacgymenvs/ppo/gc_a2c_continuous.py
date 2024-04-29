from isaacgymenvs.ppo import a2c_continuous, a2c_common
from isaacgymenvs.ppo.a2c_common import awr_loss, bc_loss
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
        self.actor_loss_func[1] = bc_loss
        self.awr_coef = self.config.get('awr_coef', 1.0)
        self.awr_critic_coef = self.config.get('awr_critic_coef', 1.0)
        self.awr_actor_coef = self.config.get('awr_actor_coef', 1.0)

    def train_actor_critic(self, original_dict, relabeled_dict):
        loss = [0, 0]
        loss_coef = [1.0, self.awr_coef]
        loss_critic_coef = [1.0, self.awr_critic_coef]
        loss_actor_coef = [1.0, self.awr_actor_coef]
        losses_dict = {}
        diagnostics = {}
        for i, input_dict in enumerate([original_dict, relabeled_dict]):
            # if i == 0: continue
            value_preds_batch = input_dict['old_values']
            old_action_log_probs_batch = input_dict['old_logp_actions']
            advantage = input_dict['advantages']
            old_mu_batch = input_dict['mu']
            old_sigma_batch = input_dict['sigma']
            return_batch = input_dict['returns']
            batch_dict = {
                'is_train': True,
                'prev_actions': input_dict['actions'], 
                'obs' : self._preproc_obs(input_dict['obs']),
            }
            lr_mul = 1.0

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

                a_loss = self.actor_loss_func[i](old_action_log_probs_batch, action_log_probs, advantage, self.ppo, self.e_clip)

                if self.has_value_loss:
                    c_loss, clip_value_frac = a2c_common.critic_loss(value_preds_batch, values, self.e_clip, return_batch, self.clip_value)
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

                loss[i] = loss_coef[i] * (a_loss * loss_actor_coef[i]
                                            + c_loss * loss_critic_coef[i] * self.critic_coef
                                            - entropy * self.entropy_coef
                                            + b_loss * self.bounds_loss_coef)
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
            diagnostics.update({
                f'explained_variance{identifier}': torch_ext.explained_variance(value_preds_batch, return_batch, rnn_masks).detach(),
                f'clipped_fraction{identifier}': torch_ext.policy_clip_fraction(action_log_probs, old_action_log_probs_batch, self.e_clip, rnn_masks).detach(),
                f'clipped_value_fraction{identifier}': clip_value_frac,
            })

            # print(f'values train {identifier}: {values.mean().item():.3f}')

        self.scaler.scale(loss[0] + loss[1]).backward()
        #TODO: Refactor this ugliest code of the year
        self.truncate_gradients_and_step()

        self.diagnostics.mini_batch(self, diagnostics)      

        return losses_dict, kl_dist, self.last_lr, lr_mul, mu.detach(), sigma.detach()
