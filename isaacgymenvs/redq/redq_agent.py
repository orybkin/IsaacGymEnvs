from isaacgymenvs.ppo.torch_ext import explained_variance

import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np

from sac.sac_agent import SACAgent

class REDQSacAgent(SACAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        
        config = self.config
        self.num_Q = config.get("num_Q", 2)
        self.num_min = config.get("num_min", 2)
        self.q_target_mode = config.get("q_target_mode", 'min')
        
    def _build_critic_optimizer(self):
        self.critic_optimizers = []
        for q_net in self.model.sac_network.critic.q_net_list:
            self.critic_optimizers.append(
                torch.optim.Adam(
                    q_net.parameters(),
                    lr=float(self.config["critic_lr"]),
                    betas=self.config.get("critic_betas", [0.9, 0.999])
                )
            )
            
    def _get_critic_optimizer_weights(self, state):
        for i, critic_optim in enumerate(self.critic_optimizers, 1):
            state[f'critic_optimizer_{i}'] = critic_optim.state_dict()
            
    def _set_critic_optimizer_weights(self, weights):
        for i, critic_optim in enumerate(self.critic_optimizers, 1):
            critic_optim.load_state_dict(weights[f'critic_optimizer_{i}'])
        
    def get_probabilistic_num_min(self, num_mins):
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
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            with torch.no_grad():
                next_action, _, _, _, _, _ = self.model.sac_network.actor(next_obs)
                next_action = next_action * self.action_scale
                target_Qs = torch.cat(self.model.sac_network.critic_target(next_obs, next_action, sample_idxs), 1)              
                if self.q_target_mode == 'min':
                    target_Q, _ = torch.min(target_Qs, dim=1, keepdim=True)
                    target_Q = reward + (not_done * self.gamma * target_Q)
                    target_Q = target_Q.detach()
                    target_Q = target_Q.expand((-1, self.num_Q)) if target_Q.shape[1] == 1 else target_Q
                else:
                    raise NotImplementedError()
            
            current_Qs = torch.cat(self.model.sac_network.critic(obs, action), dim=1)
            current_Q1 = current_Qs[0]
            
            squared_errors = F.mse_loss(current_Qs, target_Q, reduction='none')
            critic_losses = squared_errors.mean(dim=0)
        critic_loss = critic_losses.sum()
        
        info = {'losses/c_loss': critic_loss.detach(),
                'info/train_reward': reward.mean().detach(),
                'info/c_explained_variance': explained_variance(current_Q1, target_Q)}
        for i, loss in enumerate(critic_losses, 1):
            info[f'losses/c{i}_loss'] = loss.detach()

        if self.relabel_ratio > 0:
            bs = current_Q1.shape[0]
            real = int(bs * (1 - self.relabel_ratio))
            info['losses/c_loss_original'] = nn.MSELoss()(current_Q1[:real], target_Q[:real]).detach()
            info['losses/c_loss_relabeled'] = nn.MSELoss()(current_Q1[real:], target_Q[real:]).detach()

        return critic_loss, info
        
    def _critic_zero_grad(self):
        for o in self.critic_optimizers:
            o.zero_grad(set_to_none=True)
        
    def _critic_step(self):
        for o in self.critic_optimizers:
            o.step()

    def update_actor(self, obs):
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = False

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            action, _, _, log_prob, _, _ = self.model.sac_network.actor(obs)
            entropy = -log_prob.mean()
            action = action * self.action_scale
            actor_Qs = torch.cat(self.model.critic(obs, action), dim=1)
            actor_Q = torch.mean(actor_Qs, dim=1)
            actor_loss = torch.max(self.alpha.detach(), self.min_alpha) * log_prob - actor_Q
        actor_loss = actor_loss.mean()
        
        for p in self.model.sac_network.critic.parameters():
            p.requires_grad = True

        info = {'losses/a_loss': actor_loss.detach(), 
               'losses/entropy': entropy.detach(),
               'info/log_prob': log_prob.mean().detach(),
               'info/alpha': self.alpha.detach(),
               'info/actor_q': actor_Q.mean().detach(),
               'info/target_entropy': torch.ones(1) * self.target_entropy,}
        
        if self.relabel_ratio > 0:
            bs = actor_Q.shape[0]
            real = int(bs * (1 - self.relabel_ratio))
            info['info/actor_q'] = actor_Q[:real].mean().detach()
            info['info/actor_q_relabeled'] = actor_Q[real:].mean().detach()

        return actor_loss, info

    def act(self, obs, action_dim, sample=False):
        obs = self.preproc_obs(obs)
        action, _, _, _, _, _ = self.model.sac_network.actor(obs)
        action = action * self.action_scale
        action = action.clamp(*self.action_range)
        assert action.ndim == 2
        return action
    