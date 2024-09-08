# A simple MLP predicting temporal distances

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaacgymenvs.clean_awr.awr_networks import _build_sequential_mlp
from torch.utils.data import Dataset

class TemporalDistanceNetwork(nn.Module):
    def __init__(self, 
                 units, 
                 input_size, 
                 output_size, 
                 td_idx, 
                 achieved_idx, 
                 logsumexp_alpha=None,
                 classifier_selection=None):
        nn.Module.__init__(self)
        self.units = units
        self.input_size = input_size
        self.output_size = output_size
        self.td_idx = td_idx
        self.achieved_idx = achieved_idx
        self.logsumexp_alpha = logsumexp_alpha
        self.do_classification = classifier_selection is not None
        self.mlp = _build_sequential_mlp(input_size, units)
        
        if self.do_classification:
            self.default_classifier_selection = classifier_selection
        self.mlp.add_module('linear', nn.Linear(units[-1], output_size))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, *args, classifier_selection=None):
        out = torch.cat(args, dim=1)
        out = out.flatten(1)
        
        if self.do_classification:
            out = self.mlp(out)
            if classifier_selection is None:
                classifier_selection = self.default_classifier_selection
            if classifier_selection == 'mode':
                pred = torch.argmax(out, dim=1)
            elif classifier_selection == 'mean':
                probs = F.softmax(out, dim=1)
                pred = (torch.arange(self.output_size, device=out.device)[None, :] * probs).sum(dim=1)
            elif classifier_selection == 'logsumexp':
                log_probs = F.log_softmax(out, dim=1)
                exp_term = -torch.arange(self.output_size, device=out.device) / self.logsumexp_alpha
                pred = -self.logsumexp_alpha * torch.logsumexp(log_probs + exp_term[None, :], dim=1)
            else:
                raise ValueError()
            return out, pred
        else:
            pred = self.mlp(out).flatten()
            return pred
        
    def _round(self, t):
        return torch.round(t).long() if t.dtype == torch.float else t
        
    def loss(self, pair):
        goal = pair['goal'][:, self.td_idx]
        future_goal = pair['future_goal']
        target = pair['distance']
        euclidean = torch.norm(pair['goal'][:, self.achieved_idx] - future_goal, dim=1)

        if self.do_classification:
            out, pred = self.forward(goal, future_goal)
            loss = F.cross_entropy(out, target)
            res_dict = {'ce': loss}
        else:
            pred = self.forward(goal, future_goal)
            loss = F.mse_loss(pred, target.float())
            res_dict = {'mse': loss}
        accuracy = torch.mean((self._round(pred) == target).float())
        euclid_corr = torch.corrcoef(torch.stack([pred, euclidean]))[0, 1]
        return {**res_dict, 'accuracy': accuracy, 'euclid_corr': euclid_corr, 'pred': pred, 'euclidean': euclidean}


class TemporalDistanceDataset(Dataset):
    def __init__(self, buffer, config, env):
        self.config = config
        self.env = env
        self.minibatch_size = self.config['horizon_length'] * self.config['num_actors'] // self.config['num_minibatches']
        self.device = buffer.device

        # Build dataset
        pairs = self.get_positive_pairs(buffer, self.config['temporal_distance']['neg_goal_selection'])
        negative_pairs = {k: v.flip(1) for k,v in pairs.items()}  # TODO: sample randomly instead of flip
        negative_pairs['distance'][:] = max(config['relabel_every'], config['horizon_length'])
        pairs = {k: v.flatten(0, 1) for k,v in pairs.items()}
        negative_pairs = {k: v.flatten(0, 1) for k,v in negative_pairs.items()}
        negative_fraction = int(pairs['goal'].shape[0] * self.config['temporal_distance']['negative_pairs_frac'])
        pairs = {k: torch.cat([v, negative_pairs[k][:negative_fraction]], dim=0) for k,v in pairs.items()}
        
        self.pairs = pairs
        self.batch_size = self.pairs['goal'].shape[0]
        self.length = self.batch_size // self.minibatch_size
        # self.idx = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        self.idx = torch.randperm(self.batch_size, dtype=torch.long, device=self.device)
        
    def _sample_derangement(self, n):
        """Randomly samples a derangement (while loop takes e iterations in expectation)"""
        while True:
            p = torch.randperm(n)
            if not torch.any(p == torch.arange(len(p))):
                return p.to(self.device)
        
    def _torch_randint(self, low, high, size):
        return torch.randint(2**63 - 1, size=size, device=self.device) % (high - low) + low

    def get_positive_pairs(self, buffer, goal_selection):
        assert goal_selection in ('achieved', 'commanded')
        batch = self.config['num_actors']
        horizon = self.config['horizon_length']
        relabel_every = self.config.get('relabel_every', horizon)  # unused
        obs = buffer.tensor_dict['obses']
        dones = buffer.tensor_dict['dones']
        
        end_idx = self.get_relabel_idx(self.env, dones)[:, :, 0] # episode end indices
        distance = self._torch_randint(0, end_idx + 1, end_idx.shape)
        future_idx = self._torch_randint(distance, end_idx + 1, end_idx.shape)
        goal_idx = future_idx - distance
        goal = torch.gather(obs, 0, goal_idx[:,:,None].tile(obs.shape[-1]))
        future_goal = torch.gather(
            obs[:, :, self.env.achieved_idx if goal_selection == 'achieved' else self.env.desired_idx], 
            0, 
            future_idx[:,:,None].tile(len(self.env.achieved_idx)))
        
        if self.config['temporal_distance']['objective'] == 'temporal':
            final_distance = distance
        elif self.config['temporal_distance']['objective'] == 'euclidean':
            final_distance = torch.norm(goal[:, :, self.env.achieved_idx] - future_goal, dim=2)
        else:
            raise ValueError()
        
        return {'goal': goal, 'future_goal': future_goal, 'distance': final_distance}

    def get_relabel_idx(self, env, dones):
        """ copied from awr.py """
        dones = dones.clone()
        dones[-1] = 1 # for last episode, use last existing frame
        next_done_idx = dones.shape[0] - 1 - dones.flip(0).cummax(0).indices.flip(0)
        next_done_idx = next_done_idx[:, :, None].repeat(1, 1, len(env.achieved_idx))
        return next_done_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k,v in self.pairs.items():
            if v is not None:
                if type(v) is dict:
                    v_dict = { kd:vd[self.idx[start:end]] for kd, vd in v.items() }
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[self.idx[start:end]]
                
        return input_dict
