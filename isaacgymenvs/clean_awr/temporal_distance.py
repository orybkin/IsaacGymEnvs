# A simple MLP predicting temporal distances

import torch
import torch.nn as nn
import torch.nn.functional as F
from isaacgymenvs.clean_awr.awr_networks import _build_sequential_mlp
from torch.utils.data import Dataset

class TemporalDistanceNetwork(nn.Module):
    def __init__(self, units, goal_idx, output_size, classifier_selection=None, regression_coef=None):
        """
        classifier_selection is used iff `regression_coef is None`
        regression is used iff `regression_coef is not None`
        """
        nn.Module.__init__(self)
        self.units = units
        self.mlp = _build_sequential_mlp(len(goal_idx) * 2, units)
        self.goal_idx = goal_idx
        self.do_regression = regression_coef is not None
        if self.do_regression:
            self.regression_coef = regression_coef
            self.regression_mlp = nn.Linear(units[-1], 1)
            self.positive_clf = nn.Linear(units[-1], 1)
        else:
            assert classifier_selection is not None
            self.classifier_selection = classifier_selection
            self.mlp.add_module('linear', nn.Linear(units[-1], output_size))

        for m in self.modules():         
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, *args):
        out = torch.cat(args, dim=1)
        out = out.flatten(1)
        if self.do_regression:
            out = self.regression_mlp(self.mlp(out))
            is_positive = self.positive_clf(self.mlp(out))
        else:
            out = self.mlp(out)
            if self.classifier_selection == 'mode':
                pred = torch.argmax(out, dim=1)
            elif self.classifier_selection == 'mean':
                pred = torch.mean(out, dim=1).long()
            else:
                raise ValueError()
            return out, pred
            
        
    def loss(self, pair):
        goal = pair['goal']
        future_goal = pair['future_goal']
        distance = pair['distance']

        out, pred = self.forward(goal, future_goal)
        if self.do_regression:
            ...
        else:
            loss = F.cross_entropy(out, distance)
            accuracy = torch.mean((pred == distance).float())
        return {'loss': loss, 'accuracy': accuracy}


class TemporalDistanceDataset(Dataset):
    def __init__(self, buffer, config, env):
        self.config = config
        self.env = env
        self.minibatch_size = self.config['horizon_length'] * self.config['num_actors'] // self.config['num_minibatches']
        self.device = buffer.device

        # Build dataset
        pairs = self.get_positive_pairs(buffer)
        negative_pairs = {k: v.flip(1) for k,v in pairs.items()}
        negative_pairs['distance'][:] = self.config['relabel_every']  # pairs['distance'] in 0 ... relabel_every - 1
        pairs = {k: v.flatten(0, 1) for k,v in pairs.items()}
        negative_pairs = {k: v.flatten(0, 1) for k,v in negative_pairs.items()}
        negative_fraction = int(pairs['goal'].shape[0] * self.config['temporal_distance']['negative_pairs_frac'])
        pairs = {k: torch.cat([v, negative_pairs[k][:negative_fraction]], dim=0) for k,v in pairs.items()}
        pairs['is_positive'] = pairs['distance'] < self.config['relabel_every']
            
        self.pairs = pairs
        self.batch_size = self.pairs['goal'].shape[0]
        self.length = self.batch_size // self.minibatch_size
        # self.idx = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        self.idx = torch.randperm(self.batch_size, dtype=torch.long, device=self.device)

    def get_positive_pairs(self, buffer):
        batch = self.config['num_actors']
        horizon = self.config['horizon_length']
        relabel_every = self.config.get('relabel_every', horizon)
        obs = buffer.tensor_dict['obses']
        dones = buffer.tensor_dict['dones']
        goal = obs[:, :, self.env.achieved_idx]

        end_idx = self.get_relabel_idx(self.env, dones)[:, :, 0] # episode end indices
        future_step = torch.randint(0, relabel_every, (horizon, batch), device=self.device) 
        future_idx = torch.arange(horizon, device=self.device)[:, None] + future_step
        future_idx = torch.clamp(torch.minimum(future_idx, end_idx), max=horizon-1)[:,:,None].repeat(1,1,len(self.env.achieved_idx))
        future_goal = torch.gather(goal, 0, future_idx)
        return {'goal': goal, 'future_goal': future_goal, 'distance': future_step}

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
