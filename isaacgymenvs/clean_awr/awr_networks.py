from rl_games.common import vecenv
from rl_games.algos_torch import torch_ext
import os
import torch 
from torch import nn
from torch import optim
import numpy as np
import time
import gym
import math
import copy
from collections import defaultdict
from tensorboardX import SummaryWriter
from functools import partial

from torch.utils.data import Dataset
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs

def _build_sequential_mlp(input_size, units):
    print('build mlp:', input_size)
    in_size = input_size
    layers = []
    for unit in units:
        layers.append(torch.nn.Linear(in_size, unit))
        layers.append(nn.ELU())
        in_size = unit
    return nn.Sequential(*layers)

def _neglogp(x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

class AWRNetwork(nn.Module):
    def __init__(self, actions_num, input_shape, num_seqs, units, fixed_sigma, normalize_value, normalize_input):
        nn.Module.__init__(self)

        self.num_seqs = num_seqs
        self.units = units
        self.fixed_sigma = fixed_sigma
        assert self.fixed_sigma == True
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input

        self.actor_mlp = _build_sequential_mlp(input_shape[0], units)
        self.value = torch.nn.Linear(units[-1], 1)
        self.mu = torch.nn.Linear(units[-1], actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
        nn.init.constant_(self.sigma, 0)

        if normalize_value:
            self.value_mean_std = RunningMeanStd((1,)) #   GeneralizedMovingStats((self.value_size,)) #   
        if normalize_input:
            if isinstance(input_shape, dict):
                self.running_mean_std = RunningMeanStdObs(input_shape)
            else:
                self.running_mean_std = RunningMeanStd(input_shape)

        for m in self.modules():         
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value

    def forward(self, input_dict):
        is_train = input_dict.get('is_train', True)
        prev_actions = input_dict['prev_actions']
        obs = self.norm_obs(input_dict['obs'])

        out = obs
        out = out = out.flatten(1)                
        out = self.actor_mlp(out)
        c_out = a_out = out
        value = self.value(c_out)

        mu = self.mu(a_out)
        logstd = mu*0 + self.sigma

        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        if is_train:
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogp = _neglogp(prev_actions, mu, sigma, logstd)
            result = {
                'prev_neglogp' : torch.squeeze(prev_neglogp),
                'values' : value,
                'entropy' : entropy,
                'mus' : mu,
                'sigmas' : sigma
            }                
            return result
        else:
            selected_action = distr.sample()
            neglogp = _neglogp(selected_action, mu, sigma, logstd)
            result = {
                'neglogpacs' : torch.squeeze(neglogp),
                'values' : self.denorm_value(value),
                'actions' : selected_action,
                'mus' : mu,
                'sigmas' : sigma
            }
            return result