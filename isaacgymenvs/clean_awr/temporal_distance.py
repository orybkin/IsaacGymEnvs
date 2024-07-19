# A simple MLP predicting temporal distances

import torch
import torch.nn as nn
import torch.nn.functional as F
from isaacgymenvs.clean_awr.awr_utils import _build_sequential_mlp


class TemporalDistanceNetwork(nn.Module):
    def __init__(self, input_shape, units, goal_idx):
        nn.Module.__init__(self)
        self.units = units
        self.mlp = _build_sequential_mlp(input_shape[0], units)
        self.goal_idx = goal_idx
        nn.init.constant_(self.sigma, 0)

        for m in self.modules():         
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, input_dict):
        out = input_dict['obs'].flatten(1)      
        out = self.mlp(out)

        return out
        
    def loss(self, input_dict):
        # TODO get goal observations
        # TODO sample pairs
        # TODO compute loss
        # TODO save metrics
        # TODO add this to training loop
        # TODO add negative pairs

        obs = input_dict['obs'][:, self.goal_idx]
        # TODO you need to sample pairs elsewhere and compute the corresponding timings. only loss is here. 
        # there has to be a separate training loop for this.
