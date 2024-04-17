import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from isaacgymenvs.ppo.network_builder import NetworkBuilder

class GOIDGoalSampler(nn.Module):
    builder = NetworkBuilder.BaseNetwork()
    
    def __init__(self, env, obs_dim, cfg, device):
        super().__init__()
        
        self.env = env
        self.obs_dim = np.prod(obs_dim)
        self.device = device
        
        self.intermediate_lower = cfg['intermediate'].get('lower', 0.1)
        self.intermediate_upper = cfg['intermediate'].get('upper', 0.9)
        self.max_sample_attempts = cfg['intermediate'].get('max_sample_attempts', 10)
        
        self.units = cfg['mlp']['units']
        self.activation = cfg['mlp']['activation']
        self.normalization = cfg.get('normalization', None)
        self.initializer = cfg['mlp']['initializer']
        self.is_d2rl = cfg['mlp'].get('d2rl', False)
        self.norm_only_first_layer = cfg['mlp'].get('norm_only_first_layer', False)
        mlp_args = {
            'input_size': self.obs_dim,
            'units': self.units,
            'activation': self.activation,
            'norm_func_name': self.normalization,
            'dense_func' : nn.Linear,
            'd2rl': self.is_d2rl,
            'norm_only_first_layer': self.norm_only_first_layer,
        }
        self.model = self.builder._build_mlp(**mlp_args)
        self.model = nn.Sequential(
            *list(self.model.children()), 
            nn.Linear(mlp_args['units'][-1], 1), 
            nn.Sigmoid()
        )
        mlp_init = self.builder.init_factory.create(**self.initializer)
        for m in self.modules():         
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        self.lr = cfg['config']['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.loss_fn = nn.BCELoss()
        
    def forward(self, x):
        return self.model(x.view(-1, self.obs_dim))
    
    def train(self, obs_batch, success_batch):
        assert len(obs_batch) == len(success_batch)
        preds = self(obs_batch)
        loss = self.loss_fn(preds, success_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def rejection_sampling(self):
        self.model.eval()
        with torch.no_grad():
            sampled_states, sampled_obses = [], []
            num_envs = self.env.num_envs
            indices = torch.full((num_envs,), self.max_sample_attempts, device=self.device)
            num_samples = 0
            while num_samples < self.max_sample_attempts:
                states, obses = self.env.sample_goals(1)
                state, obs = states[0], obses[0]
                sampled_states.append(state)
                sampled_obses.append(obs)
                pred = self(obs).flatten()
                is_intermediate = torch.logical_and(self.intermediate_lower < pred, pred < self.intermediate_upper)
                indices = torch.min(is_intermediate * num_samples, indices)
                num_samples += 1
                if torch.all(indices < self.max_sample_attempts):
                    break
            random_idx = torch.IntTensor(np.random.randint(num_samples, size=num_envs)).to(self.device)
            filled_indices = torch.where(indices < self.max_sample_attempts, indices, random_idx)
            
            selected_states = {}
            for k in sampled_states[0].keys():
                sampled_state_k = torch.stack([sampled_state[k] for sampled_state in sampled_states], dim=0)
                selected_states[k] = sampled_state_k[filled_indices, torch.arange(num_envs)]
            sampled_obses_cat = torch.stack(sampled_obses, dim=0)
            selected_obs = sampled_obses_cat[filled_indices, torch.arange(num_envs)]
        
        # TODO: fix logging; currently takes stats over envs
        return {
            'states': selected_states,
            'obs': selected_obs,
            'stats': {
                'goid_num_samples_mean': indices[indices < self.max_sample_attempts].float().mean() + 1,
                'goid_num_samples_std': indices[indices < self.max_sample_attempts].float().std(),
                'goid_random_samples_frac': torch.sum(indices >= self.max_sample_attempts) / np.prod(indices.shape),
                'goid_mean': torch.mean(pred),
                'goid_std': torch.std(pred)
            }
        }
            