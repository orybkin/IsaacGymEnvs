import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import entropy

from isaacgymenvs.ppo.network_builder import NetworkBuilder

class GoalSampler:
    def __init__(self, env, requires_extra_sim=0):
        self.env = env
        self.requires_extra_sim = requires_extra_sim
        
    def sample(self):
        raise NotImplementedError

class UniformGoalSampler(GoalSampler):
    def __init__(self, env):
        super().__init__(env)
        
    def sample(self):
        states, obs = self.env.sample_goals(1)
        return {'states': states[0], 'obs': obs[0]}

class GOIDGoalSampler(nn.Module, GoalSampler):
    builder = NetworkBuilder.BaseNetwork()
    
    def __init__(self, env, obs_dim, cfg, device):
        nn.Module.__init__(self)
        GoalSampler.__init__(self, env, requires_extra_sim=10)
        
        self.obs_dim = np.prod(obs_dim)
        self.device = device
        
        self.intermediate_lower = cfg['intermediate'].get('lower', 0.1)
        self.intermediate_upper = cfg['intermediate'].get('upper', 0.9)
        self.max_sample_attempts = cfg['intermediate'].get('max_sample_attempts', 10)
        self.success_metric = cfg['config']['success_metric']
        
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
        self.batch_size = cfg['config']['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.loss_fn = nn.BCELoss()
        
    def forward(self, x):
        return self.model(x.view(-1, self.obs_dim))
    
    def train(self, obs_batch, success_batch):
        self.model.train()
        assert len(obs_batch) == len(success_batch)
        preds = self(obs_batch)
        loss = torch.autograd.Variable(self.loss_fn(preds, success_batch), requires_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def sample(self):
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
        
class VDSGoalSampler(GoalSampler):
    def __init__(self, env, cfg, model_runner, algo_name):
        super().__init__(env, requires_extra_sim=10)
        self.model_runner = model_runner
        self.algo_name = algo_name
        self.temperature = cfg.get('temperature', 1)
        self.n_candidates = cfg.get('n_candidates', 1)
        fn_name_to_fn = {
            'var': lambda vals: torch.var(vals, correction=0, dim=0),
            'std': lambda vals: torch.std(vals, correction=0, dim=0),
            'tanh': lambda vals: torch.tanh(torch.var(vals, correction=0, dim=0)),
            'exp': lambda vals: torch.exp(torch.std(vals, correction=0, dim=0)),
        }
        disagreement_fn_name = cfg.get('disagreement_fn_name', 'std')
        self.disagreement_fn = fn_name_to_fn[disagreement_fn_name]
    
    def sample(self):
        cand_states, cand_obses = self.env.sample_goals(self.n_candidates)
        cand_obses = torch.stack(cand_obses, dim=0)
        num_envs = cand_obses.shape[1]
        with torch.no_grad():
            values = []
            if self.algo_name == 'ppo':
                for cand_obs in cand_obses:
                    res_dict = self.model_runner({'obs': cand_obs})
                    values.append(res_dict['full_values'])
                values = torch.stack(values, dim=0)  # (n_candidates, num_envs, num_critics)
                values = values.permute(2, 1, 0)
            else:
                raise NotImplementedError
        disagreement = self.disagreement_fn(values).detach().cpu().numpy()  # (num_envs, n_candidates)
        disagreement = disagreement ** self.temperature
        sum_disagreement = np.sum(disagreement, axis=1, keepdims=True)
        if np.allclose(sum_disagreement, 0):
            disagreement = None
            indices = np.zeros(num_envs)
        else:
            disagreement /= sum_disagreement
            indices = np.apply_along_axis(lambda row: np.random.choice(len(row), p=row), axis=1, arr=disagreement)
    
        sampled_states = {}
        for k in cand_states[0].keys():
            cand_states_k = torch.stack([cand_states[i][k] for i in range(self.n_candidates)], dim=0)  # (n_candidates, num_envs, ...)
            sampled_states[k] = cand_states_k[indices, torch.arange(num_envs), ...]
        sampled_obs = cand_obses[indices, torch.arange(num_envs), :]

        # TODO: fix logging; currently takes a mean over envs
        if disagreement is None:
            d_mean = d_std = d_entropy = d_max_entropy = -1
        else:
            disagreement = np.mean(disagreement, axis=0)
            d_mean = np.mean(disagreement)
            d_std = np.std(disagreement)
            _, d_counts = np.unique(disagreement, return_counts=True)
            d_entropy = entropy(d_counts, base=2)
            d_max_entropy = np.log2(len(disagreement))
        
        return {
            'states': sampled_states,
            'obs': sampled_obs,
            'stats': {
                'disagreement_mean': d_mean,
                'disagreement_std': d_std,
                'disagreement_entropy': d_entropy,
                'disagreement_max_entropy': d_max_entropy,
            }
        }
    