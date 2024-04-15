import numpy as np
import torch
from scipy.stats import entropy

class VDSGoalSampler:
    def __init__(self, env, cfg, model_runner, algo_name, device):
        self.env = env
        self.model_runner = model_runner
        self.algo_name = algo_name
        self.device = device
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
    
    def sample_disagreement(self):
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
        disagreement = np.exp(np.log(disagreement) * self.temperature)
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

        # FIXME: disagreement logging
        if disagreement is None:
            d_mean = d_std = d_entropy = d_max_entropy = -1
        else:
            disagreement = np.mean(disagreement, axis=0)  # mean over envs
            d_mean = np.mean(disagreement)
            d_std = np.std(disagreement)
            _, d_counts = np.unique(disagreement, return_counts=True)
            d_entropy = entropy(d_counts, base=2)
            d_max_entropy = np.log2(len(disagreement))
        
        return {
            'states': sampled_states,
            'obs': sampled_obs,
            'stats': {
                'd_mean': d_mean,
                'd_std': d_std,
                'd_entropy': d_entropy,
                'd_max_entropy': d_max_entropy,
            }
        }
    