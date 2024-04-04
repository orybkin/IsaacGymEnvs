import numpy as np
import torch
from scipy.stats import entropy

class VDSGoalSampler:
    FUN_NAME_TO_FUN = {
        'var': lambda vals: torch.var(vals, dim=0),
        'std': lambda vals: torch.std(vals, dim=0),
        'tanh': lambda vals: torch.tanh(torch.var(vals, dim=0)),
        'exp': lambda vals: torch.exp(torch.std(vals, dim=0)),
    }

    def __init__(self, env, algo_name, device):
        self.env = env
        self.algo_name = algo_name
        self.device = device
    
    def sample_disagreement(self, n_candidates, model_runner, disagreement_fn_name='std'):
        """
        Args:
          n_candidates: number of goal candidates to sample
          model_runner: accepts obs dict
          disagreement_fn_name: as in VDSGoalSampler.FUN_NAME_TO_FUN
          logger
        """
        disagreement_fn = VDSGoalSampler.FUN_NAME_TO_FUN[disagreement_fn_name]
        cand_states, cand_obses = self.env.sample_goals(n_candidates)
        with torch.no_grad():
            values = []
            if self.algo_name == 'ppo':
                for cand_obs in cand_obses:
                    res_dict = model_runner({'obs': cand_obs})
                    values.append(res_dict['full_values'].unsqueeze(0))
                values = torch.cat(values, dim=0)  # (n_candidates, num_envs, num_critics)
                values = values.permute(2, 1, 0)
            else:
                raise NotImplementedError
        disagreement = disagreement_fn(values).detach().cpu().numpy()  # (num_envs, n_candidates)
        sum_disagreement = np.sum(disagreement, axis=1, keepdims=True)
        if np.allclose(sum_disagreement, 0):
            disagreement = None
        else:
            disagreement /= sum_disagreement
        indices = np.apply_along_axis(lambda row: np.random.choice(len(row), p=row), axis=1, arr=disagreement)
        
        cand_obses = torch.cat([x.unsqueeze(0) for x in cand_obses], dim=0)
        num_envs = cand_obses.shape[1]
        sampled_states = {}
        for k in cand_states[0].keys():
            cand_states_k = torch.cat([cand_states[i][k].unsqueeze(0) for i in range(n_candidates)], dim=0)  # (n_candidates, num_envs, ...)
            sampled_states[k] = cand_states_k[indices, torch.arange(num_envs), ...]
        sampled_obs = cand_obses[indices, torch.arange(num_envs), :]

        # FIXME: disagreement logging
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
    