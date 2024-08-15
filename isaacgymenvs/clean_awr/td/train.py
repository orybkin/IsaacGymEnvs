"""
Train a TemporalDistanceNetwork using data loaded from a previous run.

 - Copies how training works in the real algorithm, i.e. one buffer at a time
   and minibatches within buffers.
 - Loads the env with Isaacgym. This is just to specify an env via the command 
   line and extract some data from it, not interact with it.
"""

import isaacgym
import isaacgymenvs

import os
import numpy as np
import torch

from pathlib import Path
from torch import optim
from torch.utils.data import Dataset
from ml_collections import config_flags
from absl import app, flags

from rl_games.common import env_configurations, vecenv

from isaacgymenvs.clean_awr.temporal_distance import TemporalDistanceNetwork
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, Every


class SimpleTemporalDistanceDataset(Dataset):
    def __init__(self, config, all_data, all_epochs):
        self.all_data = all_data
        self.all_epochs = all_epochs
        self.batch_size = all_data[0]['goal'].shape[0]
        self.minibatch_size = config['horizon_length'] * config['num_actors'] // config['num_minibatches']
        self.length = len(all_data)
        self.batch_length = self.batch_size // self.minibatch_size
        self.device = config['device']
        self.last_batch = None
        self.idx = None
        
    def _regenerate_idx(self):
        self.idx = torch.randperm(self.batch_size, dtype=torch.long, device=self.device)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, key):
        if isinstance(key, int):
            self.last_batch = key
            self._regenerate_idx()
            return {k: v[self.idx] for k, v in self.all_data[key].items()}
        
        elif isinstance(key, tuple):
            assert len(key) == 2
            batch_num, minibatch_num = key
            if batch_num != self.last_batch:
                self.last_batch = batch_num
                self._regenerate_idx()
        
            start = minibatch_num * self.minibatch_size
            end = (minibatch_num + 1) * self.minibatch_size
            self.last_range = (start, end)
            input_dict = {}
            for k, v in self.all_data[batch_num].items():
                if v is not None:
                    if type(v) is dict:
                        v_dict = {kd: vd[self.idx[start:end]] for kd, vd in v.items()}
                        input_dict[k] = v_dict
                    else:
                        input_dict[k] = v[self.idx[start:end]]
            return input_dict
        
        else:
            raise ValueError()
            

class TemporalDistanceTrainer:
    def __init__(self, config, load_path, load_every=1):
        self.config = config
        self.vec_env = vecenv.create_vec_env('rlgpu', config['num_actors'])
        self.device = config['device']
        if config['temporal_distance']['regression']:
            self.model = TemporalDistanceNetwork(
                config['hidden_dims'], 
                self.vec_env.env.achieved_idx, 
                output_size=2,
                regression_coef=config['temporal_distance']['regression_coef']
            ).to(self.device)
        else:
            self.model = TemporalDistanceNetwork(
                config['hidden_dims'], 
                self.vec_env.env.achieved_idx, 
                output_size=config['relabel_every'] + 1,
                classifier_selection=config['temporal_distance']['classifier_selection']
            ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.config['lr'], eps=1e-08, weight_decay=0)
        self._load_data(load_path, load_every)
        
    def _load_data(self, load_path, load_every):
        all_data_dict = {}
        n_total_data = 0
        for filename in os.listdir(load_path):
            base, ext = os.path.splitext(filename)
            assert ext == '.npy'
            if int(base) % load_every == 0:
                data = np.load(Path(load_path) / filename, allow_pickle=True).item()
                data = {k: torch.tensor(v) for k, v in data.items()}
                data['distance'] = data['distance'].long()
                all_data_dict[int(base)] = data
                n_total_data += len(data['distance'])
        print('Number of samples:', n_total_data)
        
        all_epochs = sorted(all_data_dict.keys())
        all_data = [all_data_dict[e] for e in all_epochs]
        self.dataset = SimpleTemporalDistanceDataset(self.config, all_data, all_epochs)

    def train(self):
        for i in range(len(self.dataset)):
            for mini_ep in range(0, self.config['temporal_distance']['mini_epochs']):
                for j in range(self.dataset.batch_length):
                    self.optimizer.zero_grad()
                    losses = self.model.loss({k: v.to(self.device) for k, v in self.dataset[i, j].items()})
                    losses['loss'].backward()
                    self.optimizer.step()
            print(f'Epoch {self.dataset.all_epochs[i]}:',
                  'loss =', round(losses['loss'].item(), 4), 
                  'accuracy =', round(losses['accuracy'].item(), 4))

        
def main(_):
    FLAGS = flags.FLAGS
    FLAGS.agent['num_actors'] = FLAGS.agent['num_envs']
    FLAGS.env['env']['numEnvs'] = FLAGS.agent['num_envs']

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            -1, # seed 
            FLAGS.env['name'], 
            FLAGS.agent['num_envs'], 
            FLAGS.agent['device'],
            FLAGS.agent['device'],
            FLAGS.agent['graphics_device_id'],
            True, # headless
            False, # multi_gpu
            False, # capture_video
            True, # force_render
            FLAGS.env,
            **kwargs,
        )
        return envs
    
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })
    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        
    trainer = TemporalDistanceTrainer(
        FLAGS.agent,
        load_path='data/temporal_distance/240814-173913-470136_flc2_awr'
    )
    trainer.train()
    
    
if __name__ == '__main__':
    config_flags.DEFINE_config_file('agent', 'clean_awr/agent_config/ig_push.py', 'Agent configuration.', lock_config=False)
    config_flags.DEFINE_config_file('env', 'clean_awr/env_config/ig_push.py', 'Env configuration.', lock_config=False)
    app.run(main)