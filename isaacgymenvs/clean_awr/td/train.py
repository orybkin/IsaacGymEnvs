"""
Train a TemporalDistanceNetwork using data loaded from a previous run.

 - Copies how training works in the real algorithm, i.e. one buffer at a time
   and minibatches within buffers.
 - Loads the env with Isaacgym. This is just to specify an env via the command 
   line and extract some data from it, not interact with it.
"""

import isaacgym

import os
import numpy as np
import torch

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv

import wandb
from datetime import datetime
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset
from ml_collections import config_flags
from absl import app, flags

from collections import defaultdict
from tensorboardX import SummaryWriter

import isaacgymenvs
from isaacgymenvs.clean_awr.temporal_distance import TemporalDistanceNetwork
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv


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
        
        self.train_dir = 'runs'
        self.experiment_dir = os.path.join(self.train_dir, self.config['run_name'])
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')
        os.makedirs(self.summaries_dir, exist_ok=True)
        self.writer = SummaryWriter(self.summaries_dir)
        
        if not self.config['temporal_distance']['full_state']:
            self.td_idx = self.vec_env.env.achieved_idx  
        else:
            self.td_idx = np.arange(self.vec_env.env.cfg['env']['numObservations'])
        if not self.config['temporal_distance']['regression']:
            td_output_size = max(self.config['relabel_every'], self.config['horizon_length']) + 1
        else:
            td_output_size = 1
        kw = dict(
            units=self.config['hidden_dims'], 
            input_size=len(self.td_idx) + len(self.vec_env.env.achieved_idx),
            output_size=td_output_size
        )
        if not self.config['temporal_distance']['regression']:
            kw['classifier_selection'] = self.config['temporal_distance']['classifier_selection']
        self.model = TemporalDistanceNetwork(**kw).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.config['temporal_distance']['lr'], eps=1e-08, weight_decay=0)
        self._load_data(load_path, load_every)
        
    def _load_data(self, load_path, load_every):
        all_data_dict = {}
        n_total_data = 0
        print('negative examples labeled as:', self.config['relabel_every'])
        for filename in os.listdir(load_path):
            base, ext = os.path.splitext(filename)
            assert ext == '.npy'
            if int(base) % load_every == 0:
                data = np.load(Path(load_path) / filename, allow_pickle=True).item()
                data = {k: torch.tensor(v) for k, v in data.items()}
                data['distance'] = torch.where(data['distance'] >= self.config['horizon_length'], self.config['relabel_every'], data['distance'])
                all_data_dict[int(base)] = data
                n_total_data += len(data['distance'])
        print('number of samples:', n_total_data)
        all_epochs = sorted(all_data_dict.keys())
        all_data = [all_data_dict[e] for e in all_epochs]
        self.dataset = SimpleTemporalDistanceDataset(self.config, all_data, all_epochs)
        
    def write_stats(self, metrics, epoch_num):
        for k, v in metrics.items():
            if len(v) > 0:
                v = torch_ext.mean_list(v).item()
                self.writer.add_scalar(k, v, epoch_num)

    def train(self):
        for epoch_num in range(1, len(self.dataset) + 1):
            metrics = defaultdict(list)
            for mini_ep in range(self.config['temporal_distance']['mini_epochs']):
                for mb_num in range(self.dataset.batch_length):
                    self.optimizer.zero_grad()
                    losses = self.model.loss({k: v.to(self.device) for k, v in self.dataset[epoch_num - 1, mb_num].items()})
                    if not self.config['temporal_distance']['regression']:
                        losses['ce'].backward()
                    else:
                        losses['mse'].backward()
                    self.optimizer.step()
                    for k, v in losses.items():
                        if mini_ep == 0:
                            metrics[f'temporal_distance/val_{k}'].append(v)
                        if mini_ep == self.config['temporal_distance']['mini_epochs'] - 1:
                            metrics[f'temporal_distance/{k}'].append(v)
            self.write_stats(metrics, epoch_num)
            train_loss = torch_ext.mean_list(metrics['temporal_distance/ce']).item()
            val_loss = torch_ext.mean_list(metrics['temporal_distance/val_ce']).item()
            print(f'epoch: {epoch_num}  train loss: {round(train_loss, 4)}  val loss: {round(val_loss, 4)}')

        
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
    
    time_str = datetime.now().strftime("%y%m%d-%H%M%S-%f")
    if FLAGS.slurm_job_id != -1:
        run_name = f"{FLAGS.slurm_job_id}_{FLAGS.agent['experiment']}"
    else:
        run_name = f"{time_str}_{FLAGS.agent['experiment']}"
    FLAGS.agent['run_name'] = run_name

    env = dict(FLAGS.env)
    env['env'] = dict(env['env'])
    if 'sim' in env:
        env['sim'] = dict(env['sim'])
    
    wandb.init(
        project='taskmaster',
        entity=FLAGS.agent['wandb_entity'],
        group='Default',
        sync_tensorboard=True,
        id=run_name,
        name=run_name,
        config={'agent': dict(FLAGS.agent), 'env': env, 'experiment': FLAGS.agent['experiment']},
        # settings=wandb.Settings(start_method='fork'),
    )
        
    trainer = TemporalDistanceTrainer(
        FLAGS.agent,
        load_path='data/temporal_distance/240816-170206-826469_flc2_awr'
        # load_path='data/temporal_distance/240817-161724-059712_flc2_awr_td'
    )
    trainer.train()
    
    
if __name__ == '__main__':
    flags.DEFINE_integer('slurm_job_id', -1, '')
    config_flags.DEFINE_config_file('agent', 'clean_awr/agent_config/ig_push.py', 'Agent configuration.', lock_config=False)
    config_flags.DEFINE_config_file('env', 'clean_awr/env_config/ig_push.py', 'Env configuration.', lock_config=False)
    app.run(main)
        