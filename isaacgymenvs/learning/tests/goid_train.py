import os
import os.path as osp
import argparse
import numpy as np
import re
import torch
import wandb
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from isaacgymenvs.learning.curriculum import GOIDGoalSampler

def preprocess_data(path, batch_size):
    obs = success = None
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    for filename in sorted(os.listdir(path), key=alphanum_key):
        if osp.splitext(filename)[1] != '.npy': continue
        is_obs = filename.startswith('obs')
        data = np.load(osp.join(path, filename))
        if is_obs:
            obs = data if obs is None else np.concatenate([obs, data])
        else:
            success = data if success is None else np.concatenate([success, data])
    
    obs = torch.from_numpy(obs).float()
    success = torch.from_numpy(success).float()
    dataset = TensorDataset(obs, success)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def train_epochs(wandb_run, data_loader, n_epochs, cfg, device, print_every=100):
    goal_sampler = GOIDGoalSampler(None, (22,), cfg, device).to(device)

    for epoch in tqdm(range(1, n_epochs + 1), desc='Epoch'):
        cnt = 1
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            breakpoint()
            batch_dict = goal_sampler.train(data, labels)
            wandb_run.log({'epoch': epoch, **batch_dict})
            if cnt % print_every == 0:
                print(f'Batch {cnt}/{len(data_loader)}: ', batch_dict)
            cnt += 1
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='goid_classifier')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_project', type=str, default='taskmaster')
    parser.add_argument('--wandb_entity', type=str, default='prestonfu')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--gradient_steps', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    args = parser.parse_args()
    
    cfg = {
        'intermediate': {
            'lower': 0.1,
            'upper': 0.9,
            'max_sample_attempts': 10,
        },
        'mlp': {
            'units': [256, 128, 64],
            'activation': 'elu',
            'd2rl': False,
            'initializer': {'name': 'default'},
            'regularizer': {'name': 'None'}
        },
        'config': {
            'batch_size': args.batch_size,
            'gradient_steps': args.gradient_steps,
            'learning_rate': args.learning_rate,
            'success_metric': 'success_4',
            'collect_data': False
        }
    }
    
    try:
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S-%f")
        data_loader = preprocess_data('data/goid', args.batch_size)
        experiment_name = f'{args.experiment_name}_bs{args.batch_size}_steps{args.gradient_steps}_lr{args.learning_rate}'
        wandb_dir = os.path.join('runs', experiment_name)
        os.makedirs(wandb_dir, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group='',
            tags=['goid_classifier'],
            dir=wandb_dir,
            sync_tensorboard=True,
            name=f'{timestamp}_{experiment_name}',
            settings=wandb.Settings(start_method='fork'),
        )
        train_epochs(run, data_loader, args.n_epochs, cfg, device=args.device)
    except KeyboardInterrupt:
        pass
    wandb.finish()