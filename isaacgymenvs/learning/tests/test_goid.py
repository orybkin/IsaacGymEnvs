import os
import os.path as osp
import argparse
import numpy as np
import torch
import wandb
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from isaacgymenvs.learning.curriculum import GOIDGoalSampler

def preprocess_data(path, batch_size):
    obs = success = None
    for filename in os.listdir(path):
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
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_epochs(wandb_run, train_loader, val_loader, n_epochs, cfg, device):
    goal_sampler = GOIDGoalSampler(None, (22,), cfg, device).to(device)

    for epoch in tqdm(range(1, n_epochs + 1), desc='Epoch'):
        epoch_train_dict = {}
        for data, labels in tqdm(train_loader):
            data, labels = data.to(device), labels.to(device)
            batch_dict = goal_sampler.train(data, labels)
            for key in ['goid_train_accuracy', 'goid_train_loss']:
                value = batch_dict[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if key in epoch_train_dict:
                    epoch_train_dict[key].append(value)
                else:
                    epoch_train_dict[key] = [value]
        
        epoch_val_dict = {}
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            batch_dict = goal_sampler.eval(data, labels)
            for key in batch_dict:
                value = batch_dict[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if key in epoch_val_dict:
                    epoch_val_dict[key].append(value)
                else:
                    epoch_val_dict[key] = [value]
        
        epoch_dict = {**epoch_train_dict, **epoch_val_dict}
        epoch_dict = {k: np.mean(v) for k, v in epoch_dict.items()}
        print(epoch_dict)
        wandb_run.log({'epoch': epoch, **epoch_dict})
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='goid_classifier')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_project', type=str, default='taskmaster')
    parser.add_argument('--wandb_entity', type=str, default='prestonfu')
    parser.add_argument('--n_epochs', type=int, default=50)
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
            'gradient_steps': args.gradient_steps,
            'learning_rate': args.learning_rate,
            'success_metric': 'success_4'
        }
    }
    
    try:
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S-%f")
        train_loader, val_loader = preprocess_data('data/goid', args.batch_size)
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
        train_epochs(run, train_loader, val_loader, args.n_epochs, cfg, device=args.device)
    except KeyboardInterrupt:
        pass
    wandb.finish()