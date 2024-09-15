import torch 
import numpy as np
import gym
from collections import defaultdict
from torch.utils.data import Dataset
    

class AWRDataset(Dataset):

    def __init__(self, batch_size, minibatch_size, device):

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.idx = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

    def shuffle(self):
        self.idx = torch.randperm(self.batch_size)

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict     

    def update_mu_sigma(self, mu, sigma):	    
        start = self.last_range[0]	           
        end = self.last_range[1]	
        self.values_dict['mu'][start:end] = mu	
        self.values_dict['sigma'][start:end] = sigma 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k,v in self.values_dict.items():
            if v is not None:
                if type(v) is dict:
                    v_dict = { kd:vd[self.idx[start:end]] for kd, vd in v.items() }
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[self.idx[start:end]]
                
        return input_dict

class ExperienceBuffer:
    '''
    More generalized than replay buffers.
    Implemented for on-policy algos
    '''
    def __init__(self, observation_space, action_space, horizon_length, num_actors, device):
        self.device = device
        self.num_agents = num_actors
        self.action_space = action_space
        self.horizon_length = horizon_length
        self.observation_space = observation_space
        obs_base_shape = (self.horizon_length, self.num_agents)
        assert type(self.action_space) is gym.spaces.Box
        self.actions_shape = (self.action_space.shape[0],) 
        self.actions_num = self.action_space.shape[0]
        self.tensor_dict = {}

        self.tensor_dict['obses'] = self._create_tensor_from_space(observation_space, obs_base_shape)
        
        val_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.tensor_dict['rewards'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['values'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['neglogpacs'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.float32), obs_base_shape)
        self.tensor_dict['dones'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.uint8), obs_base_shape)
        self.tensor_dict['actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)
        self.tensor_dict['mus'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)
        self.tensor_dict['sigmas'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)

    def _create_tensor_from_space(self, space, base_shape):    
        from rl_games.algos_torch.torch_ext import numpy_to_torch_dtype_dict
        dtype = numpy_to_torch_dtype_dict[space.dtype]
        return torch.zeros(base_shape + space.shape, dtype= dtype, device = self.device)

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k,v in val.items():
                self.tensor_dict[name][k][index,:] = v
        else:
            self.tensor_dict[name][index,:] = val

    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)
        
        return res_dict
    
    def rotate(self): ...
    
class LongExperienceBuffer():
    def __init__(self, observation_space, action_space, horizon_length, num_actors, num_buffers, device):
        self.device = device
        self.num_agents = num_actors
        self.action_space = action_space
        self.horizon_length = horizon_length
        self.observation_space = observation_space
        self.num_buffers = num_buffers
        self.buffers = [ExperienceBuffer(observation_space, action_space, horizon_length, num_actors, device) for _ in range(num_buffers)]
        self.names = self.buffers[0].tensor_dict.keys()
        self.tensor_dict = {}
        self._update_full_tensor_dict()
    
    def _update_tensor_dict(self, name):
        self.tensor_dict[name] = torch.cat([buffer.tensor_dict[name] for buffer in self.buffers], dim=0)

    def _update_full_tensor_dict(self):
        for name in self.names:
            self._update_tensor_dict(name)

    def rotate(self):
        self.buffers = self.buffers[1:] + [self.buffers[0]]
        self._update_full_tensor_dict()

    def update_data(self, name, index, val):
        self.buffers[-1].update_data(name, index, val)
        self._update_tensor_dict(name)
        
    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = defaultdict(list)
        transforms = [buffer.get_transformed_list(transform_op, tensor_list) for buffer in self.buffers]
        for transform in transforms:
            for k in tensor_list:
                v = transform.get(k)
                if v is not None:
                    res_dict[k].append(v)
        for k, v in res_dict.items():
            if type(v) is dict:
                res_dict[k] = {kd: torch.cat(vd) for kd, vd in v.items()}
            else:
                res_dict[k] = torch.cat(v)
        return res_dict

class Diagnostics():
    def __init__(self):
        self.diag_dict = {}
        self.current_epoch = 0
        self.dict = defaultdict(list)

    def send_info(self, writter):
        if writter is None:
            return
        for k,v in self.diag_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            writter.add_scalar(k, v, self.current_epoch)

    def add(self, key, value):
        self.diag_dict[key] = value
    
    def epoch(self, agent, current_epoch):
        self.current_epoch = current_epoch
        for k, v in self.dict.items():
            self.diag_dict['diagnostics/{0}'.format(k)] = torch.stack(v, axis=0).mean()
        self.dict = defaultdict(list)

    def mini_epoch(self, agent, miniepoch):
        pass

    def mini_batch(self, agent, batch):
        with torch.no_grad():
            for k, v in batch.items():
                self.dict[k].append(v)