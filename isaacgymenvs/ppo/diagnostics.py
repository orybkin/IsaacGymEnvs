import torch
from collections import defaultdict

class DefaultDiagnostics(object):
    def __init__(self):
        pass
    def send_info(self, writter):
        pass    
    def epoch(self, agent, current_epoch):
        pass
    def mini_epoch(self, agent, miniepoch):
        pass
    def mini_batch(self, agent, batch, e_clip, minibatch):
        pass


class PpoDiagnostics(DefaultDiagnostics):
    def __init__(self):
        self.diag_dict = {}
        self.clip_fracs = []
        self.exp_vars = []
        self.current_epoch = 0
        self.dict = defaultdict(list)

    def send_info(self, writter):
        if writter is None:
            return
        for k,v in self.diag_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            writter.add_scalar(k, v, self.current_epoch)
    
    def epoch(self, agent, current_epoch):
        self.current_epoch = current_epoch
        for k, v in self.dict.items():
            self.diag_dict['diagnostics/{0}'.format(k)] = torch.stack(v, axis=0).mean()
        self.dict = defaultdict(list)

    def mini_batch(self, agent, batch):
        with torch.no_grad():
            for k, v in batch.items():
                self.dict[k].append(v)
            

            

