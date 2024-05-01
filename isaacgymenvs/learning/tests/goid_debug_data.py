import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import re

def preprocess_data(path):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    obs = success = None
    success_rates = []
    for filename in sorted(os.listdir(path), key=alphanum_key):
        if osp.splitext(filename)[1] != '.npy': continue
        is_obs = filename.startswith('obs')
        data = np.load(osp.join(path, filename))
        if is_obs:
            obs = data if obs is None else np.concatenate([obs, data])
        else:
            success = data if success is None else np.concatenate([success, data])
            success_rates.append(data.mean())
    return success, success_rates


success, success_rates = preprocess_data('data/goid')
print(success.shape, success.shape[0]/8192, success.mean())
plt.plot(np.arange(len(success_rates)), success_rates)
plt.savefig('goid_success.png')