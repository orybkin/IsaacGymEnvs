import ml_collections
from ml_collections import config_flags
from absl import app, flags
FLAGS = flags.FLAGS

fetch_push_config = ml_collections.ConfigDict({
    'task_name': 'FetchPush-v2',
    'name': 'MujocoGoal',
    'env': {
        'numEnvs': 0,
    }
})

def get_config():
    return fetch_push_config