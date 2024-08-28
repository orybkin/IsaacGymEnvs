import ml_collections
from ml_collections import config_flags

ig_reaching2_config = ml_collections.ConfigDict({
    'experiment': '',
    'wandb_entity': 'oleh-rybkin',
    'num_envs': 8192,
    'clip_observations': 5.0,
    'clip_actions': 1,
    'hidden_dims': (256, 128, 64),
    'separate': False,
    'temperature': 0.2, # 0 for behavior cloning.
    'horizon_length': 32,
    'num_minibatches': 16,
    'mini_epochs': 5,
    'gamma': 0.99,
    'tau': 0.95,
    'lr': 0.0005,
    'entropy_coef': 0.01,
    'normalize_value': 1,
    'normalize_input': 1,
    'normalize_advantage': 1,
    'normalize_advantage_joint': 0,
    'value_bootstrap': 1,
    'joint_value_norm': 1,
    'relabel': 1,
    'device': 'cuda:0',
    'graphics_device_id': 0,
    'test_every_episodes': 10,
    'max_epochs': 50000,
    'e_clip': 0.2,
    'clip_value': 1,
    'critic_coef': 4.,
    'bounds_loss_coef': 0.0001,
    'truncate_grads': 0,
    'grad_norm': 5,
    'action_bound': 1.1,
    'norm_by_return': 0,
    'relabeled_critic_coef': 1.0,
    'relabeled_actor_coef': 1.0,
    'relabel_every': 0,
    'relabel_strategy': 'final',
    'goal_interpolation': 1.,
    'onpolicy_coef': 1.,
    'temporal_distance': {
        'lr': 1e-3,
        'mini_epochs': 5,
        'full_state': False,
        'neg_goal_selection': 'achieved',
        'negative_pairs_frac': 1.,
        'classifier_selection': 'mode',
        'regression': False,
        'last_logit_rew': 0,
        'plot_every': 50,
        'save_data': False,
        'data_overwrite_lines': False
    }
})

def get_config():
    return ig_reaching2_config