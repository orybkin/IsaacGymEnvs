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


ig_push_config = ml_collections.ConfigDict({
    'name': 'FrankaPushing',
    "physics_engine": "physx",
    "env": {
        'numEnvs': 0,
        "envSpacing": 1.5,
        "episodeLength": 256,
        "enableDebugVis": False,
        "clipObservations": 5.0,
        "clipActions": 1.0,
        "startRotationNoise": 0.785,
        "frankaPositionNoise": 0.0,
        "frankaRotationNoise": 0.0,
        "frankaDofNoise": 0.25,
        "distRewardScale": 1,
        "distRewardDropoff": 30,
        "distRewardThreshold": 0.02,
        "aggregateMode": 3,
        "actionScale": 0.5,
        "mode": "",
        "distanceFromBlock": 0.0,
        "nCubes": 6,
        "startPositionNoise": 0.12,
        "goalPositionNoise": 0.12,
        "friction": 0.5,
        "rigidCubes": False,
        "testTask": -1,
        "renderEveryEpisodes": 1000,
        "controlType": "osc",  # options are {joint_tor, osc}
        "observeVelocities": False,
        "asset": {
            "assetRoot": "../../assets",
            "assetFileNameFranka": "urdf/franka_description/robots/franka_panda_gripper.urdf"
        },
        "enableCameraSensors": True
    },
    "sim": {
        "dt": 0.01667,  # 1/60
        "substeps": 2,
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "gravity": [0.0, 0.0, -9.81],
        "physx": {
            "num_threads": 4,
            "solver_type": 1,
            "use_gpu": True,  # set to False to run on CPU
            "num_position_iterations": 8,
            "num_velocity_iterations": 1,
            "contact_offset": 0.005,
            "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.2,
            "max_depenetration_velocity": 1000.0,
            "default_buffer_size_multiplier": 5.0,
            "max_gpu_contact_pairs": 1048576,  # 1024*1024
            "num_subscenes": 4,
            "contact_collection": 0  # 0: CC_NEVER (don't collect contact info), 1: CC_LAST_SUBSTEP (collect only contacts on last substep), 2: CC_ALL_SUBSTEPS (broken - do not use!)
        }
    },
    "task": {
        "randomize": False
    }
})


ig_fetchlike_config = ml_collections.ConfigDict({
    'name': 'FrankaPushingFetchlike',
    "physics_engine": "physx",
    "env": {
        'numEnvs': 0,
        "envSpacing": 1.5,
        "episodeLength": 50,
        "enableDebugVis": False,
        "clipObservations": 5.0,
        "clipActions": 1.0,
        "startRotationNoise": 0.785,
        "frankaPositionNoise": 0.0,
        "frankaRotationNoise": 0.0,
        "frankaDofNoise": 0.25,
        "distRewardScale": 1,
        "distRewardDropoff": 30,
        "distRewardThreshold": 0.05,
        "aggregateMode": 3,
        "actionScale": 0.5,
        "mode": "",
        "distanceFromBlock": 0.0,
        "nCubes": 1,
        "startPositionNoise": 0.15,
        "goalPositionNoise": 0.15,
        "friction": 0.5,
        "rigidCubes": False,
        "testTask": -1,
        "renderEveryEpisodes": 1000,
        "controlType": "osc",  # options are {joint_tor, osc}
        "observeVelocities": False,
        "asset": {
            "assetRoot": "../../assets",
            "assetFileNameFranka": "urdf/franka_description/robots/franka_panda_gripper.urdf"
        },
        "enableCameraSensors": True,
        # "enableCameraSensors": False,
    },
    "sim": {
        "dt": 0.04,  # 1/60
        "substeps": 2,
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "gravity": [0.0, 0.0, -9.81],
        "physx": {
            "num_threads": 4,
            "solver_type": 1,
            "use_gpu": True,  # set to False to run on CPU
            "num_position_iterations": 8,
            "num_velocity_iterations": 1,
            "contact_offset": 0.005,
            "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.2,
            "max_depenetration_velocity": 1000.0,
            "default_buffer_size_multiplier": 5.0,
            "max_gpu_contact_pairs": 1048576,  # 1024*1024
            "num_subscenes": 4,
            "contact_collection": 0  # 0: CC_NEVER (don't collect contact info), 1: CC_LAST_SUBSTEP (collect only contacts on last substep), 2: CC_ALL_SUBSTEPS (broken - do not use!)
        }
    },
    "task": {
        "randomize": False
    }
})


ant_config = ml_collections.ConfigDict({
    'name': 'Ant',
    "physics_engine": "physx",
    "env": {
        "envSpacing": 5,
        "episodeLength": 1000,
        "enableDebugVis": False,
        "clipActions": 1.0,
        "powerScale": 1.0,
        "controlFrequencyInv": 1,  # 60 Hz
        "headingWeight": 0.5,
        "upWeight": 0.1,
        "actionsCost": 0.005,
        "energyCost": 0.05,
        "dofVelocityScale": 0.2,
        "contactForceScale": 0.1,
        "jointsAtLimitCost": 0.1,
        "deathCost": -2.0,
        "terminationHeight": 0.31,
        "successThreshold": 0.0,
        "goalNoise": 100.0,
        "plane": {
            "staticFriction": 1.0,
            "dynamicFriction": 1.0,
            "restitution": 0.0,
        },
        "asset": {
            "assetFileName": "mjcf/nv_ant.xml",
        },
        # "enableCameraSensors": True,
        "enableCameraSensors": False,
    },
    "sim": {
        "dt": 0.0166,  # 1/60
        "substeps": 2,
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "gravity": [0.0, 0.0, -9.81],
        "physx": {
            "num_threads": 4,
            "solver_type": 1,
            "use_gpu": True,  # set to False to run on CPU
            "num_position_iterations": 4,
            "num_velocity_iterations": 0,
            "contact_offset": 0.02,
            "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.2,
            "max_depenetration_velocity": 10.0,
            "default_buffer_size_multiplier": 5.0,
            "max_gpu_contact_pairs": 8388608,  # 1024*1024
            "num_subscenes": 4,
            "contact_collection": 0  # 0: CC_NEVER (don't collect contact info), 1: CC_LAST_SUBSTEP (collect only contacts on last substep), 2: CC_ALL_SUBSTEPS (broken - do not use!)
        }
    },
    "task": {
        "randomize": False,
        "randomization_params": {
            "frequency": 600,  # Define how many environment steps between generating new randomizations
            "observations": {
                "range": [0, 0.002],  # range for the white noise
                "operation": "additive",
                "distribution": "gaussian"
            },
            "actions": {
                "range": [0.0, 0.02],
                "operation": "additive",
                "distribution": "gaussian"
            },
            "actor_params": {
                "ant": {
                    "color": True,
                    "rigid_body_properties": {
                        "mass": {
                            "range": [0.5, 1.5],
                            "operation": "scaling",
                            "distribution": "uniform",
                            "setup_only": True  # Property will only be randomized once before simulation is started.
                        }
                    },
                    "dof_properties": {
                        "damping": {
                            "range": [0.5, 1.5],
                            "operation": "scaling",
                            "distribution": "uniform"
                        },
                        "stiffness": {
                            "range": [0.5, 1.5],
                            "operation": "scaling",
                            "distribution": "uniform"
                        },
                        "lower": {
                            "range": [0, 0.01],
                            "operation": "additive",
                            "distribution": "gaussian"
                        },
                        "upper": {
                            "range": [0, 0.01],
                            "operation": "additive",
                            "distribution": "gaussian"
                        }
                    }
                }
            }
        }
    }
})
