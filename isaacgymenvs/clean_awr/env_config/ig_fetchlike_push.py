import ml_collections
from ml_collections import config_flags
from absl import app, flags
FLAGS = flags.FLAGS

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

def get_config():
    return ig_fetchlike_config