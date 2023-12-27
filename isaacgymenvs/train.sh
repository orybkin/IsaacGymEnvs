# AWR - 1 cube
bash run_local.sh 5 experiment= task.env.nCubes=1 train.params.config.entropy_coef=1e-2 task.env.actionScale=0.5 train=FrankaPushingAWR task.env.distRewardThreshold=0.02 train.params.config.horizon_length=128 train.params.config.learning_rate=1e-3 train.params.config.max_epochs=1000

# PPO - 1 cube
for i in {1..5}
do
   CUDA_VISIBLE_DEVICES=2 python train.py wandb_activate=True experiment= task.env.nCubes=1 train.params.config.entropy_coef=2e-3 task.env.actionScale=0.5 train.params.config.horizon_length=1024 task.env.distRewardThreshold=0.02 train.params.config.lr_schedule=fixed train.params.config.learning_rate=2e-3  train.params.config.max_epochs=280
done

# PPO - soft reward
# for i in {1..5}
# do
#    python train.py wandb_activate=True experiment= task.env.nCubes=1 train.params.config.entropy_coef=2e-3 train.params.config.horizon_length=64 task.env.actionScale=0.5 train.params.config.max_epochs=1500 x

# AWR - soft reward
bash run_local.sh experiment= task.env.nCubes=1 train.params.config.entropy_coef=1e-2 train=FrankaPushingAWR  task.env.actionScale=0.5 train.params.config.max_epochs=3000

for i in {1..5}
do
   CUDA_VISIBLE_DEVICES=1  python train.py wandb_activate=True experiment= task.env.nCubes=1 train.params.config.entropy_coef=1e-2 task.env.actionScale=0.5 train=FrankaPushingAWR task.env.distRewardThreshold=0.02 train.params.config.horizon_length=128 train.params.config.learning_rate=1e-3 train.params.config.relabel=True train.params.algo.name=gc_a2c_continuous train.params.config.max_epochs=1000
done


for i in {1..5}
do
   CUDA_VISIBLE_DEVICES=1 python train.py headless=True wandb_activate=True experiment= task=FrankaReaching2 train=FrankaPushingAWR train.params.config.entropy_coef=2e-2 train.params.config.horizon_length=64 num_envs=1024 train.params.config.learning_rate=1e-3 train.params.algo.name=gc_a2c_continuous train.params.config.relabel=True task.env.frankaDofNoise=0 +task.env.pushing_like=False train.params.config.max_epochs=25
done


for i in {1..5}
do
   python train.py headless=True wandb_activate=True experiment= task=FrankaReaching2 train=FrankaPushingAWR train.params.config.entropy_coef=2e-2 train.params.config.horizon_length=64 num_envs=1024 train.params.config.learning_rate=1e-3 task.env.frankaDofNoise=0 +task.env.pushing_like=False train.params.config.max_epochs=25
done

for i in {1..5}
do
   CUDA_VISIBLE_DEVICES=1 python train.py headless=True wandb_activate=True experiment= task=FrankaReaching2 train=FrankaPushingPAWR train.params.config.entropy_coef=2e-3  train.params.config.horizon_length=64 train.params.config.learning_rate=2e-3 train.params.config.lr_schedule=fixed  train.params.config.awr_coef=0.1 train.params.config.max_epochs=50
done


for i in {1..5}
do
   CUDA_VISIBLE_DEVICES=1 python train.py headless=True wandb_activate=True experiment= task=FrankaReaching2 train=FrankaPushingPPO train.params.config.entropy_coef=2e-3 train.params.config.horizon_length=64 train.params.config.learning_rate=2e-3 train.params.config.lr_schedule=fixed train.params.config.max_epochs=50
done

for i in {1..5}
do
   python train.py headless=True wandb_activate=True experiment= task=FrankaReaching2 train=FrankaPushingAWR train.params.config.entropy_coef=2e-2  train.params.config.horizon_length=32 train.params.config.learning_rate=1e-3 train.params.config.max_epochs=100
done


for i in {1..5}
do
   python train.py headless=True wandb_activate=True experiment= task=FrankaReaching2 train=FrankaPushingAWR train.params.config.entropy_coef=2e-2  train.params.config.horizon_length=32 train.params.config.learning_rate=1e-3 train.params.config.relabel=True train.params.algo.name=gc_a2c_continuous train.params.config.max_epochs=100
done