echo "$@"
conda activate cubes2
export LD_LIBRARY_PATH=/global/scratch/users/prestonfu/miniconda3/envs/cubes2/lib/:$LD_LIBRARY_PATH
cd /global/scratch/users/prestonfu/taskmaster/isaacgymenvs 
DEVICE=$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
python train.py wandb_activate=True sim_device=cuda:$DEVICE rl_device=cuda:$DEVICE graphics_device_id=$DEVICE "$@"
