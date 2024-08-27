echo "$@"
. /global/scratch/users/prestonfu/miniconda3/etc/profile.d/conda.sh
conda activate cubes
export LD_LIBRARY_PATH=/global/scratch/users/prestonfu/miniconda3/envs/cubes/lib/:$LD_LIBRARY_PATH
cd /global/scratch/users/prestonfu/taskmaster/isaacgymenvs
DEVICE=$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
# python train.py wandb_activate=True sim_device=cuda:$DEVICE rl_device=cuda:$DEVICE graphics_device_id=$DEVICE "$@"
python clean_awr/awr.py "$@"  --agent.device=cuda:$DEVICE --agent.graphics_device_id=$DEVICE