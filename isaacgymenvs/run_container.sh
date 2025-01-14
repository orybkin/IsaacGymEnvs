echo "$@"
. /home/miniconda3/etc/profile.d/conda.sh
conda activate cubes
export LD_LIBRARY_PATH=/home/miniconda3/envs/cubes/lib/:$LD_LIBRARY_PATH
cd /global/scratch/users/oleh/taskmaster
pip install -e . 
cd isaacgymenvs
DEVICE=$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
python train.py wandb_activate=True sim_device=cuda:$DEVICE rl_device=cuda:$DEVICE graphics_device_id=$DEVICE "$@"