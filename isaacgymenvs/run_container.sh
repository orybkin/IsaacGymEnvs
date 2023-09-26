. /home/miniconda3/etc/profile.d/conda.sh
conda activate rlgpu
export LD_LIBRARY_PATH=/home/miniconda3/envs/rlgpu/lib/:$LD_LIBRARY_PATH
cd /global/scratch/users/oleh/taskmaster
rm -rf /global/home/users/oleh/.local/lib/python3.7/site-packages/isaacgymenvs.egg-link 
pip install -q -e . 
cd isaacgymenvs
python train.py task=FrankaPushing headless=True wandb_activate=True wandb_project=taskmaster "$@"