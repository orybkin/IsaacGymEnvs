echo "$@"
. /home/miniconda3/etc/profile.d/conda.sh
conda activate cubes
export LD_LIBRARY_PATH=/home/miniconda3/envs/cubes/lib/:$LD_LIBRARY_PATH
cd /global/scratch/users/oleh/taskmaster
pip install -e . 
cd isaacgymenvs
python train.py task=FrankaPushing headless=True wandb_activate=True wandb_project=taskmaster "$@"