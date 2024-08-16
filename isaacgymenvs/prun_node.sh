#! /bin/bash

# Run PPO on compute node
echo $SLURM_JOB_ID
echo slurm job id
singularity run --nv --writable-tmpfs  --bind /global/scratch/users/oleh/tmp:/tmp /global/scratch/users/oleh/taskmaster4.sif -- bash /global/scratch/users/oleh/taskmaster/isaacgymenvs/prun_container.sh  --slurm_job_id=$SLURM_JOB_ID "$@"