#! /bin/bash

echo $SLURM_JOB_ID
echo slurm job id
singularity run -B /var/lib/dcv-gl --nv --writable-tmpfs  --bind /global/scratch/users/prestonfu/tmp:/tmp --bind /global/scratch/users/prestonfu/miniconda3/:/home/miniconda3 /global/scratch/users/prestonfu/taskmaster.sif -- bash /global/scratch/users/prestonfu/taskmaster/isaacgymenvs/run_container.sh  slurm_job_id=$SLURM_JOB_ID "$@"
