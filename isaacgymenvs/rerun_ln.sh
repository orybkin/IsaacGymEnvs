name=$(echo "$@" | grep -o 'experiment=[^ ]*' | cut -d '=' -f 2)

for i in $(seq $2 $(( $1 + $2 - 1 ))); do 
    restart=$(printf "runs/%s_%s/nn/%s.pth" "$i" "$name" "$name")
    sbatch --partition=savio4_gpu --account=co_rail --cpus-per-task=4 --gres=gpu:A5000:1 -t 50:00:00 --qos=rail_gpu4_normal --output=/global/home/users/oleh/slurm_logs/%j.log /global/scratch/users/oleh/taskmaster/isaacgymenvs/run_node.sh "${@:3}" restart=$restart
done