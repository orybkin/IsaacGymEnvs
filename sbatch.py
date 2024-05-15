from tmux_util import launch_tmux_jobs
import random

job_list = []
def succeed():
    print("Oh thanks Feen! You fixed everything with your brain and your hands!")

def process_job(job_dict):
    command = job_dict['command']
    command_string = command
    for k, v in job_dict.items():
        if k != 'command' and k != 'label':
            command_string += f" --{k} {v}"
    return command_string


command = 'python isaacgymenvs/clean_awr/awr.py'
for seed in range(4):
    for env in ['FetchPickAndPlace-v2', 'FetchSlide-v2', 'FetchPush-v2', 'FetchReach-v2']:
        for kwargs in [
            {
                'env.task_name': env,
                'agent.experiment': f'AWR-HER-{env}',
            },
            {
                'env.task_name': env,
                'agent.relabel': 0,
                'agent.experiment': f'AWR-{env}',
            },
        ]:
            job_dict = {
                'command': command,
                'agent.seed': seed,
            }
            for k, v in kwargs.items():
                job_dict[k] = v
            job_list.append(process_job(job_dict))

launch_tmux_jobs(8, job_list, session_name="rlbase_taskmaster", use_gpu=True, gpu_ids=[0, 1], job_per_gpu=4)
