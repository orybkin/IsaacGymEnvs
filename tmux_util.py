import os
import subprocess
from subprocess import run, PIPE

import libtmux
server = libtmux.Server()

def launch_tmux_jobs(num_sessions, commands, session_name="auto", use_gpu=False, gpu_ids=[0], job_per_gpu=1):
    jobs = [[] for i in range(num_sessions)]
    while len(commands) > 0:
        for i in range(num_sessions):
            if len(commands) > 0:
                c = commands.pop(0)
                print(c)
                jobs[i].append(c)
            else:
                break
    print("Launching Jobs")

    # Make tmux session.
    # run('tmux kill-server'.split(), stdout=PIPE, encoding='ascii')
    run(f'tmux new -d -s {session_name}'.split(), stdout=PIPE, encoding='ascii')
    session = server.sessions.filter(session_name=session_name)[0]
    window = session.attached_window

    assert len(gpu_ids)*job_per_gpu >= num_sessions

    for i in range(num_sessions):
        print("PANE '{}'".format(i))
        for c in commands:
            print("COMMAND: {}".format(c))
        
        if i == 0:
            pane = window.attached_pane
        else:
            pane = window.split_window()
        pane.send_keys('cd ~/taskmaster', enter=True)
        pane.send_keys('conda activate rlgames', enter=True)
        commands = jobs[i]

        for c in commands:
            gid = gpu_ids[i//job_per_gpu]
            c += f' --agent.device cuda:{gid} --agent.graphics_device_id {gid}'
            pane.send_keys(c, enter=True)
        window.select_layout('tiled')

        

def make_seeds(num_seeds, command):
    commands = []
    for seed in range(num_seeds):
        commands.append(command + ' --seed {}'.format(seed))
    return commands