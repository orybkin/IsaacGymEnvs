"""
Visualize wandb logs post-mortem.
"""
import os
import os.path as osp

import argparse
import math
import pandas as pd
import numpy as np
import re
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

logdir = "wandb_post/log"
os.makedirs(logdir, exist_ok=True)
vizdir = "wandb_post/viz"
os.makedirs(vizdir, exist_ok=True)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """https://stackoverflow.com/a/5967539"""
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def to_list(x):
    try:
        iter(x)
        if isinstance(x, pd.Series):
            return x.tolist()
        return x
    except TypeError:
        return [x]
    

class Scalar:
    def __init__(self):
        self._run_dict = {}
        
    @property
    def run_names(self):
        return list(self._run_dict.keys())
        
    @property
    def runs(self):
        return list(self._run_dict.values())
    
    def update(self, run_name, step, value):
        if run_name not in self.run_names:
            self._run_dict[run_name] = Run()
        self._run_dict[run_name].update(step, value)


class Run:
    def __init__(self):
        self.steps = []
        self.values = []
        
    @property
    def size(self):
        return len(self.values)
        
    def update(self, step, value):
        step, value = to_list(step), to_list(value)
        self.steps.extend(step)
        self.values.extend(value)


def get_max_row_size(alpha, dtype=float):
    assert 0. <= alpha < 1.
    epsilon = np.finfo(dtype).tiny
    return int(np.log(epsilon)/np.log(1-alpha)) + 1

def process_run(run, tags, data, step_attr_name="global_step"):
    print(f"Processing {run.name}")

    # fill existing data
    if osp.exists(osp.join(logdir, f"{run.name}.csv")):
        df = pd.read_csv(osp.join(logdir, f"{run.name}.csv")).set_index(step_attr_name)
        steps = df.index.tolist()
        for tag in set(df.columns).intersection(tags):
            data[tag].update(run.name, steps, df[tag])
        tags_to_fill = list(set(tags).difference(df.columns))
        history = run.scan_history(keys=tags_to_fill)
    else:
        df = None
        tags_to_fill = tags
        history = run.scan_history(keys=[step_attr_name] + tags_to_fill)

    # fill data from wandb
    if df is None:
        steps = [int(row[step_attr_name]) for row in history]
    buffer = {step_attr_name: steps}
    for tag in tags_to_fill:
        values = [row[tag] for row in history]
        data[tag].update(run.name, steps, values)
        buffer[tag] = values
    
    if df is None:
        df = pd.DataFrame(buffer)
    else:
        # append new data to old stuff
        # TODO: probably something wrong here if you try adding things on different step scales.
        # Make steps another thing in buffer and merge on
        df = pd.concat([df, pd.DataFrame(buffer).set_index(step_attr_name)], axis=1).reset_index()
    df.to_csv(osp.join(logdir, f"{run.name}.csv"), index=False)


def compute_mean_std(scalar: Scalar, ninterp=100):
    """https://gist.github.com/kylestach/6e37430bc37e4922804a7bb1a6a231a1"""
    min_step = min([run.steps[0] for run in scalar.runs])
    max_step = max([run.steps[-1] for run in scalar.runs])
    steps = np.linspace(min_step, max_step, ninterp)
    scalars_interp = np.stack([
        np.interp(steps, run.steps, run.values, left=float('nan'), right=float('nan'))
        for run in scalar.runs
    ], axis=1)

    mean = np.mean(scalars_interp, axis=1)
    std = np.std(scalars_interp, axis=1)
    return steps, mean, std

def ema(x, alpha):
    # TODO: update this (slow!)
    return pd.Series(x).ewm(alpha=1-alpha).mean()


def do_plot(ax: plt.Axes, scalar: Scalar, **kwargs):
    mode = kwargs.get('mode', 'each')
    smoothing = kwargs.get('smoothing', None)
    alpha = kwargs.get('alpha', 0)
    ymin = kwargs.get('ymin', None)
    ymax = kwargs.get('ymax', None)
    if smoothing is None:
        smoothing_fn = lambda x: x
    elif smoothing == 'ema':
        smoothing_fn = lambda x: ema(x, alpha)
    else:
        raise ValueError()
    
    if mode == 'each':
        for run_name, run in scalar._run_dict.items():
            ax.plot(run.steps, smoothing_fn(run.values), label=run_name)
    elif mode == 'avg':
        steps, mean, std = compute_mean_std(scalar)
        smoothed_mean = smoothing_fn(mean)
        smoothed_std = smoothing_fn(std)
        ax.fill_between(steps, smoothed_mean-smoothed_std, smoothed_mean+smoothed_std, alpha=0.3)
        ax.plot(steps, smoothed_mean)
    else:
        raise ValueError()
    
    # TODO: add ability to set ylim bounds independently
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-t", "--title", type=str, default=None)
    parser.add_argument("--mode", type=str, default='each')
    parser.add_argument("--smoothing", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--run_filter", type=str, default=None)
    parser.add_argument("--scalar_filter", type=str, default=None)
    args = parser.parse_args()

    if args.run_filter == None:
        response = input("Warning: Not filtering based on run names is memory-intensive. Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            exit(0)
    if args.scalar_filter == None:
        response = input("Warning: Not filtering based on scalars will produce a very large plot. Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            exit(0)

    api = wandb.Api()
    if args.run_filter is None:
        runs = api.runs(f"{args.entity}/{args.project}")
    else:
        runs = api.runs(f"{args.entity}/{args.project}", filters={"display_name": {"$regex": args.run_filter}})

    print("Collecting data")
    tags = list(runs[0].history().columns)
    if args.scalar_filter is not None:
        tags = [t for t in tags if re.match(args.scalar_filter, t)]
    tags.sort(key=natural_keys)
    num_tags = len(tags)
    print(f"{num_tags} scalar tags matched")
    data = {tag: Scalar() for tag in tags}
    runs = [run for run in runs]
    runs.sort(key=lambda run: natural_keys(run.name))
    for run in runs:
        process_run(run, tags, data)

    print("Plotting")
    if args.output:
        filename = args.output
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = f"{timestamp}.png"
        
    suptitle_fontsize = 16
    axis_label_fontsize = 12
    tick_label_fontsize = 10
    legend_fontsize = 10
    subplot_hspace = 0.1
    subplot_vspace = 0.2
    padding = 0.1
    legend_kwargs = dict(loc='lower center', bbox_to_anchor=[0.5, 1], prop={'size': legend_fontsize})
    
    if num_tags == 1:
        figsize = (5, 4)
        legend_ncol = 2
        plt.figure(figsize=figsize)
        tag = tags[0]
        for run_name, run in data[tag]._run_dict.items():
            plt.plot(run.steps, run.values, label=run_name)
        plt.xlabel("Steps")
        plt.legend(ncol=legend_ncol, **legend_kwargs)
        plt.suptitle(
            args.title if args.title is not None else tag, 
            y=1+0.05*math.ceil(len(runs)/legend_ncol), 
            fontsize=suptitle_fontsize)
    else:
        figsize = (4.5, 3)
        legend_ncol = 3
        legend_kwargs["bbox_to_anchor"][1] -= padding / 2
        plot_kwargs = dict(mode=args.mode, smoothing=args.smoothing, alpha=args.alpha, ymin=args.ymin, ymax=args.ymax)
        num_cols = math.ceil(math.sqrt(num_tags))
        num_rows = math.ceil(num_tags / num_cols)
        fig, axs = plt.subplots(
            num_rows, num_cols, sharex=True, sharey=True, squeeze=False, 
            figsize=(figsize[0] * num_cols, figsize[1] * num_rows))
        handles, labels, unique_labels = [], [], set()
        for k, tag in enumerate(tags):
            i, j = k // num_cols, k % num_cols
            ax = axs[i, j]
            ax.set_title(tag)
            do_plot(ax, data[tag], **plot_kwargs)
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in unique_labels:
                    handles.append(handle)
                    labels.append(label)
                    unique_labels.add(label)
        if len(handles) > 0:
            fig.legend(handles, labels, ncol=legend_ncol, **legend_kwargs)
            suptitle_y = 1 + 0.1 * math.ceil(len(runs)/legend_ncol) / num_rows
        else:
            suptitle_y = 1 - padding / 2
        fig.suptitle(
            args.title if args.title is not None else osp.splitext(filename)[0], 
            y=suptitle_y, fontsize=suptitle_fontsize)
        plt.setp(axs[-1, :], xlabel='Steps')
        fig.subplots_adjust(left=padding, right=1-padding, top=1-padding, bottom=padding, hspace=subplot_vspace, wspace=subplot_hspace)


    if args.title is None:
        args.title = osp.splitext()
    plt.savefig(osp.join(vizdir, filename), bbox_inches="tight")