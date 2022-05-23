import os
import argparse
from utils.metrics import *
from utils.utils import *
from datetime import datetime
from SHDataset import SHDataset
import random
import numpy as np

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='high_sample', type=str, help='Name of experiment')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--num_cpu_hmm', default=32, type=int, help='Number of CPUs to use for HMM change detector')
    parser.add_argument('--num_steps', default=5, type=int, help='How many different intervals to run for number of trajectories.')
    parser.add_argument('--noise', default=False, action='store_true', help='Add noise to trajectories')
    parser.add_argument('--noise_config', default=0, type=int, help='Which noise configuration to use')
    parser.add_argument('--max_trajectories', default=0, type=int, help='The maximum number of trajectories to use for the experiment. If 0, use all trajectories.')

    parser.add_argument('--map_index', default=0, type=int, help='Index for which map to run experiment')
    parser.add_argument('--bbox', nargs='+', default=[52.34,52.36, 4.90, 4.93], type=float, help='Set bounding box to train on map')

    parser.add_argument('--seed', default=42, type=int, help="What random seed to use for experiments for reproducibility")

    args = parser.parse_args()

    # Set seed for random libraries
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup dataset just to get total number of trajectories
    dataset = SHDataset(noise=args.noise, dataset_dir=args.dataset_dir, noise_config=args.noise_config, split_threshold=args.split_threshold)
    G1,T1,G2,T2 = dataset.read_snapshots(args.map_index, bbox=tuple(args.bbox))
    total_t2 = len(T2['T'])

    # Get the intervals/steps for which number of trajectories to run per experiment
    if args.max_trajectories > total_t2:
        raise ValueError(f'--max_trajectories argument set to {args.max_trajectories}, which is larger than the total trajectories in specified bbox: {total_t2}')
    if args.max_trajectories == 0:
        n_traj_steps = np.linspace(0, total_t2, args.num_steps+1, dtype=np.int64)[1:]
    else:
        n_traj_steps = np.linspace(0, args.max_trajectories, args.num_steps+1, dtype=np.int64)[1:]

    # Run experiments
    starttime_experiments = datetime.now()
    for i, n_traj in enumerate(n_traj_steps):
        starttime = datetime.now()
        print(f'Starting Experiment {i+1} w/ {n_traj}# of trajectories - Start Time: {starttime.strftime("%H:%M:%S")}')
        if args.noise:
            os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{args.seed}_e#{i}_{n_traj}#_t --results_dir results_{args.exp_name}_seed{args.seed} --dataset_dir {args.dataset_dir} \
                        --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {n_traj} \
                            --split_threshold {args.split_threshold} --seed {args.seed} --noise --noise_config {args.noise_config}")
        else:
            os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{args.seed}_e#{i}_{n_traj}#_t --results_dir results_{args.exp_name}_seed{args.seed} --dataset_dir {args.dataset_dir} \
                        --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {n_traj} \
                            --split_threshold {args.split_threshold} --seed {args.seed}")
        stoptime = datetime.now()
        delta = stoptime - starttime
        print(f'Experiment {i+1} Finished w/ {n_traj}# of trajectories - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')

    delta = stoptime - starttime_experiments
    print(f'All experiments finished at {stoptime.strftime("%H:%M:%S")}. Total duration: {str(delta)}')

    # Save plots
    x_labels = [f'{n_traj}' for n_traj in n_traj_steps]
    x_vs_fscore(x='# of trajectories', labels=x_labels, xlabel='# of traces',
                folder=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/', 
                savename=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/#traj_vs_fscore')
    plt.close()
    x_vs_prauc(x='# of trajectories', labels=x_labels, xlabel='# of traces',
                folder=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/', 
                savename=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/#traj_vs_prauc')

    # Save GIF of how the trajectories change
    save_gif(folder=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/',
                img_name='G1T2', savename=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/G1T2')