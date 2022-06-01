import os
import argparse
from utils.metrics import *
from utils.utils import *
from datetime import datetime
import random
import numpy as np

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='high_sample', type=str, help='Name of experiment')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--num_cpu_hmm', default=64, type=int, help='Number of CPUs to use for HMM change detector')
    parser.add_argument('--steps', default=6, type=int, help='How many different intervals to run for number of trajectories.')
    parser.add_argument('--step_size', default=3, type=int, help='How big the step is for each interval for temporal resampling of trajectories.')
    parser.add_argument('--noise', default=False, action='store_true', help='Add noise to trajectories')
    parser.add_argument('--noise_config', default=0, type=int, help='Which noise configuration to use')
    parser.add_argument('--n_traj', default=1, type=int, help='Number of trajectories to sample. 0 is all')

    parser.add_argument('--map_index', default=0, type=int, help='Index for which map to run experiment')
    # parser.add_argument('--bbox', nargs='+', default=[52.34, 52.35, 4.89, 4.93], type=float, help='Set bounding box to train on map')
    # parser.add_argument('--bbox', nargs='+', default=[52.34, 52.36, 4.895, 4.93], type=float, help='Set bounding box to train on map')
    parser.add_argument('--bbox', nargs='+', default=[52.335, 52.36, 4.89, 4.92], type=float, help='Set bounding box to train on map')

    parser.add_argument('--seeds', nargs='+', default=[42, 142, 420], type=int, help='What random seeds to use for the experiments for reproducibility')

    args = parser.parse_args()

    # Run for each seed
    for seed in args.seeds:
        # Set seed for random libraries
        np.random.seed(seed)
        random.seed(seed)

        # Get the range of resample rates to run experiments for
        resample_traj_steps = np.arange(1,(args.steps*args.step_size)+1, args.step_size)

        # Run experiments
        starttime_experiments = datetime.now()
        for i, resample_everyn in enumerate(resample_traj_steps):
            starttime = datetime.now()
            print(f'Starting Experiment {i+1} Resampling trajectories every {resample_everyn} timed points - Start Time: {starttime.strftime("%H:%M:%S")}')
            if args.noise:
                os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{seed}_resample_step{resample_everyn} --results_dir results_{args.exp_name}_seed{seed} --dataset_dir {args.dataset_dir} \
                            --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                                --split_threshold {args.split_threshold} --seed {seed} --resample_everyn_t {resample_everyn} --noise --noise_config {args.noise_config}")
            else:
                os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{seed}_resample_step{resample_everyn} --results_dir results_{args.exp_name}_seed{seed} --dataset_dir {args.dataset_dir} \
                            --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                                --split_threshold {args.split_threshold} --seed {seed} --resample_everyn_t {resample_everyn}")
            stoptime = datetime.now()
            delta = stoptime - starttime
            print(f'Experiment {i+1} Finished Resampling trajectories every {resample_everyn} timed points - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')

        delta = stoptime - starttime_experiments
        print(f'All experiments finished at {stoptime.strftime("%H:%M:%S")}. Total duration: {str(delta)}')

        # Save plots
        x_vs_fscore(x='Temporal Sparsity Interval', labels=resample_traj_steps, xlabel='Resampling Interval',
                    folder=f'./experimental_results/results_{args.exp_name}_seed{seed}/', 
                    savename=f'./experimental_results/results_{args.exp_name}_seed{seed}/tfreq_vs_fscore')
        plt.close()
        x_vs_prauc(x='Temporal Sparsity Interval', labels=resample_traj_steps, xlabel='Resampling Interval',
                    folder=f'./experimental_results/results_{args.exp_name}_seed{seed}/', 
                    savename=f'./experimental_results/results_{args.exp_name}_seed{seed}/tfreq_vs_prauc')

        # Save GIF of how the trajectories change
        save_gif(folder=f'./experimental_results/results_{args.exp_name}_seed{seed}/',
                    img_name='G1T2', savename=f'./experimental_results/results_{args.exp_name}_seed{seed}/G1T2')

    # Average results from different seeds and plot in 'figures' folder
    plot_results(folder_prefix=f'results_{args.exp_name}_seed', x='Temporal Sparsity Interval', xlabels=resample_traj_steps, xlabel='Resampling Interval')