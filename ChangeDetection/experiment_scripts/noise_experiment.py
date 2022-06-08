import os
import argparse
from utils.metrics import *
from utils.utils import *
from SHDataset import *
from datetime import datetime
import random
import numpy as np

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='high_sample', type=str, help='Name of experiment')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--n_traj', default=1, type=int, help='Number of trajectories to sample. 0 is all')
    parser.add_argument('--num_cpu_hmm', default=64, type=int, help='Number of CPUs to use for HMM change detector')

    parser.add_argument('--map_index', default=0, type=int, help='Index for which map to run experiment')
    parser.add_argument('--bbox', nargs='+', default=[52.335, 52.36, 4.89, 4.92], type=float, help='Set bounding box to train on map')

    parser.add_argument('--seeds', nargs='+', default=[42, 142, 420], type=int, help='What random seeds to use for the experiments for reproducibility')

    args = parser.parse_args()

    # Run for every seed
    for seed in args.seeds:
        # Set seed for random libraries
        np.random.seed(seed)
        random.seed(seed)

        # Start running noise experiment
        starttime = datetime.now()
        starttime_experiments = starttime
        print(f'Starting Experiment 1 w/ No Noise - Start Time: {starttime.strftime("%H:%M:%S")}')
        os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{seed}_nonoise --results_dir results_{args.exp_name}_seed{seed} --dataset_dir {args.dataset_dir} \
                    --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                        --split_threshold {args.split_threshold} --seed {seed}")
        stoptime = datetime.now()
        delta = stoptime - starttime
        print(f'Experiment 1 Finished w/ No Noise - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')

        starttime = datetime.now()
        print(f'Starting Experiment 2 w/ Noise Config 0 - Start Time: {starttime.strftime("%H:%M:%S")}')
        os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{seed}_noiseconfig0 --noise --noise_config 0 --results_dir results_{args.exp_name}_seed{seed} --dataset_dir {args.dataset_dir} \
                    --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                        --split_threshold {args.split_threshold} --seed {seed}")
        stoptime = datetime.now()
        delta = stoptime - starttime
        print(f'Experiment 2 Finished w/ Noise Config 0 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
        
        starttime = datetime.now()
        print(f'Starting Experiment 3 w/ Noise Config 1 - Start Time: {starttime.strftime("%H:%M:%S")}')
        os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{seed}_noiseconfig1 --noise --noise_config 1 --results_dir results_{args.exp_name}_seed{seed} --dataset_dir {args.dataset_dir} \
                    --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                        --split_threshold {args.split_threshold} --seed {seed}")
        stoptime = datetime.now()
        delta = stoptime - starttime
        print(f'Experiment 3 Finished w/ Noise Config 1 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
        
        starttime = datetime.now()
        print(f'Starting Experiment 4 w/ Noise Config 2 - Start Time: {starttime.strftime("%H:%M:%S")}')
        os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{seed}_noiseconfig2 --noise --noise_config 2 --results_dir results_{args.exp_name}_seed{seed} --dataset_dir {args.dataset_dir} \
                    --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                        --split_threshold {args.split_threshold} --seed {seed}")
        stoptime = datetime.now()
        delta = stoptime - starttime
        print(f'Experiment 4 Finished w/ Noise Config 2 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
        
        starttime = datetime.now()
        print(f'Starting Experiment 5 w/ Noise Config 3 - Start Time: {starttime.strftime("%H:%M:%S")}')
        os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{seed}_noiseconfig3 --noise --noise_config 3 --results_dir results_{args.exp_name}_seed{seed} --dataset_dir {args.dataset_dir} \
                    --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                        --split_threshold {args.split_threshold} --seed {seed}")
        stoptime = datetime.now()
        delta = stoptime - starttime
        print(f'Experiment 5 Finished w/ Noise Config 3 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
        delta = stoptime - starttime_experiments
        print(f'All experiments finished at {stoptime.strftime("%H:%M:%S")}. Total duration: {str(delta)}')
    
        # Save plots
        noise_in_meters = measure_noise()
        x_labels = [f"{round(noise_conf['meters'],1)}m"  for noise_conf in noise_in_meters]
        x_labels = ['No Noise', *x_labels]

        x_vs_fscore(x='Noise', labels=x_labels, xlabel='Noise',
                    folder=f'./experimental_results/results_{args.exp_name}_seed{seed}/', 
                    savename=f'./experimental_results/results_{args.exp_name}_seed{seed}/noise_vs_fscore')
        plt.close()
        x_vs_prauc(x='Noise', labels=x_labels, xlabel='Noise',
                    folder=f'./experimental_results/results_{args.exp_name}_seed{seed}/', 
                    savename=f'./experimental_results/results_{args.exp_name}_seed{seed}/noise_vs_prauc')

        # Save GIF of how the trajectories change
        save_gif(folder=f'./experimental_results/results_{args.exp_name}_seed{seed}/',
                    img_name='G1T2', savename=f'./experimental_results/results_{args.exp_name}_seed{seed}/G1T2')
    
    # Average results from different seeds and plot in 'figures' folder
    plot_results(folder_prefix=f'results_{args.exp_name}_seed', x='Noise', xlabels=x_labels, xlabel='Noise')