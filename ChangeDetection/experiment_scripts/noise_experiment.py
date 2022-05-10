import os
import argparse
from utils.metrics import *
from datetime import datetime

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='high_sample', type=str, help='Name of experiment')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--n_traj', default=1, type=int, help='Number of trajectories to sample. 0 is all')
    parser.add_argument('--num_cpu_hmm', default=64, type=int, help='Number of CPUs to use for HMM change detector')

    parser.add_argument('--map_index', default=0, type=int, help='Index for which map to run experiment')
    parser.add_argument('--bbox', nargs='+', default=[52.34, 52.35, 4.89, 4.93], type=float, help='Set bounding box to train on map')

    parser.add_argument('--seed', default=42, type=int, help="What random seed to use for experiments for reproducibility")

    args = parser.parse_args()

    # Start running noise experiment
    starttime = datetime.now()
    starttime_experiments = starttime
    print(f'Starting Experiment 1 w/ No Noise - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{args.seed}_nonoise --results_dir results_{args.exp_name}_seed{args.seed} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold} --seed {args.seed}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 1 Finished w/ No Noise - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')

    starttime = datetime.now()
    print(f'Starting Experiment 2 w/ Noise Config 0 - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{args.seed}_noiseconfig0 --noise --noise_config 0 --results_dir results_{args.exp_name}_seed{args.seed} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold} --seed {args.seed}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 2 Finished w/ Noise Config 0 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
    
    starttime = datetime.now()
    print(f'Starting Experiment 3 w/ Noise Config 1 - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{args.seed}_noiseconfig1 --noise --noise_config 1 --results_dir results_{args.exp_name}_seed{args.seed} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold} --seed {args.seed}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 3 Finished w/ Noise Config 1 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
    
    starttime = datetime.now()
    print(f'Starting Experiment 4 w/ Noise Config 2 - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{args.seed}_noiseconfig2 --noise --noise_config 2 --results_dir results_{args.exp_name}_seed{args.seed} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold} --seed {args.seed}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 4 Finished w/ Noise Config 2 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
    
    starttime = datetime.now()
    print(f'Starting Experiment 5 w/ Noise Config 3 - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python -m experiment_scripts.exp_all_methods --exp_name {args.exp_name}_seed{args.seed}_noiseconfig3 --noise --noise_config 3 --results_dir results_{args.exp_name}_seed{args.seed} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold} --seed {args.seed}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 5 Finished w/ Noise Config 3 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
    delta = stoptime - starttime_experiments
    print(f'All experiments finished at {stoptime.strftime("%H:%M:%S")}. Total duration: {str(delta)}')
 
    # Save plots
    x_labels = ['No Noise', 'Noise Config 1', 'Noise Config 2', 'Noise Config 3', 'Noise Config 4']
    x_vs_fscore(x='Noise', labels=x_labels,
                folder=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/', 
                savename=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/noise_vs_fscore')
    plt.close()
    x_vs_prauc(x='Noise', labels=x_labels,
                folder=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/', 
                savename=f'./experimental_results/results_{args.exp_name}_seed{args.seed}/noise_vs_prauc')