import os
import argparse
from metrics import *
from datetime import datetime

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='high_sample', type=str, help='Name of experiment')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--n_traj', default=1, type=int, help='Number of trajectories to sample. 0 is all')
    parser.add_argument('--num_cpu_hmm', default=4, type=int, help='Number of CPUs to use for HMM change detector')

    parser.add_argument('--map_index', default=0, type=int, help='Index for which map to run experiment')
    parser.add_argument('--bbox', nargs='+', default=[52.355, 52.365, 4.860, 4.900], type=float, help='Set bounding box to train on map')

    args = parser.parse_args()

    starttime = datetime.now()
    starttime_experiments = starttime
    print(f'Starting Experiment 1 w/ No Noise - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python run_experiments.py --exp_name {args.exp_name}_nonoise --results_dir results_{args.exp_name} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 1 Finished w/ No Noise - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')

    starttime = datetime.now()
    print(f'Starting Experiment 2 w/ Noise Config 0 - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python run_experiments.py --exp_name {args.exp_name}_noiseconfig0 --noise --noise_config 0 --results_dir results_{args.exp_name} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 2 Finished w/ Noise Config 0 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
    
    starttime = datetime.now()
    print(f'Starting Experiment 3 w/ Noise Config 1 - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python run_experiments.py --exp_name {args.exp_name}_noiseconfig1 --noise --noise_config 1 --results_dir results_{args.exp_name} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 3 Finished w/ Noise Config 1 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
    
    starttime = datetime.now()
    print(f'Starting Experiment 4 w/ Noise Config 2 - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python run_experiments.py --exp_name {args.exp_name}_noiseconfig2 --noise --noise_config 2 --results_dir results_{args.exp_name} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 4 Finished w/ Noise Config 2 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
    
    starttime = datetime.now()
    print(f'Starting Experiment 5 w/ Noise Config 3 - Start Time: {starttime.strftime("%H:%M:%S")}')
    os.system(f"python run_experiments.py --exp_name {args.exp_name}_noiseconfig3 --noise --noise_config 3 --results_dir results_{args.exp_name} --dataset_dir {args.dataset_dir} \
                --num_cpu_hmm {args.num_cpu_hmm} --map_index {args.map_index} --bbox {args.bbox[0]} {args.bbox[1]} {args.bbox[2]} {args.bbox[3]} --n_traj {args.n_traj} \
                    --split_threshold {args.split_threshold}")
    stoptime = datetime.now()
    delta = stoptime - starttime
    print(f'Experiment 5 Finished w/ Noise Config 3 - End Time: {stoptime.strftime("%H:%M:%S")}, Duration: {str(delta)}')
    delta = stoptime - starttime_experiments
    print(f'All experiments finished at {stoptime.strftime("%H:%M:%S")}. Total duration: {str(delta)}')

    fscore_vs_noise(folder=f'./results_{args.exp_name}/', savename=f'./results_{args.exp_name}/fscore_vs_noise')
    plt.close()
    prauc_vs_noise(folder=f'./results_{args.exp_name}/', savename=f'./results_{args.exp_name}/prauc_vs_noise')