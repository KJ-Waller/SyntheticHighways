python run_experiments.py --exp_name nonoise --dataset_dir ./dataset/ --num_cpu_hmm 16 --map_index 0 --bbox 52.34 52.35 4.89 4.93 --n_traj 2000
# python run_experiments.py --exp_name noiseconfig0 --dataset_dir ./dataset/ --noise --num_cpu_hmm 8 --noise_config 0 --map_index 0 --bbox 52.34 52.35 4.89 4.93 --n_traj 2000
# python run_experiments.py --exp_name noiseconfig1 --dataset_dir ./dataset/ --noise --num_cpu_hmm 8 --noise_config 1 --map_index 0 --bbox 52.34 52.35 4.89 4.93 --n_traj 2000
python run_experiments.py --exp_name noiseconfig2 --dataset_dir ./dataset/ --noise --num_cpu_hmm 16 --noise_config 2 --map_index 0 --bbox 52.34 52.35 4.89 4.93 --n_traj 2000
python run_experiments.py --exp_name noiseconfig3 --dataset_dir ./dataset/ --noise --num_cpu_hmm 16 --noise_config 3 --map_index 0 --bbox 52.34 52.35 4.89 4.93 --n_traj 2000

# python run_experiments.py --exp_name highsample_nonoise --dataset_dir ./dataset/ --num_cpu_hmm 8 --noise_config 0 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000
# python run_experiments.py --exp_name highsample_noiseconfig0 --dataset_dir ./dataset/ --noise --num_cpu_hmm 8 --noise_config 0 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000
# python run_experiments.py --exp_name highsample_noiseconfig1 --dataset_dir ./dataset/ --noise --num_cpu_hmm 8 --noise_config 1 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000
# python run_experiments.py --exp_name highsample_noiseconfig2 --dataset_dir ./dataset/ --noise --num_cpu_hmm 8 --noise_config 2 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000
# python run_experiments.py --exp_name highsample_noiseconfig3 --dataset_dir ./dataset/ --noise --num_cpu_hmm 8 --noise_config 3 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000