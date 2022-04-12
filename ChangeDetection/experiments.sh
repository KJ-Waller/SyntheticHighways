python run_experiments.py --exp_name nonoise --dataset_dir ./dataset/ --noise False --noise_config 0 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 2000
python run_experiments.py --exp_name noiseconfig0 --dataset_dir ./dataset/ --noise True --noise_config 0 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 2000
python run_experiments.py --exp_name noiseconfig1 --dataset_dir ./dataset/ --noise True --noise_config 1 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 2000
python run_experiments.py --exp_name noiseconfig2 --dataset_dir ./dataset/ --noise True --noise_config 2 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 2000
python run_experiments.py --exp_name noiseconfig3 --dataset_dir ./dataset/ --noise True --noise_config 3 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 2000

python run_experiments.py --exp_name highsample_nonoise --dataset_dir ./dataset/ --noise False --noise_config 0 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000
python run_experiments.py --exp_name highsample_noiseconfig0 --dataset_dir ./dataset/ --noise True --noise_config 0 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000
python run_experiments.py --exp_name highsample_noiseconfig1 --dataset_dir ./dataset/ --noise True --noise_config 1 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000
python run_experiments.py --exp_name highsample_noiseconfig2 --dataset_dir ./dataset/ --noise True --noise_config 2 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000
python run_experiments.py --exp_name highsample_noiseconfig3 --dataset_dir ./dataset/ --noise True --noise_config 3 --map_index 0 --bbox 52.355 52.365 4.860 4.900 --n_traj 10000