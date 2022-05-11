# python -m experiment_scripts.noise_experiment --exp_name noise_exp --n_traj 0;
python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000_wnoise --max_trajectories 10000 --noise --noise_config 1;
python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_wnoise --n_traj 0 --noise --noise_config 1;
# python -m experiment_scripts.histogram_experiment --exp_name hist_exp;
python -m experiment_scripts.histogram_resolution_experiment --exp_name histres_exp;


python -m experiment_scripts.noise_experiment --exp_name noise_exp --n_traj 0 --seed 142;
python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000_wnoise --max_trajectories 10000 --noise --noise_config 1 --seed 142;
python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_wnoise --n_traj 0 --noise --noise_config 1 --seed 142;
python -m experiment_scripts.histogram_experiment --exp_name hist_exp --seed 142;
python -m experiment_scripts.histogram_resolution_experiment --exp_name histres_exp --seed 142;