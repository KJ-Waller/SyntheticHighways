
# Run all experiments on seed 42
python -m experiment_scripts.noise_experiment --exp_name noise_exp --n_traj 0 --num_cpu_hmm 64 --seed 42;
python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000_wnoise --max_trajectories 10000 --noise --noise_config 1 --num_cpu_hmm 64 --seed 42;
python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_wnoise --n_traj 0 --noise --noise_config 1 --num_cpu_hmm 64 --seed 42;
python -m experiment_scripts.histogram_experiment --exp_name hist_exp --seed 42;
python -m experiment_scripts.histogram_resolution_experiment --exp_name histres_exp --seed 42;
python -m experiment_scripts.rulebased_experiment --exp_name rulebased_exp --seed 42;

# Run all experiments on seed 142
# python -m experiment_scripts.noise_experiment --exp_name noise_exp --n_traj 0 --num_cpu_hmm 64 --seed 142;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000_wnoise --max_trajectories 10000 --noise --noise_config 1 --num_cpu_hmm 64 --seed 142;
# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_wnoise --n_traj 0 --noise --noise_config 1 --num_cpu_hmm 64 --seed 142;
# python -m experiment_scripts.histogram_experiment --exp_name hist_exp --seed 142;
# python -m experiment_scripts.histogram_resolution_experiment --exp_name histres_exp --seed 142;
# python -m experiment_scripts.rulebased_experiment --exp_name rulebased_exp --seed 142;


# python -m experiment_scripts.noise_experiment --exp_name noise_exp --n_traj 0 --seed 142 --num_cpu_hmm 32;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000_wnoise --max_trajectories 10000 --noise --noise_config 1 --seed 142 --num_cpu_hmm 32;
# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_wnoise --n_traj 0 --noise --noise_config 1 --seed 142 --num_cpu_hmm 32;
# python -m experiment_scripts.histogram_experiment --exp_name hist_exp --seed 142;
# python -m experiment_scripts.histogram_resolution_experiment --exp_name histres_exp --seed 142;
# python -m experiment_scripts.rulebased_experiment --exp_name rulebased_exp --seed 142;

# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_moresteps --n_traj 0 --seed 42 --num_cpu_hmm 64;
# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_moresteps_wnoise --n_traj 0 --noise --noise_config 1 --seed 42 --num_cpu_hmm 64;

# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_moresteps_max10000 --n_traj 10000 --seed 42 --num_cpu_hmm 64;
# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_moresteps_wnoise_max10000 --n_traj 10000 --noise --noise_config 1 --seed 42 --num_cpu_hmm 64;