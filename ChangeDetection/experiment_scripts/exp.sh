# Experiment running on multiple seeds

# python -m experiment_scripts.noise_experiment --exp_name noise_exp --n_traj 0 --num_cpu_hmm 64 --seeds 42 142 420;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_wnoise --noise --noise_config 1 --num_cpu_hmm 64 --seeds 42 142 420;
# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_wnoise --n_traj 0 --noise --noise_config 1 --num_cpu_hmm 64 --seeds 42 142 420;

# python -m experiment_scripts.noise_experiment --exp_name noise_exp_seeds_test --n_traj 160 --num_cpu_hmm 64 --seeds 42 142 420;

python -m experiment_scripts.histogram_resolution_experiment --exp_name histres_exp_2 --seed 42;

# Run all experiments on seed 42
# python -m experiment_scripts.noise_experiment --exp_name noise_exp --n_traj 0 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000_wnoise --max_trajectories 10000 --noise --noise_config 1 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_wnoise --n_traj 0 --noise --noise_config 1 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.histogram_experiment --exp_name hist_exp --seed 42;
# python -m experiment_scripts.histogram_resolution_experiment --exp_name histres_exp --seed 42;
# python -m experiment_scripts.rulebased_experiment --exp_name rulebased_exp --seed 42;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_wnoise --noise --noise_config 1 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.hmm_analysis --exp_name hmm_analysis_exp --n_traj 0 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000 --max_trajectories 10000 --num_cpu_hmm 64 --seed 42;

# Different bbox
# python -m experiment_scripts.noise_experiment --exp_name noise_exp_bbox2 --n_traj 0 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000_wnoise_bbox2 --max_trajectories 10000 --noise --noise_config 1 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.tfreq_experiment --exp_name tfreq_exp_wnoise_bbox2 --n_traj 0 --noise --noise_config 1 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.histogram_experiment --exp_name hist_exp_bbox2 --seed 42;
# python -m experiment_scripts.histogram_resolution_experiment --exp_name histres_exp_bbox2 --seed 42;
# python -m experiment_scripts.rulebased_experiment --exp_name rulebased_exp_bbox2 --seed 42;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_wnoise_bbox2 --noise --noise_config 1 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_bbox2 --num_cpu_hmm 64 --seed 42;
# python -m experiment_scripts.mapmatch_analysis --exp_name hmm_analysis_exp_bbox2 --n_traj 5000 --num_cpu_hmm 64 --seed 42 --map_matcher hmm --replot_only;
# python -m experiment_scripts.mapmatch_analysis --exp_name rb_analysis_exp_bbox2 --n_traj 5000 --num_cpu_hmm 64 --seed 42 --map_matcher rb --replot_only;
# python -m experiment_scripts.sparsity_experiment --exp_name sparsity_exp_max10000_bbox2 --max_trajectories 10000 --num_cpu_hmm 64 --seed 42;

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