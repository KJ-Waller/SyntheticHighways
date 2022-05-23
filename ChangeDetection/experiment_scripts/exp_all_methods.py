import argparse
from SHDataset import SHDataset
import os
from utils.utils import *
from utils.metrics import *
from models.random import RandomDetector
from models.rulebased import RulebasedDetector
from models.hmm_fast import HMMChangeDetectorFast
from models.histogram import HistogramDetector
import random
import pickle5 as pickle
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--exp_name', default='high_sample', type=str, help='Name of experiment')
    parser.add_argument('--results_dir', default='results', type=str, help='Folder in which to store results')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--noise', default=False, action='store_true', help='Add noise to trajectories')
    parser.add_argument('--noise_config', default=0, type=int, help='Which noise configuration to use')
    parser.add_argument('--resample_everyn_t', default=1, type=int, help='Resample trajectories every n timed point')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--n_traj', default=1, type=int, help='Number of trajectories to sample. 0 is all')
    parser.add_argument('--num_cpu_hmm', default=4, type=int, help='Number of CPUs to use for HMM change detector')
    parser.add_argument('--histogram_dims', nargs='+', default=[200, 200], type=int, help='What dimensions to make the histogram for the Histogram based change detector')

    parser.add_argument('--map_index', default=0, type=int, help='Index for which map to run experiment')
    parser.add_argument('--bbox', nargs='+', default=[52.34, 52.36, 4.895, 4.93], type=float, help='Set bounding box to train on map')
    parser.add_argument('--seed', default=42, type=int, help="What random seed to use for experiments for reproducibility")

    args = parser.parse_args()

    # Set seed for random libraries
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create folder for experimental results
    if not os.path.exists('./experimental_results/'):
        os.mkdir('./experimental_results/')
    args.results_dir = os.path.join('./experimental_results/', args.results_dir)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    results_folder = os.path.join(args.results_dir, args.exp_name)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    
    # Setup dataset snapshots
    if args.resample_everyn_t == 1:
        dataset = SHDataset(noise=args.noise, dataset_dir=args.dataset_dir,
                            noise_config=args.noise_config, split_threshold=args.split_threshold,
                            resample_timedp=False)
    else:
        dataset = SHDataset(noise=args.noise, dataset_dir=args.dataset_dir,
                            noise_config=args.noise_config, split_threshold=args.split_threshold,
                            resample_timedp=True, resample_everyn=args.resample_everyn_t)

    G1,T1,G2,T2 = dataset.read_snapshots(args.map_index, bbox=tuple(args.bbox))

    # Sample subset of trajectories
    total_t2 = len(T2['T'])
    if args.n_traj != 0 and args.n_traj != total_t2:
        T1['T'] = random.sample(T1['T'], k=args.n_traj)
        T2['T'] = random.sample(T2['T'], k=args.n_traj)
    print(f"Sampled {len(T2['T'])}/{total_t2} trajectories for T2")

    # Save figures from bbox showing raw traces and changes
    plot_graph(snapshot_to_nxgraph(G1,T2['T']), figsize=(8,8), savename=os.path.join(results_folder, 'G1T2'), show_img=False)
    _, G12_d, _, _ = compare_snapshots(G1,G2)
    plot_graph(G12_d, figsize=(8,8), savename=os.path.join(results_folder , 'Changes'), 
                show_nodes=True, show_img=False, removed_road_edge_width=3)

    # Get groundtruth labels
    gt_labels = groundtruth_labels(G1, G2)

    # Run experiment for random change detector
    random_det = RandomDetector(G1)
    G2_pred_rand = random_det.forward(T2)
    plot_graph(G2_pred_rand, use_weights=True, figsize=(8,8), 
                savename=os.path.join(results_folder, 'heatmap_random'), show_img=False)
    scores_rand = predicted_labels(G2_pred_rand)
    p_rand, r_rand, ts_rand, pr_auc_rand = PRCurve(gt_labels, scores_rand, 
                savename=os.path.join(results_folder, 'prcurve_logscale_random'))
    p_rand, r_rand, ts_rand, pr_auc_rand = PRCurve(gt_labels, scores_rand, 
                savename=os.path.join(results_folder, 'prcurve_random'), log_scale=False)
    fscore_rand = fscore(gt_labels, scores_rand)

    # Run experiment for rulebased change detector
    rule_det = RulebasedDetector(G1)
    G2_pred_rb = rule_det.forward(T2['T'])
    plot_graph(G2_pred_rb, use_weights=True, figsize=(8,8), 
                savename=os.path.join(results_folder, 'heatmap_rulebased'), show_img=False)
    scores_rb = predicted_labels(G2_pred_rb)
    p_rb, r_rb, ts_rb, pr_auc_rb = PRCurve(gt_labels, scores_rb, 
                savename=os.path.join(results_folder, 'prcurve_logscale_rulebased'))
    p_rb, r_rb, ts_rb, pr_auc_rb = PRCurve(gt_labels, scores_rb, 
                savename=os.path.join(results_folder, 'prcurve_rulebased'), log_scale=False)
    predictions_rb = {k: int(scores_rb[k] == 0) for k in gt_labels}
    fscore_rb = fscore(gt_labels, predictions_rb)

    # Run experiment for histogram based change detector
    hist_det = HistogramDetector(G1, tuple(args.bbox), hist_dims=tuple(args.histogram_dims))
    G2_pred_hist = hist_det.forward(T2['T'])
    plot_graph(G2_pred_hist, use_weights=True, figsize=(8,8), 
                savename=os.path.join(results_folder, 'heatmap_hist'), show_img=False)
    scores_hist = predicted_labels(G2_pred_hist)
    p_hist, r_hist, ts_hist, pr_auc_hist = PRCurve(gt_labels, scores_hist, 
                savename=os.path.join(results_folder, 'prcurve_logscale_hist'))
    p_hist, r_hist, ts_hist, pr_auc_hist = PRCurve(gt_labels, scores_hist, 
                savename=os.path.join(results_folder, 'prcurve_hist'), log_scale=False)
    threshold_hist = hist_det.find_threshold()
    predictions_hist = {k: 0 if scores_hist[k] < threshold_hist else 1 for k in gt_labels}
    fscore_hist = fscore(gt_labels, predictions_hist)
    # For histogram method, also save the actual histogram
    save_hist(hist_det.hist, savename=os.path.join(results_folder, 'histogram'))

    # TODO: Remove this
    # T1['T'] = random.sample(T1['T'], k=1000)
    # T2['T'] = random.sample(T2['T'], k=1000)

    # Run experiment for hmm change detector
    hmm_det = HMMChangeDetectorFast(G1, num_cpu=args.num_cpu_hmm, use_latlon=False)
    G2_pred_hmm = hmm_det.forward(T2['T'])
    plot_graph(G2_pred_hmm, use_weights=True, figsize=(8,8), 
                savename=os.path.join(results_folder, 'heatmap_hmm'), show_img=False)
    scores_hmm = predicted_labels(G2_pred_hmm)
    p_hmm, r_hmm, ts_hmm, pr_auc_hmm = PRCurve(gt_labels, scores_hmm, 
                savename=os.path.join(results_folder, 'prcurve_logscale_hmm'))
    p_hmm, r_hmm, ts_hmm, pr_auc_hmm = PRCurve(gt_labels, scores_hmm, 
                savename=os.path.join(results_folder, 'prcurve_hmm'), log_scale=False)
    predictions_hmm = {k: int(scores_hmm[k] == 0) for k in gt_labels}
    fscore_hmm = fscore(gt_labels, predictions_hmm)

    # Create combine PR curve and save it
    prs, rs, aucs = [p_rand, p_rb, p_hist, p_hmm], [r_rand, r_rb, r_hist, r_hmm], [pr_auc_rand, pr_auc_rb, pr_auc_hist, pr_auc_hmm]
    PRCombine(ps=prs, rs=rs, aucs=aucs, labels=['Random', 'Rule-based', 'Histogram', 'HMM'], 
                savename=os.path.join(results_folder, 'prcurve_logscale_combined'), log_scale=True)
    PRCombine(ps=prs, rs=rs, aucs=aucs, labels=['Random', 'Rule-based', 'Histogram', 'HMM'], 
                savename=os.path.join(results_folder, 'prcurve_combined'), log_scale=False)

    quant_results = {
        'experiment_name': args.exp_name,
        'args': {
            'noise': args.noise,
            'noise_config': args.noise_config,
            'map_index': args.map_index,
            'bbox': args.bbox,
            'n_trajectories': args.n_traj
        },
        'results':{
            'random': {
                'precision': p_rand,
                'recall': r_rand,
                'thresholds': ts_rand,
                'pr_auc': pr_auc_rand,
                'scores': scores_rand,
                'fscore': fscore_rand
            },
            'rulebased': {
                'precision': p_rb,
                'recall': r_rb,
                'thresholds': ts_rb,
                'pr_auc': pr_auc_rb,
                'scores': scores_rb,
                'fscore': fscore_rb
            },
            'histogram': {
                'precision': p_hist,
                'recall': r_hist,
                'thresholds': ts_hist,
                'pr_auc': pr_auc_hist,
                'scores': scores_hist,
                'fscore': fscore_hist
            },
            'hmm': {
                'precision': p_hmm,
                'recall': r_hmm,
                'thresholds': ts_hmm,
                'pr_auc': pr_auc_hmm,
                'scores': scores_hmm,
                'fscore': fscore_hmm
            },
        }
    }

    quantitative_results_fname = os.path.join(results_folder, f'quantitative_results.hdf5')
    with open(quantitative_results_fname, 'wb') as handle:
        pickle.dump(quant_results, handle, protocol=pickle.HIGHEST_PROTOCOL)