import argparse
from SHDataset import SHDataset
from models.histogram import HistogramDetector
import random
import numpy as np
from utils.metrics import *
from utils.utils import *

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='high_sample', type=str, help='Name of experiment')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--noise', default=False, action='store_true', help='Add noise to trajectories')
    parser.add_argument('--noise_config', default=0, type=int, help='Which noise configuration to use')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--n_traj', default=0, type=int, help='Number of trajectories to sample. 0 is all')
    parser.add_argument('--histogram_dims', nargs='+', default=[200, 200], type=int, help='What dimensions to make the histogram for the Histogram based change detector')

    parser.add_argument('--map_index', default=0, type=int, help='Index for which map to run experiment')
    parser.add_argument('--bbox', nargs='+', default=[52.34, 52.35, 4.89, 4.93], type=float, help='Set bounding box to train on map')
    parser.add_argument('--seed', default=42, type=int, help="What random seed to use for experiments for reproducibility")

    args = parser.parse_args()

    # Set seed for random libraries
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create folder for experimental results
    if not os.path.exists('./experimental_results/'):
        os.mkdir('./experimental_results/')
    results_dir = os.path.join('./experimental_results/', f'results_{args.exp_name}_seed{args.seed}')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Setup Dataset
    dataset = SHDataset(noise=args.noise, dataset_dir=args.dataset_dir,
                        noise_config=args.noise_config, split_threshold=args.split_threshold,
                        resample_timedp=False)

    G1,T1,G2,T2 = dataset.read_snapshots(args.map_index, bbox=tuple(args.bbox))

    # Sample subset of trajectories
    total_t2 = len(T2['T'])
    if args.n_traj != 0 and args.n_traj != total_t2:
        T1['T'] = random.sample(T1['T'], k=args.n_traj)
        T2['T'] = random.sample(T2['T'], k=args.n_traj)
    print(f"Sampled {len(T2['T'])}/{total_t2} trajectories for T2")

    # Save figures from bbox showing raw traces and changes
    plot_graph(snapshot_to_nxgraph(G1,T2['T']), figsize=(10,10), savename=os.path.join(results_dir , 'G1T2'), show_img=False)
    _, G12_d, _, _ = compare_snapshots(G1,G2)
    plot_graph(G12_d, figsize=(10,10), savename=os.path.join(results_dir , 'Changes'), show_nodes=True, show_img=False)

    # Get groundtruth labels
    gt_labels = groundtruth_labels(G1, G2)

    # First experiment: Histogram method accumulating trajectories per cell and calculating scores by looking at all cells edges intersect
    hist_det = HistogramDetector(G1, tuple(args.bbox), hist_dims=tuple(args.histogram_dims),
                                 score_calc_method='intersect', accumulate_scores_hist=True)
    G2_pred_hist_ia = hist_det.forward(T2['T'])
    plot_graph(G2_pred_hist_ia, use_weights=True, figsize=(10,10), 
                savename=os.path.join(results_dir , 'heatmap_hist_intersect_accum'), show_img=False)
    scores_hist_ia = predicted_labels(G2_pred_hist_ia)
    p_hist_ia, r_hist_ia, ts_hist_ia, pr_auc_hist_ia = PRCurve(gt_labels, scores_hist_ia, 
        savename=os.path.join(results_dir, 'prcurve_logscale_hist_intersect_accum'))
    p_hist_ia, r_hist_ia, ts_hist_ia, pr_auc_hist_ia = PRCurve(gt_labels, scores_hist_ia, 
        savename=os.path.join(results_dir, 'prcurve_hist_intersect_accum'), log_scale=False)
    threshold_hist_ia = hist_det.find_threshold()
    predictions_hist_ia = {k: 0 if scores_hist_ia[k] < threshold_hist_ia else 1 for k in gt_labels}
    fscore_hist_ia = fscore(gt_labels, predictions_hist_ia)

    # Second experiment: Histogram method accumulating trajectories per cell and calculating scores by looking at all cells edges intersect in the center
    hist_det = HistogramDetector(G1, tuple(args.bbox), hist_dims=tuple(args.histogram_dims), 
                                score_calc_method='center', accumulate_scores_hist=True)
    G2_pred_hist_ca = hist_det.forward(T2['T'])
    plot_graph(G2_pred_hist_ca, use_weights=True, figsize=(10,10), 
                savename=os.path.join(results_dir , 'heatmap_hist_center_accum'), show_img=False)
    scores_hist_ca = predicted_labels(G2_pred_hist_ca)
    p_hist_ca, r_hist_ca, ts_hist_ca, pr_auc_hist_ca = PRCurve(gt_labels, scores_hist_ca, 
        savename=os.path.join(results_dir, 'prcurve_logscale_hist_center_accum'))
    p_hist_ca, r_hist_ca, ts_hist_ca, pr_auc_hist_ca = PRCurve(gt_labels, scores_hist_ca, 
        savename=os.path.join(results_dir, 'prcurve_hist_center_accum'), log_scale=False)
    threshold_hist_ca = hist_det.find_threshold()
    predictions_hist_ca = {k: 0 if scores_hist_ca[k] < threshold_hist_ca else 1 for k in gt_labels}
    fscore_hist_ca = fscore(gt_labels, predictions_hist_ca)

    # Third experiment: Histogram method w/ trajectory occurence per cell and calculating scores by looking at all cells edges intersect
    hist_det = HistogramDetector(G1, tuple(args.bbox), hist_dims=tuple(args.histogram_dims), 
                                score_calc_method='intersect', accumulate_scores_hist=False)
    G2_pred_hist_io = hist_det.forward(T2['T'])
    plot_graph(G2_pred_hist_io, use_weights=True, figsize=(10,10), 
                savename=os.path.join(results_dir , 'heatmap_hist_intersect_occur'), show_img=False)
    scores_hist_io = predicted_labels(G2_pred_hist_io)
    p_hist_io, r_hist_io, ts_hist_io, pr_auc_hist_io = PRCurve(gt_labels, scores_hist_io, 
        savename=os.path.join(results_dir, 'prcurve_logscale_hist_intersect_occur'))
    p_hist_io, r_hist_io, ts_hist_io, pr_auc_hist_io = PRCurve(gt_labels, scores_hist_io, 
        savename=os.path.join(results_dir, 'prcurve_hist_intersect_occur'), log_scale=False)
    threshold_hist_io = hist_det.find_threshold()
    predictions_hist_io = {k: 0 if scores_hist_io[k] < threshold_hist_io else 1 for k in gt_labels}
    fscore_hist_io = fscore(gt_labels, predictions_hist_io)

    # Fourth experiment: Histogram method w/ trajectory occurence per cell and calculating scores by looking at all cells edges intersect in the center
    hist_det = HistogramDetector(G1, tuple(args.bbox), hist_dims=tuple(args.histogram_dims), 
                                score_calc_method='center', accumulate_scores_hist=False)
    G2_pred_hist_co = hist_det.forward(T2['T'])
    plot_graph(G2_pred_hist_co, use_weights=True, figsize=(10,10), 
                savename=os.path.join(results_dir , 'heatmap_hist_center_occur'), show_img=False)
    scores_hist_co = predicted_labels(G2_pred_hist_co)
    p_hist_co, r_hist_co, ts_hist_co, pr_auc_hist_co = PRCurve(gt_labels, scores_hist_co, 
        savename=os.path.join(results_dir, 'prcurve_logscale_hist_center_occur'))
    p_hist_co, r_hist_co, ts_hist_co, pr_auc_hist_co = PRCurve(gt_labels, scores_hist_co, 
        savename=os.path.join(results_dir, 'prcurve_hist_center_occur'), log_scale=False)
    threshold_hist_co = hist_det.find_threshold()
    predictions_hist_co = {k: 0 if scores_hist_co[k] < threshold_hist_co else 1 for k in gt_labels}
    fscore_hist_co = fscore(gt_labels, predictions_hist_co)

    # Plot combined pr curves
    prs, rs, aucs = [p_hist_ia, p_hist_ca, p_hist_io, p_hist_co], \
                    [r_hist_ia, r_hist_ca, r_hist_io, r_hist_co], \
                    [pr_auc_hist_ia, pr_auc_hist_ca, pr_auc_hist_io, pr_auc_hist_co]
    labels = ['Intersect+Accum', 'Center+Accum', 'Intersect+Occur', 'Center+Occur']
    PRCombine(ps=prs, rs=rs, aucs=aucs, labels=labels, savename=os.path.join(results_dir, 'prcurve_logscale_combined'), log_scale=True)
    PRCombine(ps=prs, rs=rs, aucs=aucs, labels=labels, savename=os.path.join(results_dir, 'prcurve_combined'), log_scale=False)

    # Plot bar chart comparing f-scores
    plt.close()
    fscores = [fscore_hist_ia, fscore_hist_ca, fscore_hist_io, fscore_hist_co]
    bar_fscore(fscores, labels, savename=os.path.join(results_dir, 'hist_comp_barplot'))