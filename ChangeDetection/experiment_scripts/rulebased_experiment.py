import argparse
from SHDataset import SHDataset
from models.rulebased import RulebasedDetector
from utils.metrics import *
from utils.utils import *
import random
import numpy as np

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='high_sample', type=str, help='Name of experiment')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--noise', default=False, action='store_true', help='Add noise to trajectories')
    parser.add_argument('--noise_config', default=0, type=int, help='Which noise configuration to use')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--n_traj', default=0, type=int, help='Number of trajectories to sample. 0 is all')
    parser.add_argument('--steps', default=10, type=int, help='How many different intervals to run for heading parameter')

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

    # Get the range of values to use for heading parameter Ch
    heading_weights = np.linspace(0, 1.0, args.steps+1)

    # Run experiments while saving results
    results = []
    for i, Ch in enumerate(heading_weights):
        print(f'Running experiment {i}/{heading_weights.size}')
        rule_det = RulebasedDetector(G1, Ch=Ch)
        G2_pred_rb = rule_det.forward(T2['T'])
        plot_graph(G2_pred_rb, use_weights=True, figsize=(10,10), savename=os.path.join(results_dir, f'heatmap_rulebased_Ch{Ch}'), show_img=False)
        scores_rb = predicted_labels(G2_pred_rb)
        p_rb, r_rb, ts_rb, pr_auc_rb = PRCurve(gt_labels, scores_rb, savename=os.path.join(results_dir, f'prcurve_logscale_rulebased_Ch{Ch}'))
        p_rb, r_rb, ts_rb, pr_auc_rb = PRCurve(gt_labels, scores_rb, savename=os.path.join(results_dir, f'prcurve_rulebased_Ch{Ch}'), log_scale=False)
        predictions_rb = {k: int(scores_rb[k] == 0) for k in gt_labels}
        fscore_rb = fscore(gt_labels, predictions_rb)

        results.append({
            'scores': scores_rb,
            'precision': p_rb,
            'recall': r_rb,
            'prauc': pr_auc_rb,
            'thresholds': ts_rb,
            'predictions': predictions_rb,
            'fscore': fscore_rb,
            'Ch': Ch
        })
    
    # Save figures
    fscores = [res['fscore'] for res in results]
    Chs = [res['Ch'] for res in results]
    praucs = [res['prauc'] for res in results]
    plt.close()
    Ch_vs_y(fscores, Chs, y='F-Score', savename=os.path.join(results_dir, f'Ch_vs_fscore'))
    Ch_vs_y(praucs, Chs, y='PR-AUC', savename=os.path.join(results_dir, f'Ch_vs_prauc'))