import argparse
import os
from SHDataset import SHDataset
from utils.utils import *
from utils.metrics import *
from models.hmm import HMMChangeDetector
from models.rulebased import RulebasedDetector
from multiprocessing import Pool
import random
from tqdm import tqdm
import pickle5 as pickle


def find_hmm_mismatches_parallel(G1, T2):
    hmm_det = HMMChangeDetector(G1)
    removed_edges = [(edge[0], edge[1]) for edge in G1.edges(data=True) if edge[2]['color'] == 'magenta']
    not_matched = []
    mismatched = []
    for t in tqdm(T2, desc='Matching trajectories'):
        path = hmm_det.map_match_trajectory(t)
        if len(path) == 0:
            not_matched.append(t)
            continue
        else:
            mismatched_t = False
            for e in path:
                if e in removed_edges:
                    mismatched_t = True
            if mismatched_t:
                mismatched.append((t, path))
        
    return not_matched, mismatched

def find_hmm_mismatches(G1, T2, num_cpu=64):
    G1 = G1.copy()
    
    # Split trajectories into equal sized chunks given cpu count
    T2 = np.array_split(T2, num_cpu)
    pool_input = [(G1, t) for t in T2]

    # Create processes pool and input
    pool = Pool(num_cpu)

    # Run pool map
    results = pool.starmap(find_hmm_mismatches_parallel, pool_input)
    pool.close()
    pool.join()
    
    not_matched, mismatched = [], []
    for r in results:
        not_matched += r[0]
        mismatched += r[1]
    
    return not_matched, mismatched

def find_rb_mismatches(G1, T2):
    G1 = G1.copy()

    # Create rule-based map matcher
    rb_det = RulebasedDetector(G1)

    # Find mismatched and non-matched trajectories
    removed_edges = [(edge[0], edge[1]) for edge in G1.edges(data=True) if edge[2]['color'] == 'magenta']
    not_matched = []
    mismatched = []
    for t in tqdm(T2, desc='Matching trajectories'):
        path = rb_det.map_match_trajectory(t)
        if len(path) == 0:
            not_matched.append(t)
            continue
        else:
            mismatched_t = False
            for e in path:
                if e in removed_edges:
                    mismatched_t = True
            if mismatched_t:
                mismatched.append((t, path))
        
    return not_matched, mismatched

def save_mismatches(G1, results):
    # Save mismatched and non matched trajectories and paths to a pickle file
    with open(os.path.join(args.results_dir, 'mismatched_trajectories.hdf5'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    removed_edges = [(edge[0], edge[1]) for edge in G1.edges(data=True) if edge[2]['color'] == 'magenta']
    # Unpack the results, save images of all trajectories with no matches 
    not_matched, mismatched = results
    nomatch_dir = os.path.join(args.results_dir, 'no_matches')
    if not os.path.exists(nomatch_dir):
        os.mkdir(nomatch_dir)
    for i, nm in enumerate(not_matched):
        plot_graph(snapshot_to_nxgraph(G1, [nm]), zoom_on_traj=True, T_edge_width=2, T_node_size=5,
            show_nodes=True, savename=os.path.join(nomatch_dir, f'{args.map_matcher}_nomatch_t_#{i}'), show_img=False)
    
    # Save images of all trajectories and their paths that were mismatched to removed roads
    mismatched_dir = os.path.join(args.results_dir, 'mismatched')
    if not os.path.exists(mismatched_dir):
        os.mkdir(mismatched_dir)
    for i, mm in enumerate(mismatched):
        t, path = mm
        mismatched_edges = []
        remaining_path = []
        for edge in path:
            if edge in removed_edges:
                mismatched_edges.append(edge)
            else:
                remaining_path.append(edge)

        # Color the path green, with mismatched removed edges colored as orange
        path_remaining_colors = {edge: 'green' for edge in remaining_path}
        path_removed_colors =  {edge: 'orange' for edge in mismatched_edges}
        nx.set_edge_attributes(G1, path_remaining_colors, name='color')
        nx.set_edge_attributes(G1, path_removed_colors, name='color')

        # Save image of trajectory
        plot_graph(snapshot_to_nxgraph(G1, [t]), zoom_on_traj=True, T_edge_width=3, T_node_size=8, figsize=(8,8), 
            show_nodes=True, traj_alpha=0.25, G_edge_width=3, G_node_size=8, savename=os.path.join(mismatched_dir, f'{args.map_matcher}_mismatched_t_#{i}'), show_img=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', default='hmm_analysis', type=str, help='Name of experiment')
    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--noise', default=False, action='store_true', help='Add noise to trajectories')
    parser.add_argument('--noise_config', default=0, type=int, help='Which noise configuration to use')
    parser.add_argument('--resample_everyn_t', default=1, type=int, help='Resample trajectories every n timed point')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--n_traj', default=0, type=int, help='Number of trajectories to sample. 0 is all')
    parser.add_argument('--num_cpu_hmm', default=10, type=int, help='Number of CPUs to use for HMM change detector')
    parser.add_argument('--replot_only', default=False, action='store_true', help='Whether to load mismatches from hdf5 pickle file or rerun HMM map-matcher')
    parser.add_argument('--map_matcher', default='hmm', type=str, help="Which map-matcher to use. 'hmm' and 'rb' are the options")

    parser.add_argument('--map_index', default=0, type=int, help='Index for which map to run experiment')
    parser.add_argument('--bbox', nargs='+', default=[52.335, 52.36, 4.89, 4.92], type=float, help='Set bounding box to train on map')
    parser.add_argument('--seed', default=42, type=int, help="What random seed to use for experiments for reproducibility")

    args = parser.parse_args()

    # Set seed for random libraries
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create folder for experimental results
    if not os.path.exists('./experimental_results/'):
        os.mkdir('./experimental_results/')
    args.results_dir = os.path.join('./experimental_results/', args.exp_name)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Setup dataset
    bbox = tuple(args.bbox)
    dataset = SHDataset(noise=False)
    G1,T1,G2,T2 = dataset.read_snapshots(0, bbox=bbox)
    G12_diff = compare_snapshots(G1, G2)[1]
    
    # Sample subset of trajectories
    total_t2 = len(T2['T'])
    if args.n_traj != 0 and args.n_traj != total_t2:
        T1['T'] = random.sample(T1['T'], k=args.n_traj)
        T2['T'] = random.sample(T2['T'], k=args.n_traj)
    print(f"Sampled {len(T2['T'])}/{total_t2} trajectories for T2")

    # Run trajectory analysis
    if args.replot_only:
        results_fname = os.path.join(args.results_dir, 'mismatched_trajectories.hdf5')
        if not os.path.exists(results_fname):
            raise ValueError(f'File {results_fname} not found. Please rerun map-matcher without using --replot_only')
        with open(results_fname, 'rb') as handle:
            results = pickle.load(handle)
    else:
        if args.map_matcher.lower() == 'hmm':
            results = find_hmm_mismatches(G12_diff, T2['T'], args.num_cpu_hmm)
        elif args.map_matcher.lower() == 'rb':
            results = find_rb_mismatches(G12_diff, T2['T'])

    # Save mismatches as images
    save_mismatches(G12_diff, results)