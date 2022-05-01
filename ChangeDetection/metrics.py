from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prcurve
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_curve, roc_auc_score
import networkx as nx
import numpy as np
import os
import pickle5 as pickle
from tqdm import tqdm

from SHDataset import SHDataset
from utils import compare_snapshots

def fscore(gt_labels, pred_labels):
    """
    Calculates f score between ground truth labels and predicted labels
    """
    labels = []
    predictions = []
    for key in gt_labels.keys():
        gt_score = gt_labels[key]
        pred_score = pred_labels[key]
        labels.append(gt_score)
        predictions.append(pred_score)
        
    return f1_score(labels, predictions)

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def groundtruth_labels(G1,G2):
    """
    Calculates the ground truth labels from the changes between G1 and G2
    """
    
    G1_edges = set([edge for edge in G1.edges])
    G2_edges = set([edge for edge in G2.edges])
    
    removed_edges = G1_edges.difference(G2_edges)
    labels = {edge: 1 if edge in removed_edges else 0 for edge in G1_edges}
    sorted_labels = {label: labels[label] for label in sorted(labels)}
    return sorted_labels

def predicted_labels(G_pred):
    """
    Extracts the predicted labels from a networkx graph
    """
    scores = nx.get_edge_attributes(G_pred, 'weight')
    sorted_scores = {edge: scores[edge] for edge in sorted(scores)}
    return sorted_scores
    
def PRCurve(gt_labels, pred_scores, log_scale=True, norm=False, savename=None):
    """
    Plots the precision recall curve given ground truth labels and predicted scores
    """
    plt.clf()
    predictions = {k: pred_scores[k] for k in gt_labels}

    if log_scale:
        ax = plt.gca()
        ax.set_yscale("log")
    if norm:
        gt_scores = normalize(list(gt_labels.values()))
        predictions = normalize(list(predictions.values()))
    else:
        gt_scores = np.array(list(gt_labels.values()))
        predictions = np.array(list(predictions.values()))
        

    p, r, ts = prcurve(gt_scores, predictions, pos_label=1.0)
    
    pr_auc = auc(r, p)

    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC: {round(pr_auc, 2)})')

    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()
    
    return p, r, ts, pr_auc

def PRCombine(ps, rs, aucs, labels=['Random', 'Rule-based'], log_scale=True, savename=None):
    """
    Combines the precision and recall curves for multiple methods, plots labels and AUCs
    """
    plt.clf()
    if log_scale:
        ax = plt.gca()
        ax.set_yscale("log")
    
    for p, r in zip(ps, rs):
        plt.plot(r, p)
        
    labels = [f'{label} (auc: {round(auroc, 2)})' for label, auroc in list(zip(labels, aucs))]

    plt.legend(labels)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()
    
def prune_edges(G, threshold):
    """
    Prune edges from map G with weight higher than threshold
    """
    
    G = G.copy()
    
    edges = G.copy().edges(data=True)
    for edge in edges:
        weight = edge[2]['weight']
        if weight > threshold:
            G.remove_edge(edge[0], edge[1])

    return G

def read_fscores_experiment(folder):
    folders = [f for f in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder,f))]
    results = []
    for f in folders:
        results_folder = os.path.join(folder, f)
        files = [file for file in os.listdir(results_folder) if '.hdf5' in file]
        if len(files) != 1:
            raise ValueError(f"Expected one results file (hdf5) in {results_folder}, but found {len(files)}: {files}")
        results_file = os.path.join(results_folder, files[0])
        with open(results_file, 'rb') as handle:
            result = pickle.load(handle)
        results.append(result)
        
    results = sorted(results, key=lambda res: res['experiment_name'], reverse=False)
    results = [results[-1], *results[0:-1]]
    
    fscores_random = []
    fscores_rulebased = []
    fscores_hist = []
    fscores_hmm = []
    for result in results:
        fscores_random.append(result['results']['random']['fscore'])
        fscores_rulebased.append(result['results']['rulebased']['fscore'])
        fscores_hist.append(result['results']['histogram']['fscore'])
        fscores_hmm.append(result['results']['hmm']['fscore'])
    return fscores_random, fscores_rulebased, fscores_hist, fscores_hmm

def read_prauc_experiment(folder):
    folders = [f for f in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder,f))]
    results = []
    for f in folders:
        results_folder = os.path.join(folder, f)
        files = [file for file in os.listdir(results_folder) if '.hdf5' in file]
        if len(files) != 1:
            raise ValueError(f"Expected one results file (hdf5) in {results_folder}, but found {len(files)}: {files}")
        results_file = os.path.join(results_folder, files[0])
        with open(results_file, 'rb') as handle:
            result = pickle.load(handle)
        results.append(result)
        
    results = sorted(results, key=lambda res: res['experiment_name'], reverse=False)
    results = [results[-1], *results[0:-1]]
    
    prauc_random = []
    prauc_rulebased = []
    prauc_hist = []
    prauc_hmm = []
    for result in results:
        prauc_random.append(result['results']['random']['pr_auc'])
        prauc_rulebased.append(result['results']['rulebased']['pr_auc'])
        prauc_hist.append(result['results']['histogram']['pr_auc'])
        prauc_hmm.append(result['results']['hmm']['pr_auc'])

    return prauc_random, prauc_rulebased, prauc_hist, prauc_hmm

def fscore_vs_noise(folder='./dummy_results/', savename=None):
    """
    Plots noise vs fscore, by reading results from the given folder
    """

    fscores_random, fscores_rulebased, fscores_hist, fscores_hmm = read_fscores_experiment(folder)
    
    x = ['No Noise', 'Noise Config 1', 'Noise Config 2', 'Noise Config 3', 'Noise Config 4']
    
    plt.plot(x, fscores_random, '-o')
    plt.plot(x, fscores_rulebased, '-o')
    plt.plot(x, fscores_hist, '-o')
    plt.plot(x, fscores_hmm, '-o')
    plt.ylabel('F-Score')
    plt.title('Noise vs F-Score')
    plt.legend(['Random', 'Rule-based', 'Histogram', 'HMM'])
    plt.ylim(0, 1)
    
    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()

def prauc_vs_noise(folder='./dummy_results/', savename=None):
    """
    Plots noise vs precision and recall AUC given folder containing results over multiple noise configurations
    """

    prauc_random, prauc_rulebased, prauc_hist, prauc_hmm = read_prauc_experiment(folder)
    
    x = ['No Noise', 'Noise Config 1', 'Noise Config 2', 'Noise Config 3', 'Noise Config 4']
    
    plt.plot(x, prauc_random, '-o')
    plt.plot(x, prauc_rulebased, '-o')
    plt.plot(x, prauc_hist, '-o')
    plt.plot(x, prauc_hmm, '-o')
    plt.ylabel('PR-AUC')
    plt.title('Noise vs PR-AUC')
    plt.legend(['Random', 'Rule-based', 'Histogram', 'HMM'])
    plt.ylim(0, 1)
    
    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()

def compare_experiments_fscore(folders=['results_high_sample', 'results_high_sample_v3'], labels=['Dirty', 'Clean'], savename=None):
    x = ['No Noise', 'Noise Config 1', 'Noise Config 2', 'Noise Config 3', 'Noise Config 4']
    fscores_random, fscores_rb, fscores_hmm = read_fscores_experiment(folders[0])
    plt.plot(x, fscores_random, '--o', color='dodgerblue')
    plt.plot(x, fscores_rb, '--o', color='orange')
    plt.plot(x, fscores_hmm, '--o', color='green')


    fscores_random, fscores_rb, fscores_hmm = read_fscores_experiment(folders[1])
    plt.plot(x, fscores_random, '-o', color='dodgerblue')
    plt.plot(x, fscores_rb, '-o', color='orange')
    plt.plot(x, fscores_hmm, '-o', color='green')


    plt.title('Noise vs F-Score')
    plt.legend(['Random', 'Rule-based', 'HMM'])

    plt.ylabel('F-Score')
    plt.ylim(0, 1)

    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()

def compare_experiments_prauc(folders=['results_high_sample', 'results_high_sample_v3'], labels=['Dirty', 'Clean'], savename=None):
    x = ['No Noise', 'Noise Config 1', 'Noise Config 2', 'Noise Config 3', 'Noise Config 4']
    praucs_random, praucs_rb, praucs_hmm = read_prauc_experiment(folders[0])
    plt.plot(x, praucs_random, '--o', color='dodgerblue')
    plt.plot(x, praucs_rb, '--o', color='orange')
    plt.plot(x, praucs_hmm, '--o', color='green')


    praucs_random, praucs_rb, praucs_hmm = read_prauc_experiment(folders[1])
    plt.plot(x, praucs_random, '-o', color='dodgerblue')
    plt.plot(x, praucs_rb, '-o', color='orange')
    plt.plot(x, praucs_hmm, '-o', color='green')


    plt.title('Noise vs PR-AUC')
    plt.legend(['Random', 'Rule-based', 'HMM'])

    plt.ylabel('PR-AUC')
    plt.ylim(0, 1)

    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()

def recalc_pr_scores(folder):
    pbar = tqdm([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])
    for sf in pbar:
        pbar.set_description(f'Recalulating results for folder: {sf}')
        result_files = [os.path.join(sf,f) for f in os.listdir(sf) if 'hdf5' in f]
        if len(result_files) != 1:
            raise ValueError(f"Expected one results file (hdf5) in {sf}, but found {len(result_files)}: {result_files}")
        res_fname = result_files[0]
        with open(res_fname, 'rb') as handle:
            result = pickle.load(handle)

        # Read arguments from results and setup dataset
        noise, noise_config, map_index, bbox = result['args']['noise'], result['args']['noise_config'], \
            result['args']['map_index'], result['args']['bbox']
        dataset = SHDataset(noise=noise, noise_config=noise_config)
        G1,_,G2,_ = dataset.read_snapshots(map_index, bbox=bbox)
        gt_labels = groundtruth_labels(G1, G2)

        # Read scores for random method, recompute precision recall, overwrite old results
        scores_rand = result['results']['random']['scores']
        ps_rand, rs_rand, ts_rand, prauc_rand = PRCurve(gt_labels, scores_rand, log_scale=False, savename=os.path.join(sf, 'prcurve_random'))
        PRCurve(gt_labels, scores_rand, log_scale=True, savename=os.path.join(sf, 'prcurve_logscale_random'))
        result['results']['random']['precision'] = ps_rand
        result['results']['random']['recall'] = rs_rand
        result['results']['random']['thresholds'] = ts_rand
        result['results']['random']['pr_auc'] = prauc_rand

        # Read scores for rule-based method, recompute precision recall, overwrite old results
        scores_rb = result['results']['rulebased']['scores']
        ps_rb, rs_rb, ts_rb, prauc_rb = PRCurve(gt_labels, scores_rb, log_scale=False, savename=os.path.join(sf, 'prcurve_rulebased'))
        PRCurve(gt_labels, scores_rb, log_scale=True, savename=os.path.join(sf, 'prcurve_logscale_rulebased'))
        result['results']['rulebased']['precision'] = ps_rb
        result['results']['rulebased']['recall'] = rs_rb
        result['results']['rulebased']['thresholds'] = ts_rb
        result['results']['rulebased']['pr_auc'] = prauc_rb

        # Read scores for HMM method, recompute precision recall, overwrite old results
        scores_hmm = result['results']['hmm']['scores']
        ps_hmm, rs_hmm, ts_hmm, prauc_hmm = PRCurve(gt_labels, scores_hmm, log_scale=False, savename=os.path.join(sf, 'prcurve_hmm'))
        PRCurve(gt_labels, scores_hmm, log_scale=True, savename=os.path.join(sf, 'prcurve_logscale_hmm'))
        result['results']['hmm']['precision'] = ps_hmm
        result['results']['hmm']['recall'] = rs_hmm
        result['results']['hmm']['thresholds'] = ts_hmm
        result['results']['hmm']['pr_auc'] = prauc_hmm

        # Overwrite combined PR curves
        PRCombine(ps=[ps_rand, ps_rb, ps_hmm], rs=[rs_rand, rs_rb, rs_hmm], aucs=[prauc_rand, prauc_rb, prauc_hmm]\
                    ,labels=['Random', 'Rule-based', 'HMM'], log_scale=False, savename=os.path.join(sf, 'prcurve_combined'))
        PRCombine(ps=[ps_rand, ps_rb, ps_hmm], rs=[rs_rand, rs_rb, rs_hmm], aucs=[prauc_rand, prauc_rb, prauc_hmm]\
                    ,labels=['Random', 'Rule-based', 'HMM'], log_scale=True, savename=os.path.join(sf, 'prcurve_logscale_combined'))

        # Resave pickled results
        with open(res_fname, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.close()
    fscore_vs_noise(folder=folder, savename=os.path.join(folder, 'fscore_vs_noise'))
    plt.close()
    prauc_vs_noise(folder=folder, savename=os.path.join(folder, 'prauc_vs_noise'))
    plt.close()

        

# recalc_pr_scores(folder='results_high_sample_10000')
# recalc_pr_scores(folder='results_high_sample_v3')
# recalc_pr_scores(folder='results_high_sample')
# recalc_pr_scores(folder='results_high_sample_coveragepatch')
# recalc_pr_scores(folder='results_low_sample_coveragepatch')


# compare_experiments()
# compare_experiments_fscore(folders=[ 'results_high_sample_10000_tuned', 'results_high_sample_10000'], savename='fscore_tuned_vs_untuned')
# compare_experiments_prauc(folders=['results_high_sample_10000', 'results_high_sample_10000_tuned'], savename='prauc_tuned_vs_untuned')