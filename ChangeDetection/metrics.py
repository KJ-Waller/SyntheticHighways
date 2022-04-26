from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prcurve
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_curve, roc_auc_score
import networkx as nx
import numpy as np
import os
import pickle5 as pickle

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
    predictions = {k: int(pred_scores[k] == 0) for k in gt_labels}

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
    fscores_hmm = []
    for result in results:
        fscores_random.append(result['results']['random']['fscore'])
        fscores_rulebased.append(result['results']['rulebased']['fscore'])
        fscores_hmm.append(result['results']['hmm']['fscore'])
    return fscores_random, fscores_rulebased, fscores_hmm

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
    prauc_hmm = []
    for result in results:
        prauc_random.append(result['results']['random']['pr_auc'])
        prauc_rulebased.append(result['results']['rulebased']['pr_auc'])
        prauc_hmm.append(result['results']['hmm']['pr_auc'])

    return prauc_random, prauc_rulebased, prauc_hmm

def fscore_vs_noise(folder='./dummy_results/', savename=None):
    """
    Plots noise vs fscore, by reading results from the given folder
    """

    fscores_random, fscores_rulebased, fscores_hmm = read_fscores_experiment(folder)
    
    x = ['No Noise', 'Noise Config 1', 'Noise Config 2', 'Noise Config 3', 'Noise Config 4']
    
    plt.plot(x, fscores_random, '-o')
    plt.plot(x, fscores_rulebased, '-o')
    plt.plot(x, fscores_hmm, '-o')
    plt.ylabel('F-Score')
    plt.title('Noise vs F-Score')
    plt.legend(['Random', 'Rule-based', 'HMM'])
    
    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()

def prauc_vs_noise(folder='./dummy_results/', savename=None):
    """
    Plots noise vs precision and recall AUC given folder containing results over multiple noise configurations
    """

    prauc_random, prauc_rulebased, prauc_hmm = read_prauc_experiment(folder)
    
    x = ['No Noise', 'Noise Config 1', 'Noise Config 2', 'Noise Config 3', 'Noise Config 4']
    
    plt.plot(x, prauc_random, '-o')
    plt.plot(x, prauc_rulebased, '-o')
    plt.plot(x, prauc_hmm, '-o')
    plt.ylabel('PR-AUC')
    plt.title('Noise vs PR-AUC')
    plt.legend(['Random', 'Rule-based', 'HMM'])
    
    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()

def compare_experiments(folders=['results_high_sample', 'results_high_sample_v3'], labels=['Dirty', 'Clean'], savename=None):
    x = ['No Noise', 'Noise Config 1', 'Noise Config 2', 'Noise Config 3', 'Noise Config 4']
    fscores_random, fscores_rb, fscores_hmm = read_fscores_experiment(folders[0])
    plt.plot(x, fscores_random, '-o', color='dodgerblue')
    plt.plot(x, fscores_rb, '-o', color='orange')
    plt.plot(x, fscores_hmm, '-o', color='green')


    fscores_random, fscores_rb, fscores_hmm = read_fscores_experiment(folders[1])
    plt.plot(x, fscores_random, '--o', color='dodgerblue')
    plt.plot(x, fscores_rb, '--o', color='orange')
    plt.plot(x, fscores_hmm, '--o', color='green')


    plt.legend(['Random', 'Rule-based', 'HMM'])

    plt.ylabel('F-Score')


    
    if savename is not None:
        plt.savefig(f'{savename}.png')
    else:
        plt.show()



compare_experiments()
# compare_experiments(savename='fscore_cleaned_vs_dirty')