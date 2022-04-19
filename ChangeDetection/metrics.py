import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prcurve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, auc, roc_curve, roc_auc_score
import networkx as nx
import numpy as np
import os
import pickle5 as pickle

def fscore(gt_labels, pred_labels):
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
    scores = nx.get_edge_attributes(G_pred, 'weight')
    sorted_scores = {edge: scores[edge] for edge in sorted(scores)}
    return sorted_scores
    
def PRCurve(gt_labels, pred_scores, log_scale=True, norm=False, savename=None):
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

def ROCCurve(gt_labels, pred_scores):
    gt_scores = np.array(list(gt_labels.values()))
    pred_scores = np.array(list(pred_scores.values()))
    fpr, tpr, ts = roc_curve(gt_scores, pred_scores)
    auroc = roc_auc_score(gt_scores, pred_scores)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUROC: {round(auroc, 2)})')
    plt.show()

    return tpr, fpr, auroc

def ROCCombine(fprs, tprs, aurocs, labels):
    for fpr, tpr in zip(fprs, tprs):
        plt.plot(fpr, tpr)

    labels = [f'{label} (auc: {round(auroc, 2)})' for label, auroc in list(zip(labels, aurocs))]
    plt.legend(labels)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def PRCombine(ps, rs, aucs, labels=['Random', 'Rule-based'], log_scale=True, savename=None):
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

def fscore_vs_noise(folder='./dummy_results/', savename=None):
    folders = sorted(os.listdir(folder))
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

fscore_vs_noise('./results/', savename='./results/fscore_vs_noise')