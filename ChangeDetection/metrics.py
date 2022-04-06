import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prcurve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, auc, roc_curve, roc_auc_score
import networkx as nx
import numpy as np

def FScore(gt_labels, pred_scores):
    f1 = f1_score(list(gt_labels.values()), list(pred_scores.values()), average='binary')
    return f1

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
    
def PRCurve(gt_labels, pred_scores, log_scale=True, norm=False):
    if log_scale:
        ax = plt.gca()
        ax.set_yscale("log")
    if norm:
        gt_scores = normalize(list(gt_labels.values()))
        pred_scores = normalize(list(pred_scores.values()))
    else:
        gt_scores = np.array(list(gt_labels.values()))
        pred_scores = np.array(list(pred_scores.values()))

    p, r, ts = prcurve(gt_scores, pred_scores, pos_label=1.0)
    
    pr_auc = auc(r, p)


    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC: {round(pr_auc, 2)})')
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



# def PRCurveManual(gt_labels, pred_scores, log_scale=True, norm=False):
#     if log_scale:
#         ax = plt.gca()
#         ax.set_yscale("log")
#     if norm:
#         gt_scores = normalize(list(gt_labels.values()))
#         pred_scores = normalize(list(pred_scores.values()))
#     else:
#         gt_scores = np.array(list(gt_labels.values()))
#         pred_scores = np.array(list(pred_scores.values()))

#     ts = np.linspace(0, 1.1, num=10)
#     ps, rs = [], []
#     for t in ts:
#         pred_labels = np.where(pred_scores > t, np.ones_like(pred_scores), np.zeros_like(pred_scores))
#         # pred_labels = np.where(pred_scores <= t, np.zeros_like(pred_scores), np.ones_like(pred_scores))
#         p = precision_score(gt_scores, pred_labels)
#         r = recall_score(gt_scores, pred_labels)
#         ps.append(p)
#         rs.append(r)

#     ps, rs = np.array(ps), np.array(rs)
#     print(ps)
#     print(rs)
#     print(ts)

#     plt.plot(rs, ps)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.show()

#     return ps, rs, ts
    





def PRCombine(ps, rs, aucs, labels=['Random', 'Rule-based'], log_scale=True):
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