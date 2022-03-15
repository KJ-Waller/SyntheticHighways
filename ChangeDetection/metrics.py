import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prcurve
import networkx as nx

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
    return scores
    
def PRCurve(gt_labels, pred_scores):
    p, r, ts = prcurve(list(gt_labels.values()), list(pred_scores.values()))
    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
    
    return p, r, ts

def PRCombine(ps, rs, labels=['Random', 'Rule-based']):
    
    for p, r in zip(ps, rs):
        plt.plot(r, p)
        
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