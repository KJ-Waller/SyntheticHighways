import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prcurve
from sklearn.metrics import f1_score, auc
import networkx as nx
import numpy as np
import os
import pickle5 as pickle
from natsort import natsorted

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
    
def PRCurve(gt_labels, pred_scores, log_scale=True, norm=False, savename=None, figsize=(8,6)):
    """
    Plots the precision recall curve given ground truth labels and predicted scores
    """
    plt.clf()

    # Make sure labels are aligned according to order of edges in gt_labels
    predictions = {k: np.float64(pred_scores[k]).item() for k in gt_labels}

    # Plot log scale if specified
    if log_scale:
        ax = plt.gca()
        ax.set_yscale("log")
    if norm:
        gt_scores = normalize(list(gt_labels.values()))
        predictions = normalize(list(predictions.values()))
    else:
        gt_scores = np.array(list(gt_labels.values()))
        predictions = np.array(list(predictions.values()))
        
    # Get precision, recall and thresholds
    p, r, ts = prcurve(gt_scores, predictions, pos_label=1.0)
    
    # Get the area-under-curve for PR (PR-AUC)
    pr_auc = auc(r, p)

    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()

    # Plot the results
    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC: {round(pr_auc, 2)})')

    # Save plot if specified
    if savename is not None:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return p, r, ts, pr_auc

def PRCombine(ps, rs, aucs, labels=['Random', 'Rule-based'], log_scale=True, savename=None):
    """
    Combines the precision and recall curves for multiple methods, plots labels and AUCs
    """
    plt.clf()

    # Plot log scale if specified
    if log_scale:
        ax = plt.gca()
        ax.set_yscale("log")
    
    # Plot the individual precision and recalls for each method
    for p, r in zip(ps, rs):
        plt.plot(r, p)
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()
        
    # Plot the results
    labels = [f'{label} (auc: {round(auroc, 2)})' for label, auroc in list(zip(labels, aucs))]
    plt.legend(labels)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    # Save plot if specified
    if savename is not None:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
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
    """
    Reads the fscores from an experiment run using the "exp_all_methods.py" script, which are the three main experiments,
    namely sparsity, tfreq and noise experiments. It reads the folders and collects the fscores from each experiment for all methods
    """

    # Sort the folder in correct order of results
    folders = [f for f in natsorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder,f))]

    # Make sure "nonoise" folder is first in the folders list
    nonoise_check = [True for f in folders if 'nonoise' in f]
    if len(nonoise_check) == 1 and nonoise_check[0]:
        folders = [folders[-1], *folders[0:-1]]

    # Get results for each folder
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
    
    # Get fscores for each folder
    fscores_random = []
    fscores_rulebased = []
    fscores_hist = []
    fscores_hmm = []
    for result in results:
        fscores_random.append(result['results']['random']['fscore'])
        fscores_rulebased.append(result['results']['rulebased']['fscore'])
        fscores_hist.append(result['results']['histogram']['fscore'])
        fscores_hmm.append(result['results']['hmm']['fscore'])

    # Return fscores
    return fscores_random, fscores_rulebased, fscores_hist, fscores_hmm

def read_prauc_experiment(folder):
    """
    Reads the PR-AUC from an experiment run using the "exp_all_methods.py" script, which are the three main experiments,
    namely sparsity, tfreq and noise experiments. It reads the folders and collects the PR-AUCs from each experiment for all methods
    """

    # Sort the folder in correct order of results
    folders = [f for f in natsorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder,f))]

    # Make sure "nonoise" folder is first in the folders list
    nonoise_check = [True for f in folders if 'nonoise' in f]
    if len(nonoise_check) == 1 and nonoise_check[0]:
        folders = [folders[-1], *folders[0:-1]]

    # Get results for each folder
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
    
    
    # Get PR-AUCs for each folder
    prauc_random = []
    prauc_rulebased = []
    prauc_hist = []
    prauc_hmm = []
    for result in results:
        prauc_random.append(result['results']['random']['pr_auc'])
        prauc_rulebased.append(result['results']['rulebased']['pr_auc'])
        prauc_hist.append(result['results']['histogram']['pr_auc'])
        prauc_hmm.append(result['results']['hmm']['pr_auc'])

    # Return PR-AUCs
    return prauc_random, prauc_rulebased, prauc_hist, prauc_hmm

def Ch_vs_y(fscores, Chs, y='F-Score', savename=None, figsize=(8,6)):
    """
    Plots the Ch (heading parameter) vs some custom variable y for the Rule-based method ablation experiments
    """
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()

    # Plot results
    plt.figure(figsize=figsize)
    plt.plot(Chs, fscores, '-o')
    plt.ylabel(y)
    plt.xlabel('Heading Weight (Ch)')
    plt.title(f'Heading Weight vs {y}')
    plt.ylim(0, 1)
    
    # Save figure if specified
    if savename is not None:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def dim_vs_y(fscores, dims, y='F-Score', savename=None, figsize=(10,7)):
    """
    Plots the dim/resolution of the histogram vs some custom variable y for the histogram method ablation experiments
    """
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()

    # Plot results
    plt.figure(figsize=figsize)
    plt.plot(dims, fscores, '-o')
    plt.ylabel(y)
    plt.xlabel('Histogram Dimension')
    plt.title(f'Histogram Dimension vs {y}')
    plt.ylim(0, 1)
    
    # Save results if specified
    if savename is not None:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def x_vs_fscore(x, labels, xlabel=None, folder='./dummy_results/', savename=None, figsize=(10,7)):
    """
    Plots some variable vs fscore, by reading results from the given folder
    """
    
    # Read the results from the folders
    fscores_random, fscores_rulebased, fscores_hist, fscores_hmm = read_fscores_experiment(folder)
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()

    # Plot the 4 different methods
    plt.plot(labels, fscores_random, '-o')
    plt.plot(labels, fscores_rulebased, '-o')
    plt.plot(labels, fscores_hist, '-o')
    plt.plot(labels, fscores_hmm, '-o')
    
    # Set axes labels, title, legend and y limit
    plt.ylabel('F-Score')
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.title(f'{x} vs F-Score')
    plt.legend(['Random', 'Rule-based', 'Histogram', 'HMM'])
    plt.ylim(0, 1)
    
    # Save figure if specified
    if savename is not None:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def set_plot_style():
    # use a gray background
    ax = plt.axes(facecolor='#E6E6E6')
    ax.set_axisbelow(True)

    # draw solid white grid lines
    plt.grid(color='w', linestyle='solid')

    # hide axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # hide top and right ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # lighten ticks and labels
    ax.tick_params(colors='gray', direction='out')
    for tick in ax.get_xticklabels():
        tick.set_color('gray')
    for tick in ax.get_yticklabels():
        tick.set_color('gray')

def x_vs_prauc(x, labels, xlabel=None, folder='./dummy_results/', savename=None, figsize=(10,7)):
    """
    Plots some variable vs precision and recall AUC given folder containing results over multiple noise configurations
    """
    
    # Read the results from the folders
    prauc_random, prauc_rulebased, prauc_hist, prauc_hmm = read_prauc_experiment(folder)
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()
    
    # Plot the 4 different methods
    plt.plot(labels, prauc_random, '-o')
    plt.plot(labels, prauc_rulebased, '-o')
    plt.plot(labels, prauc_hist, '-o')
    plt.plot(labels, prauc_hmm, '-o')

    # Set axes labels, title, legend and y limit
    plt.ylabel('PR-AUC')
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.title(f'{x} vs PR-AUC')
    plt.legend(['Random', 'Rule-based', 'Histogram', 'HMM'])
    plt.ylim(0, 1)
    
    # Save figure if specified
    if savename is not None:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def bar_fscore(fscores, labels, savename=None, figsize=(8,6)):
    """
    Plots a sequence of F-Scores in a bar chart for comparing histogram configurations
    """
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()
    
    # Plot results
    plt.bar(labels, fscores)
    plt.ylabel('F-Score')
    plt.title('Histogram Configuration vs F-Score')
    plt.ylim(0, 1)
    for i in range(len(labels)):
        plt.text(i-0.1,round(fscores[i],2),round(fscores[i],2))

    # Save results if specified
    if savename is not None:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
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
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
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
        plt.savefig(f'{savename}.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def replot_tfreq_results(folder, steps=6, step_size=3):
    resample_traj_steps = np.arange(1,(steps*step_size)+1, step_size)

    exp_folder = os.path.join('./experimental_results', folder)
    if not os.path.exists(exp_folder):
        raise ValueError(f'Folder {exp_folder} not found')
    # Save plots
    x_vs_fscore(x='Trace Frequency', labels=resample_traj_steps, xlabel='Trace Frequency',
                folder=exp_folder,
                savename=os.path.join(exp_folder, "tfreq_vs_fscore"), figsize=(8,6))
    plt.close()
    x_vs_prauc(x='Trace Frequency', labels=resample_traj_steps, xlabel='Trace Frequency',
                folder=exp_folder,
                savename=os.path.join(exp_folder, "tfreq_vs_prauc"), figsize=(8,6))

def replot_tfreq_exps():
    folders = [f for f in os.listdir('experimental_results') if 'tfreq' in f]
    for f in folders:
        if 'moresteps' in f:
            replot_tfreq_results(f, steps=6, step_size=3)
        else:
            replot_tfreq_results(f, steps=5, step_size=1)

replot_tfreq_exps()