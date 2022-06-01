import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prcurve
from sklearn.metrics import f1_score, auc
import networkx as nx
import numpy as np
import os
import pickle5 as pickle
from natsort import natsorted
import re

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
    plt.close()
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
    plt.title(f'Precision-Recall Curve (AUC: {round(pr_auc, 2)})', fontsize=18)

    # Save plot if specified
    if savename is not None:
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()
    
    return p, r, ts, pr_auc

def PRCombine(ps, rs, aucs, labels=['Random', 'Rule-based'], log_scale=True, savename=None, figsize=(8,6)):
    """
    Combines the precision and recall curves for multiple methods, plots labels and AUCs
    """
    # Clear current figure before plotting
    plt.clf()
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()

    # Plot log scale if specified
    if log_scale:
        ax = plt.gca()
        ax.set_yscale("log")
    
    # Plot the individual precision and recalls for each method
    for p, r in zip(ps, rs):
        plt.plot(r, p)
        
    # Plot the results
    labels = [f'{label} (auc: {round(auroc, 2)})' for label, auroc in list(zip(labels, aucs))]
    plt.legend(labels)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=18)

    # Save plot if specified
    if savename is not None:
        plt.gcf()
        plt.savefig(f'{savename}.png', bbox_inches='tight')
    else:
        plt.show()

    plt.clf()
    
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
    plt.clf()
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()

    # Plot results
    plt.plot(Chs, fscores, '-o')
    plt.ylabel(y)
    plt.xlabel('Heading Weight (Ch)')
    plt.title(f'Heading Weight vs {y}', fontsize=18)
    plt.ylim(0, 1)
    
    # Save figure if specified
    if savename is not None:
        plt.gcf()
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()

    plt.clf()

def dim_vs_y(fscores, dims, y='F-Score', savename=None, figsize=(10,7)):
    """
    Plots the dim/resolution of the histogram vs some custom variable y for the histogram method ablation experiments
    """
    plt.clf()
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()

    # Plot results
    plt.plot(dims, fscores, '-o')
    plt.ylabel(y)
    plt.xlabel('Histogram Dimension')
    plt.title(f'Histogram Dimension vs {y}', fontsize=18)
    plt.ylim(0, 1)
    
    # Save results if specified
    if savename is not None:
        plt.gcf()
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()
    
    plt.clf()

def x_vs_fscore(x, labels, xlabel=None, folder='./dummy_results/', savename=None, figsize=(10,7), preloaded_results=None):
    """
    Plots some variable vs fscore, by reading results from the given folder
    """
    plt.clf()
    
    # Read the results from the folders
    if folder is not None:
        fscores_random, fscores_rulebased, fscores_hist, fscores_hmm = read_fscores_experiment(folder)
    else:
        fscores_random, fscores_rulebased, fscores_hist, fscores_hmm = preloaded_results
    
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
    plt.title(f'{x} vs F-Score', fontsize=18)
    plt.legend(['Random', 'Rule-based', 'Histogram', 'HMM'])
    plt.ylim(0, 1)
    
    # Save figure if specified
    if savename is not None:
        plt.gcf()
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()

    plt.clf()

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

def x_vs_prauc(x, labels, xlabel=None, folder='./dummy_results/', savename=None, figsize=(10,7), preloaded_results=None):
    """
    Plots some variable vs precision and recall AUC given folder containing results over multiple noise configurations
    """
    plt.clf()
    
    # Read the results from the folders
    if folder is not None:
        prauc_random, prauc_rulebased, prauc_hist, prauc_hmm = read_prauc_experiment(folder)
    else:
        prauc_random, prauc_rulebased, prauc_hist, prauc_hmm = preloaded_results
    
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
    plt.title(f'{x} vs PR-AUC', fontsize=18)
    plt.legend(['Random', 'Rule-based', 'Histogram', 'HMM'])
    plt.ylim(0, 1)
    
    # Save figure if specified
    if savename is not None:
        plt.gcf()
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()

    plt.clf()

def bar_fscore(fscores, labels, savename=None, figsize=(8,6)):
    """
    Plots a sequence of F-Scores in a bar chart for comparing histogram configurations
    """
    plt.clf()
    
    # Initialize plot figure size and style
    plt.figure(figsize=figsize)
    set_plot_style()
    
    # Plot results
    plt.bar(labels, fscores)
    plt.ylabel('F-Score')
    plt.title('Histogram Configuration vs F-Score', fontsize=18)
    plt.ylim(0, 1)
    for i in range(len(labels)):
        plt.text(i-0.1,round(fscores[i],2),round(fscores[i],2))

    # Save results if specified
    if savename is not None:
        plt.gcf()
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()

    plt.clf()

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


    plt.title('Noise vs F-Score', fontsize=18)
    plt.legend(['Random', 'Rule-based', 'HMM'])

    plt.ylabel('F-Score')
    plt.ylim(0, 1)

    if savename is not None:
        plt.gcf()
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()

    plt.clf()

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


    plt.title('Noise vs PR-AUC', fontsize=18)
    plt.legend(['Random', 'Rule-based', 'HMM'])

    plt.ylabel('PR-AUC')
    plt.ylim(0, 1)

    if savename is not None:
        plt.gcf()
        plt.savefig(f'{savename}.png',bbox_inches='tight')
    else:
        plt.show()

    plt.clf()

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

def read_results_seeds(folder_prefix):
    results_folder = './experimental_results'
    results_folders = os.listdir(results_folder)
    results_folders = natsorted([os.path.join(results_folder, f) for f in results_folders if folder_prefix in f])

    results = {}
    for seed_folder in results_folders:
        match = re.findall(r'seed([0-9]+)', seed_folder)
        seed = int(match[0])
        seed_folders = [f for f in os.listdir(seed_folder) if os.path.isdir(os.path.join(seed_folder, f))]

        curr_seed_pickle_fnames = []
        for f in seed_folders:
            curr_folder = os.path.join(seed_folder, f)
            curr_files = [os.path.join(curr_folder, f) for f in os.listdir(curr_folder) if 'hdf5' in f]
            if len(curr_files) > 1:
                raise ValueError(f'More than 1 pickle results file found in {curr_folder}')
            elif len(curr_files) < 1:
                raise ValueError(f'No pickle files found in  {curr_folder}')
            pickle_fname = curr_files[0]
            with open(pickle_fname, 'rb') as handle:
                result = pickle.load(handle)
            
            curr_seed_pickle_fnames.append({
                'pickle_fname': pickle_fname,
                'results': result
            })

        results[seed] = curr_seed_pickle_fnames

    return results

def plot_results(folder_prefix, x, xlabels, xlabel):
    # First, collect all the results with the given folder prefix
    results = read_results_seeds(folder_prefix)
    
    # Create empty numpy arrays for every method and metric for all seeds in shape (num_seeds, num_experiments)
    results_shape = (len(results), len(results[list(results.keys())[0]]))
    fscores_random, praucs_random = np.zeros(results_shape), np.zeros(results_shape)
    fscores_rb, praucs_rb = np.zeros(results_shape), np.zeros(results_shape)
    fscores_hist, praucs_hist = np.zeros(results_shape), np.zeros(results_shape)
    fscores_hmm, praucs_hmm = np.zeros(results_shape), np.zeros(results_shape)
    
    # Collect prauc and fscores for each method into numpy arrays
    for i, seed in enumerate(results.keys()):
        res = results[seed]
        for j, r in enumerate(res):
            fscores_random[i,j] = r['results']['results']['random']['fscore']
            praucs_random[i,j] = r['results']['results']['random']['pr_auc']
            fscores_rb[i,j] = r['results']['results']['rulebased']['fscore']
            praucs_rb[i,j] = r['results']['results']['rulebased']['pr_auc']
            fscores_hist[i,j] = r['results']['results']['histogram']['fscore']
            praucs_hist[i,j] = r['results']['results']['histogram']['pr_auc']
            fscores_hmm[i,j] = r['results']['results']['hmm']['fscore']
            praucs_hmm[i,j] = r['results']['results']['hmm']['pr_auc']
    
    # Take average over the seeds
    fscores_random = np.mean(fscores_random, axis=0)
    praucs_random = np.mean(praucs_random, axis=0)
    fscores_rb = np.mean(fscores_rb, axis=0)
    praucs_rb = np.mean(praucs_rb, axis=0)
    fscores_hist = np.mean(fscores_hist, axis=0)
    praucs_hist = np.mean(praucs_hist, axis=0)
    fscores_hmm = np.mean(fscores_hmm, axis=0)
    praucs_hmm = np.mean(praucs_hmm, axis=0)

    # Plot the results
    x_vs_fscore(x='Noise', labels=xlabels, xlabel=xlabel,
                folder=None, 
                savename=f'./figures/{x.lower()}_vs_fscore',
                preloaded_results=(fscores_random, fscores_rb, fscores_hist, fscores_hmm))
    x_vs_prauc(x='Noise', labels=xlabels, xlabel=xlabel,
                folder=None, 
                savename=f'./figures/{x.lower()}_vs_prauc',
                preloaded_results=(praucs_random, praucs_rb, praucs_hist, praucs_hmm))

# from SHDataset import measure_noise
# noise_in_meters = measure_noise()
# x_labels = [f"{round(noise_conf['meters'],1)}m"  for noise_conf in noise_in_meters]
# x_labels = ['No Noise', *x_labels]
# plot_results('results_noise_exp_seeds_test', x='Noise', xlabels=x_labels)