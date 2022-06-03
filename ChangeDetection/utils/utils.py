import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
from PIL import Image
from tqdm import tqdm
import pyproj
geodesic = pyproj.Geod(ellps='WGS84')

def get_heading(x,y):
    """
    Calculates the heading given a 2D velocity vector defined by x,y
    """

    # Initialize north and direction vector which to calculate angle between
    v = np.array([x,y])
    north = np.array([1.0,0])

    # Calculate angle of vector and north
    v_ang = np.arctan2(*v[::-1])
    north_ang = np.arctan2(*north[::-1])

    # Calculate clockwise angle (heading) between the vectors
    heading = np.rad2deg((v_ang - north_ang) % (2 * np.pi))

    return heading

def traj_to_nxgraph(T, stringify_id=True, color = 'red'):
    """
    Converts a set of trajectories to a NetworkX graph

    -------
    Params
    -------
    T : list
        List of trajectories
    stringify_id : boolean
        Sets node ids for trajectories as strings rather than numbers. Useful when
        combining trajectories and map into single NetworkX graph to avoid duplicate node
        and edge ids
    color : string
        What color attribute to give trajectories, for visualizing trajectories
    """
    G = nx.Graph()


    key = 0
    for traj in T:
        prev_node_id = None
        for point in traj:
            node_id = str(key) if stringify_id else key

            G.add_node(
                node_id, 
                label=node_id,
                lat=point['lat'],
                lon=point['lon'],
                color=color
            )
            if prev_node_id is not None:
                G.add_edge(
                    prev_node_id, node_id,
                    color=color
                )
            prev_node_id = node_id
            key += 1

    return G

def snapshot_to_nxgraph(G, T):
    """
    Converts a single snapshot (a map G + a set of trajectories T) to a network x graph,
    useful for visualization with NetworkX

    -------
    Params
    -------
    G : NetworkX Graph
        The graph of the map
    T : list
        List of trajectories

    -------
    Returns
    -------
    G_T : NetworkX Graph
        NetworkX graph containing the map and trajectories
    """
    T = traj_to_nxgraph(T)
    G_T = combine_graphs(G,T)
    return G_T

def combine_graphs(G1,G2):
    """
    Combines two NetworkX graphs into a single one
    """
    G = nx.compose(G1,G2)
    return G

def save_hist(hist, savename):
    plt.imsave(f'{savename}.png', np.rot90(hist))

def plot_graph(G, figsize=(8,8), show_nodes=False, show_labels=False, 
                T_node_size=5, G_node_size=5, T_edge_width=1.0, G_edge_width=1.0, 
                removed_road_edge_width=None, use_weights=False, traj_alpha=1.0, 
                show_img=True, title=None, fontsize=5, equal_axis_ratio=False, 
                zoom_on_traj=False, show_legend=True, savename=None):
    """
    Visualizes the NetworkX graph

    -------
    Params
    -------
    G : NetworkX Graph
        The graph to visualize
    figsize : tuple of ints
        What size to plot the figure
    show_nodes : boolean
        Whether to show nodes of trajectories and map or not
    show_labels : boolean
        Whether to show node ids or not
    T_node_size : float
        How big to plot nodes for trajectories (show nodes must be set to True)
    G_node_size : float
        How big to plot nodes for the map (show nodes must be set to True)
    T_edge_width : float
        How wide to plot the edges for the trajectories
    G_edge_width : float
        How wide to plot the edges for the map
    removed_road_edge_width : float
        How wide to plot the edges of roads that are removed
    use_weights : boolean
        Whether to use the weights to color the edges to make a heatmap
    traj_alpha : float range(0,1)
        Alpha/transparency of trajectories in visualization
    show_img : boolean
        Whether to show image or not
    title : str
        What title to plot with the figure. None implies no title
    fontsize : float
        What fontsize to use for labels, if enabled
    equal_axis_ratio : boolean
        Whether to plot x,y axes with equal aspect ratios
    zoom_on_traj : boolean
        Whether to zoom in on the trajectories or to show the entire map
    show_legend : boolean
        Whether to show a legend for map and trajectories
    savename : string
        Location and filename to save plot to. If set to None, it will not save anything
    """

    if G.size() == 0:
        print("Graph is empty")
        return

    # Change map to tab:blue
    blue_map_colors = {(edge[0], edge[1]): 'tab:blue' for edge in G.edges(data=True) if edge[2]['color'] == 'blue'}
    nx.set_edge_attributes(G, blue_map_colors, name='color')

    # Create the lines for the legend
    map_line = mlines.Line2D([], [], color='tab:blue', marker='.' if show_nodes else '',
                          markersize=G_node_size, linewidth=G_edge_width, label='Map')
    map_removed_road_line = mlines.Line2D([], [], color='magenta', marker='',
                          markersize=G_node_size, linewidth=removed_road_edge_width, label='Removed Edge')
    traj_line = mlines.Line2D([], [], color='red', marker='.' if show_nodes else '',
                          markersize=T_node_size, linewidth=T_edge_width, alpha=traj_alpha, label='Trajectory')
    mismatched_line = mlines.Line2D([], [], color='orange', marker='',
                          markersize=G_node_size, linewidth=T_edge_width, label='Mismatched Edge')
    path_line = mlines.Line2D([], [], color='green', marker='',
                          markersize=G_node_size, linewidth=T_edge_width, label='Path')
    
    # Get list of green and orange edges (representing path and mismatched edges)
    # for checking if path and mismatched edges are to be plotted
    e_colors = nx.get_edge_attributes(G, name='color')
    green_edges = [col for col in e_colors.values() if col == 'green']
    orange_edges = [col for col in e_colors.values() if col == 'orange']
    
    # Fetch the node positions and colors
    node_lats = nx.get_node_attributes(G, 'lat')
    node_lons = nx.get_node_attributes(G, 'lon')
    node_pos = {node: (node_lons[node], node_lats[node]) for node in node_lats.keys()}
    node_colors = list(nx.get_node_attributes(G, 'color').values())
    nodes = list(node_pos.keys())
    
    # Set figure size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the nodes if enabled
    if show_nodes:
        nodecols = nx.get_node_attributes(G, 'color')
        node_sizes = [T_node_size if nodecols[node] == 'red' else G_node_size for node in nodes]
        nx.draw_networkx_nodes(G, node_pos, nodelist=nodes, node_color=node_colors, node_size=node_sizes, ax=ax)
    if show_labels:
        node_labels = {node: str(node) for node in G.nodes}
        nx.draw_networkx_labels(G, node_pos, labels=node_labels, font_size=fontsize, ax=ax)
        
    # Plot edges using weights if enabled
    if use_weights:
        # Get edge weights
        edges, edge_weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        
        # Split weights into weights for map and trajectories
        map_weights = [(edge, edge_weights[i]) for i, edge in enumerate(edges) if type(edge[0]) == int or type(edge[0]) == float]
        traj_weights = [(edge, edge_weights[i]) for i, edge in enumerate(edges) if type(edge[0]) == str]
        
        # If both map and trajectories are being plotted, plot in blue and red respectively
        if len(traj_weights) != 0 and len(map_weights) != 0:
            edges_map, eweights_map = zip(*map_weights)
            edges_traj, eweights_traj = zip(*traj_weights)
            nx.draw_networkx_edges(G, node_pos, edgelist=edges_map, width=G_edge_width, edge_color=eweights_map, edge_cmap=plt.cm.viridis, ax=ax)
            nx.draw_networkx_edges(G, node_pos, edgelist=edges_traj, width=T_edge_width, edge_color=eweights_traj, edge_cmap=plt.cm.Reds, alpha=traj_alpha, ax=ax)
        # If only map available, plot in blue only
        elif len(traj_weights) == 0 and len(map_weights) != 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin = np.min(edge_weights), vmax=np.max(edge_weights)))
            nx.draw_networkx_edges(G, node_pos, width=G_edge_width, edge_color=edge_weights, edge_cmap=plt.cm.viridis, ax=ax)
            plt.colorbar(sm)

        # if only trajectories available, plot in red
        elif len(traj_weights) != 0 and len(map_weights) == 0:
            nx.draw_networkx_edges(G, node_pos, width=T_edge_width, edge_color=edge_weights, edge_cmap=plt.cm.Reds, alpha=traj_alpha, ax=ax)
            if show_legend:
                plt.legend(handles=[traj_line])
        
    # Otherwise, plot edges with regular color
    else:
        edges, edge_colors = zip(*nx.get_edge_attributes(G, 'color').items())
        map_colors = [(edge, color) for (edge, color) in zip(edges, edge_colors) if type(edge[0]) == int or type(edge[0]) == float]
        traj_colors = [(edge, color) for (edge, color) in zip(edges, edge_colors) if type(edge[0]) == str]
        
        # If both map and trajectories are being plotted, plot in blue and red respectively
        if len(traj_colors) != 0 and len(map_colors) != 0:
            edges_map, colors_map = zip(*map_colors)
            edges_traj, colors_traj = zip(*traj_colors)
            nx.draw_networkx_edges(G, node_pos, edgelist=edges_map, width=G_edge_width, edge_color=colors_map, edge_cmap=plt.cm.Blues, ax=ax)
            nx.draw_networkx_edges(G, node_pos, edgelist=edges_traj, width=T_edge_width, edge_color=colors_traj, edge_cmap=plt.cm.Reds, alpha=traj_alpha, ax=ax)
            if show_legend:
                if len(green_edges) > 0 and len(orange_edges) > 0:
                    plt.legend(handles=[map_line, traj_line, map_removed_road_line, path_line, mismatched_line])
                else:
                    plt.legend(handles=[map_line, traj_line])
        # If only map available, and we don't want to highlight removed roads, plot in blue only
        elif len(traj_colors) == 0 and len(map_colors) != 0 and removed_road_edge_width is None:
            nx.draw_networkx_edges(G, node_pos, width=G_edge_width, edge_color=edge_colors, edge_cmap=plt.cm.Blues, ax=ax)
            if show_legend:
                if len(green_edges) > 0 and len(orange_edges) > 0:
                    plt.legend(handles=[map_line, traj_line, map_removed_road_line, path_line, mismatched_line])
                else:
                    plt.legend(handles=[map_line])
        # If only map available, and we want to highlight removed roads
        elif len(traj_colors) == 0 and len(map_colors) != 0 and removed_road_edge_width is not None:
            # Get edgelists and color lists for remaining and removed roads
            G_colors = [(edge, color) for (edge, color) in zip(edges, edge_colors) if color == 'tab:blue']
            G_diff_colors = [(edge, color) for (edge, color) in zip(edges, edge_colors) if color == 'magenta']
            traj_colors = [(edge, color) for (edge, color) in zip(edges, edge_colors) if type(edge[0]) == str]
            G_edgelist, G_colors = zip(*G_colors)
            G_diff_edgelist, G_diff_colors = zip(*G_diff_colors)
            
            # Plot
            nx.draw_networkx_edges(G, node_pos, edgelist=G_edgelist, width=G_edge_width, edge_color=G_colors, edge_cmap=plt.cm.Blues, ax=ax)
            nx.draw_networkx_edges(G, node_pos, edgelist=G_diff_edgelist, width=removed_road_edge_width, edge_color=G_diff_colors, edge_cmap=plt.cm.Reds, ax=ax)
            if show_legend:
                plt.legend(handles=[map_line, map_removed_road_line])
        # if only trajectories available, plot in red
        elif len(traj_colors) != 0 and len(map_colors) == 0:
            nx.draw_networkx_edges(G, node_pos, width=T_edge_width, edge_color=edge_colors, edge_cmap=plt.cm.Reds, alpha=traj_alpha, ax=ax)
            if show_legend:
                plt.legend(handles=[traj_line])

    # Add axis labels for lat lon
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    # Adjust xlim and ylim if zoom_on_traj is enabled, meaning we should zoom in on the trajectory
    if zoom_on_traj:
        offset = 1e-3
        t_lats, t_lons = zip(*[(node[1]['lat'], node[1]['lon']) for node in G.nodes(data=True) if node[1]['color'] == 'red'])
        lat_bounds = (np.min(t_lats)-offset, np.max(t_lats)+offset)
        lon_bounds = (np.min(t_lons)-offset, np.max(t_lons)+offset)
        ax.set_xlim(*lon_bounds)
        ax.set_ylim(*lat_bounds)

    # Make it so that the xticks and yticks are at the same interval
    if equal_axis_ratio:
        ax.axis('equal')

    # Plot title if specified
    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')

    # Plot the figure (but don't show yet)
    plt.plot()

    # Show only if enabled
    if show_img:
        plt.show()

    # Save figure if savename is specified
    if savename is not None:
        fig1 = plt.gcf()
        fig1.savefig(f'{savename}.png', bbox_inches = "tight")
    
def compare_snapshots(G1, G2):
    """
    Compares the maps from snapshots G1 and G2, returns the difference between them
    """
    
    # Get edges of both graphs as sets
    G1_edges = set(G1.edges)
    G2_edges = set(G2.edges)
    
    # Create graph highlighting edges in G1 not in G2
    G1_diff_edges = G1_edges - G2_edges
    G1_diff_edges = [(edge) for edge in G1.edges(data=True) if (edge[0], edge[1]) in G1_diff_edges]
    G1_d = nx.create_empty_copy(G1)
    G1_d.add_edges_from(G1_diff_edges)
    nx.set_edge_attributes(G1_d, 'magenta', name='color')
    G12_d = nx.compose(G2, G1_d)
    
    # Create graph highlighting edges in G2 not in G1
    G2_diff_edges = G2_edges - G1_edges
    G2_diff_edges = [(edge) for edge in G2.edges(data=True) if (edge[0], edge[1]) in G2_diff_edges]
    G2_d = nx.create_empty_copy(G2)
    G2_d.add_edges_from(G2_diff_edges)
    nx.set_edge_attributes(G2_d, 'magenta', name='color')
    G21_d = nx.compose(G1, G2_d)
    
    return G1_d, G12_d, G2_d, G21_d

def cart_to_wgs84(ref_coord, x, y, scale_factor=1):
    """
    Converts cartesian coordinates to wgs84 coordinates relative to given reference coordinate
    """
    x,y = np.array(x)*scale_factor, np.array(y)*scale_factor
    ref_lat, ref_lon = ref_coord[0], ref_coord[1]
    
    lat = ref_lat + (180.0/np.pi)*(y/6378137)
    lon = ref_lon + (180.0/np.pi)*(x/6378137)/np.cos(np.deg2rad(ref_lat))

    axis = 1 if x.size > 1 else 0
    return np.stack((lat, lon),axis=axis)

def filter_bbox_snapshots(G1,T1,G2,T2, bbox, map_offset=0.0000):
    """
    Takes a pair of snapshots, and filters out everything not in the given bounding box
    """
    lat_min, lat_max, lon_min, lon_max = bbox

    # Create copies of the graphs
    G1, G2 = G1.copy(), G2.copy()

    # For all nodes in graph, filter them out if they're not in the bounding box
    for node in G1.copy().nodes(data=True):
        if node[1]['lat'] < (lat_min-map_offset) or node[1]['lat'] > (lat_max+map_offset) \
        or node[1]['lon'] < (lon_min-(3*map_offset)) or node[1]['lon'] > (lon_max+(3*map_offset)):
            G1.remove_node(node[0])
    for node in G2.copy().nodes(data=True):
        if node[1]['lat'] < (lat_min-map_offset) or node[1]['lat'] > (lat_max+map_offset) \
        or node[1]['lon'] < (lon_min-(3*map_offset)) or node[1]['lon'] > (lon_max+(3*map_offset)):
            G2.remove_node(node[0])

    # Create new T1
    T1_new = []

    # For all trajectories, filter them out if they're not in bounding box
    # Split them up if part of trajectory is outside of bounding box
    for t in T1['T']:
        t_new = []
        for p in t:
            if p['lat'] >= lat_min and p['lat'] <= lat_max and p['lon'] >= lon_min and p['lon'] <= lon_max:
                t_new.append(p)
            else:
                if len(t_new) > 0:
                    T1_new.append(t_new)
                t_new = []
        if len(t_new) > 0:
            T1_new.append(t_new)

    # Do the same for T2
    T2_new = []
    for t in T2['T']:
        t_new = []
        for p in t:
            if p['lat'] >= lat_min and p['lat'] <= lat_max and p['lon'] >= lon_min and p['lon'] <= lon_max:
                t_new.append(p)
            else:
                if len(t_new) > 0:
                    T2_new.append(t_new)
                t_new = []
        if len(t_new) > 0:
            T2_new.append(t_new)

    
    return G1, {'T': T1_new, 'P': T1['P']}, G2, {'T': T2_new, 'P': T2['P']}


def save_gif(folder, img_name, savename):
    """
    Combines images from an experiment into a GIF (only experiments which use the exp_all_methods.py was)
    """
    folders = [os.path.join(folder, f) for f in os.listdir(folder)]
    folders = [folder for folder in folders if os.path.isdir(folder)]
    all_files = [os.listdir(folder) for folder in folders]
    files = [file for files in all_files for file in files if img_name in file]
    files = [os.path.join(f, file) for f, file in list(zip(folders, files))]
    
    imgs = [Image.open(file) for file in files]
    imgs[0].save(f'{savename}.gif', save_all=True, append_images=imgs[1:], duration=500, loop=0)

def save_histres_gif(folder, savename):
    """
    Combines the histograms from the histogram resolution experiments into a GIF
    """
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file[:4] == 'hist']
    
    imgs = [Image.open(file) for file in files]
    
    imgs = [img.resize(imgs[-1].size, Image.ANTIALIAS) for img in imgs]
    
    imgs[0].save(f'{savename}.gif', save_all=True, append_images=imgs[1:], duration=500, loop=0)


def measure_noise_t(t, t_):
    """
    Measures the average noise in meters between a trajectory with no noise (t) and trajectory with noise (t_)
    """
    t = np.stack([t['lat'].view('f8'), t['lon'].view('f8')], axis=1)
    t_ = np.stack([t_['lat'].view('f8'), t_['lon'].view('f8')], axis=1)
    _, _, dists = geodesic.inv(t[:,1], t[:,0], t_[:,1], t_[:,0])
    return np.mean(dists)