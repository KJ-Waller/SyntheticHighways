import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from utils import *

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

def plot_graph(G, figsize=(10,10), show_nodes=False, show_labels=False, 
                node_size=5, edge_width=1.0, use_weights=False, traj_alpha=1.0, 
                show_img=True, fontsize=5, equal_axis_ratio=False, zoom_on_traj=False,
                savename=None):
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
    node_size : float
        How big to plot nodes (show nodes must be set to True)
    edge_width : float
        How wide to plot the edges in the graph
    use_weights : boolean
        Whether to use the weights to color the edges to make a heatmap
    traj_alpha : float range(0,1)
        Alpha/transparency of trajectories in visualization
    show_img : boolean
        Whether to show image or not
    fontsize : float
        What fontsize to use for labels, if enabled
    equal_axis_ratio : boolean
        Whether to plot x,y axes with equal aspect ratios
    zoom_on_traj : boolean
        Whether to zoom in on the trajectories or to show the entire map
    savename : string
        Location and filename to save plot to. If set to None, it will not save anything
    """

    if G.size() == 0:
        print("Graph is empty")
        return
    
    # Fetch the node positions and colors
    node_lats = nx.get_node_attributes(G, 'lat')
    node_lons = nx.get_node_attributes(G, 'lon')
    node_pos = {node: (node_lons[node], node_lats[node]) for node in node_lats.keys()}

    node_colors = list(nx.get_node_attributes(G, 'color').values())
    nodes = list(node_pos.keys())
    
    # plt.figure(1,figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the nodes if enabled
    if show_nodes:
        nx.draw_networkx_nodes(G, node_pos, nodelist=nodes, node_color=node_colors, node_size=node_size, ax=ax)
    if show_labels:
        # node_labels = nx.get_node_attributes(G, 'label')
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
            nx.draw_networkx_edges(G, node_pos, edgelist=edges_map, width=edge_width, edge_color=eweights_map, edge_cmap=plt.cm.viridis, ax=ax)
            nx.draw_networkx_edges(G, node_pos, edgelist=edges_traj, width=edge_width, edge_color=eweights_traj, edge_cmap=plt.cm.Reds, alpha=traj_alpha, ax=ax)
        # If only map available, plot in blue only
        elif len(traj_weights) == 0 and len(map_weights) != 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin = np.min(edge_weights), vmax=np.max(edge_weights)))
            nx.draw_networkx_edges(G, node_pos, width=edge_width, edge_color=edge_weights, edge_cmap=plt.cm.viridis, ax=ax)
            plt.colorbar(sm)

        # if only trajectories available, plot in red
        elif len(traj_weights) != 0 and len(map_weights) == 0:
            nx.draw_networkx_edges(G, node_pos, width=edge_width, edge_color=edge_weights, edge_cmap=plt.cm.Reds, alpha=traj_alpha, ax=ax)
        
    # Otherwise, plot edges with regular color
    else:
        edges, edge_colors = zip(*nx.get_edge_attributes(G, 'color').items())
        map_colors = [(edge, color) for (edge, color) in zip(edges, edge_colors) if type(edge[0]) == int or type(edge[0]) == float]
        traj_colors = [(edge, color) for (edge, color) in zip(edges, edge_colors) if type(edge[0]) == str]
        
        # If both map and trajectories are being plotted, plot in blue and red respectively
        if len(traj_colors) != 0 and len(map_colors) != 0:
            edges_map, colors_map = zip(*map_colors)
            edges_traj, colors_traj = zip(*traj_colors)
            nx.draw_networkx_edges(G, node_pos, edgelist=edges_map, width=edge_width, edge_color=colors_map, edge_cmap=plt.cm.Blues, ax=ax)
            nx.draw_networkx_edges(G, node_pos, edgelist=edges_traj, width=edge_width, edge_color=colors_traj, edge_cmap=plt.cm.Reds, alpha=traj_alpha, ax=ax)
        # If only map available, plot in blue only
        elif len(traj_colors) == 0 and len(map_colors) != 0:
            nx.draw_networkx_edges(G, node_pos, width=edge_width, edge_color=edge_colors, edge_cmap=plt.cm.Blues, ax=ax)
        # if only trajectories available, plot in red
        elif len(traj_colors) != 0 and len(map_colors) == 0:
            nx.draw_networkx_edges(G, node_pos, width=edge_width, edge_color=edge_colors, edge_cmap=plt.cm.Reds, alpha=traj_alpha, ax=ax)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    if zoom_on_traj:
        offset = 1e-3
        t_lats, t_lons = zip(*[(node[1]['lat'], node[1]['lon']) for node in G.nodes(data=True) if node[1]['color'] == 'red'])
        lat_bounds = (np.min(t_lats)-offset, np.max(t_lats)+offset)
        lon_bounds = (np.min(t_lons)-offset, np.max(t_lons)+offset)
        ax.set_xlim(*lon_bounds)
        ax.set_ylim(*lat_bounds)
    if equal_axis_ratio:
        ax.axis('equal')
    plt.plot()
    if show_img:
        plt.show()
    if savename is not None:
        fig1 = plt.gcf()
        fig1.savefig(f'{savename}.png')
    
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
    nx.set_edge_attributes(G1_d, 'red', name='color')
    G12_d = nx.compose(G2, G1_d)
    
    # Create graph highlighting edges in G2 not in G1
    G2_diff_edges = G2_edges - G1_edges
    G2_diff_edges = [(edge) for edge in G2.edges(data=True) if (edge[0], edge[1]) in G2_diff_edges]
    G2_d = nx.create_empty_copy(G2)
    G2_d.add_edges_from(G2_diff_edges)
    nx.set_edge_attributes(G2_d, 'red', name='color')
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