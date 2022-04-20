import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def get_heading(x,y):
    # Initialize north and direction vector which to calculate angle between
    v = np.array([x,y])
    north = np.array([1.0,0])

    # Calculate angle of vector and north
    v_ang = np.arctan2(*v[::-1])
    north_ang = np.arctan2(*north[::-1])

    # Calculate clockwise angle (heading) between the vectors
    heading = np.rad2deg((v_ang - north_ang) % (2 * np.pi))

    return heading

def nxgraph_to_map(G):
    nodes = []
    for node in G.nodes(data=True):
        nodes.append({
            'node_id': node['label'],
            'lat': node['lat'],
            'lon': node['lon'],
        })

    edges = []
    for edge in G.edges(data=True):
        edge.append({
            'segment_id': edge['label'],
            'start_nodeid': edge['start_nodeid'],
            'start_nodeid': edge['start_nodeid'],
        })

    return {
        'nodes': nodes,
        'edges': edges,
    }

def traj_to_nxgraph(graph, stringify_id=True):
    G = nx.Graph()

    color = 'red'

    key = 0
    for traj in graph:
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
    T = traj_to_nxgraph(T)
    G_T = combine_graphs(G,T)
    return G_T

def combine_graphs(G1,G2):
    G = nx.compose(G1,G2)
    return G

def plot_graph(G, figsize=(20,40), show_nodes=False, show_labels=False, node_size=5, edge_width=1.0, use_weights=False, traj_alpha=1.0, show_img=True, fontsize=5, savename=None):
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
    plt.plot()
    if show_img:
        plt.show()
    if savename is not None:
        fig1 = plt.gcf()
        fig1.savefig(f'{savename}.png')
    
def compare_snapshots(G1, G2):
    # Convert graphs to nx graphs
    # G1, G2 = map_to_nxgraph(G1), map_to_nxgraph(G2)
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
    x,y = np.array(x)*scale_factor, np.array(y)*scale_factor
    ref_lat, ref_lon = ref_coord[0], ref_coord[1]
    
    lat = ref_lat + (180.0/np.pi)*(y/6378137)
    lon = ref_lon + (180.0/np.pi)*(x/6378137)/np.cos(np.deg2rad(ref_lat))

    axis = 1 if x.size > 1 else 0
    return np.stack((lat, lon),axis=axis)


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return np.array((min(x_coordinates), min(y_coordinates))), np.array((max(x_coordinates), max(y_coordinates)))

def filter_bbox(points, bbox):
    inidx = filter_bbox_idxs(points, bbox)
    inbox = points[inidx]
    return inbox, inidx

def filter_bbox_idxs(points, bbox):
    ll, ur = bbox[0], bbox[1]
    inidx = np.all(np.logical_and(ll <= points, points <= ur), axis=1)
    return inidx

def filter_bbox_snapshots(G1,T1,G2,T2, bbox, map_offset=0.0008):
    lat_min, lat_max, lon_min, lon_max = bbox

    G1, G2 = G1.copy(), G2.copy()
    for node in G1.copy().nodes(data=True):
        if node[1]['lat'] < (lat_min-map_offset) or node[1]['lat'] > (lat_max+map_offset) \
        or node[1]['lon'] < (lon_min-(3*map_offset)) or node[1]['lon'] > (lon_max+(3*map_offset)):
            G1.remove_node(node[0])
    for node in G2.copy().nodes(data=True):
        if node[1]['lat'] < (lat_min-map_offset) or node[1]['lat'] > (lat_max+map_offset) \
        or node[1]['lon'] < (lon_min-(3*map_offset)) or node[1]['lon'] > (lon_max+(3*map_offset)):
            G2.remove_node(node[0])

    T1_new = []
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



def get_bboxs(graph, mode='map'):
    if mode == 'map':
        node_points = np.array([(node['lat'], node['lon']) for node in graph['nodes']])
        ll, ur = bounding_box(node_points)
        lats = np.linspace(ll[0], ur[0], num=10)
        longs = np.linspace(ll[1], ur[1], num=10)

        bboxs = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(longs):
                if i == len(lats)-1:
                    continue
                if j == len(longs)-1:
                    continue
                ll_cur = np.array((lat, lon))
                ur_cur = np.array((lats[i+1], longs[j+1]))
                inbox, inidx = filter_bbox(node_points, (ll_cur, ur_cur))
                if inbox.size == 0:
                    continue
                bboxs.append((ll_cur, ur_cur))

        return bboxs
    elif mode == 'traj':
#         node_points = np.array([(node['lat'], node['lon']) for node in graph['nodes']])
        node_points = np.array([(node['lat'], node['lon']) for traj in graph for node in traj])
        node_points = np.array([(node['lat'], node['lon']) for traj in graph for node in traj])
        print(node_points)
        return
    
def filter_map_bbox(G, bbox):
    # Get all node points
    node_points = np.array([(node['lat'], node['lon']) for node in G['nodes']])
    # Get all nodes separately
    nodes = np.array(G['nodes'])
    # Filter all node points from map according to bounding box
    inidxs = filter_bbox_idxs(node_points, bbox)
    
    nodes_bbox = [node for node in nodes[inidxs]]
    
    nodeids_bbox = [node['node_id'] for node in nodes[inidxs]]
    edges = G['edges']
    edges_bbox = []
    for edge in G['edges']:
        startid = edge['start_nodeid']
        endid = edge['end_nodeid']
        if startid in nodeids_bbox and endid in nodeids_bbox:
            edges_bbox.append(edge)
            
            
    return {
        'nodes': nodes_bbox,
        'edges': edges_bbox
    }

def filter_trajectories_bbox(T, bbox):
    # Get all node points
    node_points = np.array([(node['lat'], node['lon']) for traj in T for node in traj])
    # Get all nodes separately
    nodes = np.array([node for traj in T for node in traj])
    # Get trajectory number/index for each node point
    traj_idxs = np.array([i for i, traj in enumerate(T) for node in traj])
    # Filter all node points from trajectory according to bounding box
    inidxs = filter_bbox_idxs(node_points, bbox)
    # Only get trajectory number for nodes within bounding box
    traj_idxs_bbox = traj_idxs[inidxs]
    traj_bboxs = node_points[inidxs]
    nodes_bbox = nodes[inidxs]
    T_bbox = []
    prev_traj_idx = None
    for (traj_idx, node) in zip(traj_idxs_bbox, nodes_bbox):
        if prev_traj_idx is None:
            curr_traj = []
            curr_traj.append(node)
        if prev_traj_idx != traj_idx:
            T_bbox.append(curr_traj)
            curr_traj = []
            curr_traj.append(node)
        if prev_traj_idx == traj_idx:
            curr_traj.append(node)
        prev_traj_idx = traj_idx
    
    return T_bbox


import random

def format_trajmatrix(T, T_num_max=500, T_len_max=50):
    T_maxlen = np.max([len(traj) for traj in T])
    T_num = len(T)
    
    if len(T) > T_num_max:
        sampled_T = random.sample(T, T_num_max)
    else:
        sampled_T = T
        
    traj_matrix = np.zeros((T_num_max, T_len_max, 2))
    for i, traj in enumerate(sampled_T):
        print(traj)
        for j, t in enumerate(traj):
            if j >= (T_len_max-1):
                continue
            traj_matrix[i,j,0] = t['lat']
            traj_matrix[i,j,1] = t['lon']
            
    return traj_matrix

def format_mapmatrix(G, G_num_max=100, G_len_max=20):
    G_nx = map_to_nxgraph(G)
    G_pos_dict = {node['node_id']: (node['lat'], node['lon']) for node in G['nodes']}
    
    edge_traversal = list(nx.dfs_edges(G_nx, depth_limit=G_len_max))
    paths = []
    prev_node = None
    for edge in edge_traversal:
        curr_node = edge[0]
        if prev_node is None:
            prev_node = edge[1]
            path = []
            path.append(edge)
        else:

            if curr_node == prev_node:
                path.append(edge)
            if curr_node != prev_node:
                paths.append(path)
                path = []
                path.append(edge)

            prev_node = edge[1]
        
        
    map_matrix = np.zeros((G_num_max, G_len_max, 2))
    if len(paths) > G_num_max:
        sampled_paths = random.sample(paths, G_num_max)
    else:
        sampled_paths = paths
        
    for i, path in enumerate(sampled_paths):
        for j, p in enumerate(path):
            if j >= (G_len_max-1):
                continue
            map_matrix[i,j,0] = G_pos_dict[p[0]][0]
            map_matrix[i,j,1] = G_pos_dict[p[0]][1]
    return map_matrix


from geographiclib.geodesic import Geodesic
from geopy import distance

def calc_dist(p1, p2):
    dist = distance.distance((p1[0], p1[1]), (p2[0], p2[1])).m
    return dist

def get_next_seed(p1, p2, offset=50):
    
    d = distance.distance(meters=offset)
    inv = Geodesic.WGS84.Inverse(*p1, *p2)
    azi = inv['azi1']
    next_seed = d.destination(point=p1, bearing=azi)
    next_seed = (next_seed[0], next_seed[1])
    return (next_seed, azi)

def place_seeds(G, interval=50):

    edge_traversal = list(nx.edge_dfs(G))

    node_pos_dict = {node[0]: (node[1]['lat'], node[1]['lon']) for node in G.nodes(data=True)}
    edge_traversal_dist = []

    for edge in edge_traversal:
        snode, enode = edge[0], edge[1]
        spos, epos = node_pos_dict[snode], node_pos_dict[enode]
        dist = calc_dist((spos[0], spos[1]), (epos[0], epos[1]))
        edge_traversal_dist.append({
            'startid': edge[0],
            'endid': edge[1],
            'startpos': node_pos_dict[edge[0]],
            'endpos': node_pos_dict[edge[1]],
            'dist': dist
        })

    edge_traversal = edge_traversal_dist
    if len(edge_traversal) == 0:
        return None, None

    # List to keep track of seeds ((lat, lon), azi)
    seeds = []
    # List to keep track of corresponding seed edges (n1, n2)
    seed_edges = []
    
    (_, azi) = get_next_seed(edge_traversal[0]['startpos'], edge_traversal[0]['endpos'], interval)
    seeds.append((edge_traversal[0]['startpos'], azi))
    seed_edge = (edge_traversal[0]['startid'], edge_traversal[0]['endid'])
    seed_edges.append(seed_edge)
    dist_remain = 0
    

    for edge in edge_traversal:
        dist = edge['dist']
        
        # If there is distance remaining, add a seed with offset
        if dist_remain != 0:
            next_seed = get_next_seed(edge['startpos'], edge['endpos'], interval-dist_remain)
            seeds.append(next_seed)
            
            seed_edge = (edge['startid'], edge['endid'])
            seed_edges.append(seed_edge)

        prev_seed = seeds[-1]
        # If distance from prev_seed to next node <50, place seed 50 meters ahead
        dist_remain = calc_dist(prev_seed[0], edge['endpos'])
        
        # Keep placing seeds as long as dist_remain is > 50
        while dist_remain > interval:
            # Add seeds, updating dist remain from prev seed to next node
            next_seed = get_next_seed(prev_seed[0], edge['endpos'], interval)
            seeds.append(next_seed)
            seed_edge = (edge['startid'], edge['endid'])
            seed_edges.append(seed_edge)
            prev_seed = next_seed
            dist_remain = calc_dist(prev_seed[0], edge['endpos'])


        # If distance from prev_seed to next node <50, continue to next edge
        if dist_remain < interval:
            continue
        
    
    return seeds, seed_edges

def T_to_img(T, img_dim=(500,500)):

    lats = np.array([t['lat'] for traj in T for t in traj])
    lons = np.array([t['lon'] for traj in T for t in traj])
    lat_bins = np.linspace(lats.min(), lats.max(), num=img_dim[1]-1)
    lon_bins = np.linspace(lons.min(), lons.max(), num=img_dim[0]-1)
    lats_binned = np.digitize(lats, lat_bins)
    lons_binned = np.digitize(lons, lon_bins)

    img = np.zeros(img_dim)

    for idx, (lat,lon) in enumerate(zip(lats_binned, lons_binned)):
        img[lon,lat] = 1

    # img = np.flip(img, axis=1)
    img = np.rot90(img, k=1)
    
    return img

def G_to_img(G, img_dim=(500,500)):
    
    seeds, _ = place_seeds(G, interval=5)
    if seeds is None:
        return None
    lats = np.array([seed[0][0] for seed in seeds])
    lons = np.array([seed[0][1] for seed in seeds])
    lat_bins = np.linspace(lats.min(), lats.max(), num=img_dim[1]-1)
    lon_bins = np.linspace(lons.min(), lons.max(), num=img_dim[0]-1)
    lats_binned = np.digitize(lats, lat_bins)
    lons_binned = np.digitize(lons, lon_bins)

    img = np.zeros(img_dim)

    for idx, (lat,lon) in enumerate(zip(lats_binned, lons_binned)):
        img[lon,lat] += 1

    # img = np.flip(img, axis=1)
    img = np.rot90(img, k=1)
    
    return img

def bin_img(lats, lons, lat_bins, lon_bins, img_dim):

    lats_binned = np.digitize(lats, lat_bins)
    lons_binned = np.digitize(lons, lon_bins)

    img = np.zeros(img_dim)

    for idx, (lat,lon) in enumerate(zip(lats_binned, lons_binned)):
        img[lon,lat] = 1

    img = np.flip(img, axis=0)

    return img


def tile_to_img(G1, T2, G2, img_dim=(500,500)):
    seeds1, _ = place_seeds(G1, interval=5)
    seeds2, _ = place_seeds(G2, interval=5)

    G1_lats = np.array([seed[0][0] for seed in seeds1])
    G1_lons = np.array([seed[0][1] for seed in seeds1])
    G2_lats = np.array([seed[0][0] for seed in seeds2])
    G2_lons = np.array([seed[0][1] for seed in seeds2])

    
    T2_lats = np.array([t['lat'] for traj in T2 for t in traj])
    T2_lons = np.array([t['lon'] for traj in T2 for t in traj])

    lat_max = np.max([G1_lats.max(), G2_lats.max(), T2_lats.max()])
    lat_min = np.max([G1_lats.min(), G2_lats.min(), T2_lats.min()])
    lon_max = np.max([G1_lons.max(), G2_lons.max(), T2_lons.max()])
    lon_min = np.max([G1_lons.min(), G2_lons.min(), T2_lons.min()])

    
    lat_bins = np.linspace(lat_min, lat_max, num=img_dim[1]-1)
    lon_bins = np.linspace(lon_min, lon_max, num=img_dim[0]-1)

    G1_img = bin_img(G1_lats, G1_lons, lat_bins, lon_bins, img_dim)
    G2_img = bin_img(G2_lats, G2_lons, lat_bins, lon_bins, img_dim)
    T2_img = bin_img(T2_lats, T2_lons, lat_bins, lon_bins, img_dim)

    x = np.stack((G1_img, T2_img), axis=0)
    y = G2_img

    return x, y