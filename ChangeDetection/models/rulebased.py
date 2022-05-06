import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from tqdm import tqdm

"""
This class implements the rule-based change detector
"""

class RulebasedDetector(object):
    def __init__(self, G1, k=10, Cd=1.0, Ch=.1):
        """
        Initializes the rule-based change detector

        -------
        Params
        -------
        G1 : NetworkX Graph
            The graph of the map in the first snapshot (ie, before changes have occured)
        k : int
            The number of nearby edges/roads to consider when matching a timed point from a trajectory
        Cd : float
            The weight of the distance score
        Ch : float
            The weight of the heading score
        """

        # Save global parameters
        self.G1 = G1.copy()
        self.k = k
        self.Cd, self.Ch = Cd, Ch
        
        # Give a number/index to each edge in the graph
        self.edge_idx = {i: edge for i, edge in enumerate(self.G1.edges(data=True))}
        
        # Extract the headings and center coordinates for edges in the map for querying the KD-Tree
        self.G_cp = np.array([(edge[2]['middle_coordinate']['lat'], edge[2]['middle_coordinate']['lon']) for edge in self.G1.edges(data=True)])
        self.G_headings = np.array([edge[2]['fwd_azimuth'] for edge in self.G1.edges(data=True)])

        # Initialize KD-Tree on center points of edges
        self.tree = KDTree(self.G_cp)
        
        
    def forward(self, T2):
        """
        Infers the weights/scores for each edge in the map self.G1, given trajectories T2

        -------
        Params
        -------
        T2 : list
            List of trajectories
        """
        G_edge_scores = {}
        
        # Loop through trajectories, and for each selected edge, decrement the weight of the edge
        for t in tqdm(T2, desc='Running Rule-based Change Detector'):
            selected_edges = self.map_match_trajectory(t)
            for selected_edge in selected_edges:
                if selected_edge not in G_edge_scores.keys():
                    G_edge_scores[selected_edge] = -1
                else:
                    G_edge_scores[selected_edge] -= 1
                        
        # Set any missed edges to 0
        edges = set(self.G1.edges)
        selected_edges = set(G_edge_scores.keys())
        missing_edges = edges.difference(selected_edges)
        for edge in missing_edges:
            G_edge_scores[edge] = 0
        
        # Set weight in self.G1 to the scores computed above
        nx.set_edge_attributes(self.G1, G_edge_scores, name='weight')
        
        return self.G1
    
    def map_match_trajectory(self, t):
        """
        Matches a single trajectory to a set of selected edges
        """
        selected_edges = []
        for p in t:
            selected_edge = self.map_match_point(p)
            selected_edges.append(selected_edge)
            
        return selected_edges
            
            
    def map_match_point(self, p):
        """
        Matches a single timed point to an edge
        """

        # Get the position/coordinates and heading of the timed point
        pos = np.array([p['lat'], p['lon']])
        heading = np.deg2rad(p['heading'])

        # Query the KD-Tree for nearest self.k edges in map to timed point
        dists, idxs = self.tree.query(pos, k=self.k)
        dists, idxs = dists[dists != np.inf], idxs[dists != np.inf]
        
        # Get headings of the edges to consider
        headings = self.G_headings[idxs]
        
        # Calculate a score based on heading difference between the edges and the timed point
        heading_scores = (headings - heading) % 360
        heading_scores = np.abs(np.where(heading_scores >= 180, heading_scores - 360, heading_scores))
        heading_scores = (heading_scores - heading_scores.min()) / (heading_scores.max() - heading_scores.min())
        
        # Calculate a score based off of distance
        p = pos
        a = np.array([(self.edge_idx[idx][2]['endpoints']['lat1'], self.edge_idx[idx][2]['endpoints']['lon1']) for idx in idxs])
        b = np.array([(self.edge_idx[idx][2]['endpoints']['lat2'], self.edge_idx[idx][2]['endpoints']['lon2']) for idx in idxs])
        
        # normalized tangent vectors
        d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                               .reshape(-1, 1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=1)
        t = np.multiply(p - b, d).sum(axis=1)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors  
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
            
        # Calculate normalized distance
        segment_dists = np.hypot(h, c)
        segment_dists = segment_dists
        segment_dists = (segment_dists - segment_dists.min()) / (segment_dists.max() - segment_dists.min())
        
        # Calculate weighted score for each edge
        scores = (self.Cd*segment_dists) + (self.Ch*heading_scores)
        
        # The selected edge is the edge with the lowest score (lower score = more similar)
        best_idx = np.argmin(scores)
        selected_edge = self.edge_idx[idxs[best_idx]][:2]
    
        return selected_edge