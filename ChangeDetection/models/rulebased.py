import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.special import softmax

class RulebasedDetector(object):
    def __init__(self, G1, k=10, Cd=1.0, Ch=.1):
        self.G1 = G1.copy()
        self.k = k
        self.Cd, self.Ch = Cd, Ch
        
        self.edge_idx = {i: edge for i, edge in enumerate(G1.edges(data=True))}
        
        self.G_cp = np.array([(edge[2]['middle_coordinate']['lat'], edge[2]['middle_coordinate']['lon']) for edge in self.G1.edges(data=True)])
        self.G_headings = np.array([edge[2]['fwd_azimuth'] for edge in self.G1.edges(data=True)])
        self.tree = KDTree(self.G_cp)
        
        
    def forward(self, T2):
        G_edge_scores = {}

        for i, t in enumerate(T2):
            selected_edges = self.map_match_trajectory(t)
            for selected_edge in selected_edges:
                if selected_edge not in G_edge_scores.keys():
                    G_edge_scores[selected_edge] = -1
                else:
                    G_edge_scores[selected_edge] -= 1
                        

        edges = set(self.G1.edges)
        selected_edges = set(G_edge_scores.keys())
        missing_edges = edges.difference(selected_edges)
        for edge in missing_edges:
            G_edge_scores[edge] = 0
        
        nx.set_edge_attributes(self.G1, G_edge_scores, name='weight')
        
        return self.G1
    
    def map_match_trajectory(self, t):
        selected_edges = []
        for p in t:
            selected_edge = self.map_match_point(p)
            selected_edges.append(selected_edge)
            
        return selected_edges
            
            
    def map_match_point(self, p):
        pos = np.array([p['lat'], p['lon']])
        heading = np.deg2rad(p['heading'])
        # heading = p['heading']

        dists, idxs = self.tree.query(pos, k=self.k)
        dists, idxs = dists[dists != np.inf], idxs[dists != np.inf]
        
        headings = self.G_headings[idxs]
        
        # Calculate a score based on heading difference
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
        
        # Calculate a score based on heading and distance
        scores = (self.Cd*segment_dists) + (self.Ch*heading_scores)
        
        # Get edge with lowest value
        best_idx = np.argmin(scores)
        selected_edge = self.edge_idx[idxs[best_idx]][:2]
    
        return selected_edge