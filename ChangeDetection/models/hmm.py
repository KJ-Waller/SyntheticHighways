from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
import networkx as nx
from tqdm import tqdm

class HMMChangeDetector(object):
    def __init__(self, G1):
        self.G1 = G1.copy()
        graph = self.format_map_hmm(self.G1)
        self.hmm_map = InMemMap("G1", graph=graph, use_latlon=True)
        self.hmm_matcher = DistanceMatcher(self.hmm_map, max_dist=1000, max_dist_init=1000, min_prob_norm=0.001, non_emitting_length_factor=0.75, obs_noise=10, obs_noise_ne=75, dist_noise=10, non_emitting_edgeid=False)
    
    def format_map_hmm(self, G):
        graph = {node[0]: ((node[1]['lat'], node[1]['lon']), list(nx.all_neighbors(G, node[0]))) for node in G.nodes(data=True)}

        return graph
    
    def format_traj_hmm(self, t):
        path = [(p['lat'], p['lon']) for p in t]
        return path
        
    def forward(self, T2):
        G_edge_scores = {}
        pbar = tqdm(enumerate(T2))
        
        no_matches = []
        
        for i, t in pbar:
            pbar.set_description(desc=f"Map matching trajectory: {i}/{len(T2)}")
            selected_edges = self.map_match_trajectory(t)
            if len(selected_edges) == 0:
                no_matches.append(i)
            
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
        
        print(f'No matches for {len(no_matches)}/{len(T2)} trajectories')
        
        return self.G1
    
    def map_match_trajectory(self, t):
        path = self.format_traj_hmm(t)
        states, num = self.hmm_matcher.match(path)
            
        return states