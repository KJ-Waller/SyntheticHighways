from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
import networkx as nx
from tqdm import tqdm

"""
This class implements the HMM change detector
"""

class HMMChangeDetector(object):
    def __init__(self, G1, use_latlon=False, obs_noise=4, obs_noise_ne=4, max_dist_init=100,
                max_dist=100, min_prob_norm=0.001, non_emitting_states=True, non_emitting_length_factor=0.75,
                max_lattice_width=None, dist_noise=10, dist_noise_ne=10, restrained_ne=True, avoid_goingback=True,
                enable_pbar=True):

        """
        Initializes the HMM change detector

        -------
        Params
        -------
        G1 : NetworkX Graph
            The graph of the map in the first snapshot (ie, before changes have occured)
        use_latlon : boolean
            Whether to use latitude/longitude or convert the coordinates to x,y coordinates

        """

        # Initialize global variables/parameters
        self.use_latlon = use_latlon
        self.obs_noise, self.obs_noise_ne, self.max_dist_init, self.max_dist, self.min_prob_norm, \
            self.non_emitting_states, self.non_emitting_length_factor, self.max_lattice_width, \
            self.dist_noise, self.dist_noise_ne, self.restrained_ne, self.avoid_goingback \
            = obs_noise, obs_noise_ne, max_dist_init, max_dist, min_prob_norm, non_emitting_states,\
            non_emitting_length_factor, max_lattice_width, dist_noise, dist_noise_ne, restrained_ne, \
            avoid_goingback
        self.enable_pbar = enable_pbar
        
        # Setup HMM map and matcher
        self.G1 = G1.copy()
        graph = self.format_map_hmm(self.G1)

        if not self.use_latlon:
            map_latlon = InMemMap("G1", graph=graph, use_latlon=not self.use_latlon, use_rtree=True, index_edges=True)
            self.hmm_map = map_latlon.to_xy()
        else:
            self.hmm_map = InMemMap("G1", graph=graph, use_latlon=self.use_latlon, use_rtree=True, index_edges=True)

        self.hmm_matcher = DistanceMatcher(map_con=self.hmm_map, obs_noise=self.obs_noise, obs_noise_ne=self.obs_noise_ne,\
                                            max_dist_init=self.max_dist_init, max_dist=self.max_dist, min_prob_norm=self.min_prob_norm,\
                                            non_emitting_states=self.non_emitting_states, non_emitting_length_factor=self.non_emitting_length_factor,\
                                            max_lattice_width=self.max_lattice_width, dist_noise=self.dist_noise, dist_noise_ne=self.dist_noise_ne,
                                            restrained_ne=self.restrained_ne, avoid_goingback=self.avoid_goingback)
    
    def format_map_hmm(self, G):
        """
        Converts the map G into a format used by the HMM DistanceMatcher
        """
        graph = {node[0]: ((node[1]['lat'], node[1]['lon']), list(nx.all_neighbors(G, node[0]))) for node in G.nodes(data=True)}

        return graph
    
    def format_traj_hmm(self, t):
        """
        Converts the path of a trajectory to a format used by the HMM DistanceMatcher
        """
        path = [(p['lat'], p['lon']) for p in t]
        return path
        
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

        # Setup progress bar
        if self.enable_pbar:
            pbar = tqdm(enumerate(T2))
        else:
            pbar = enumerate(T2)
        
        # Keep track of what trajectories have no match found
        no_matches = []
        
        # Loop through trajectories, and for each selected edge, decrement the weight of the edge
        for i, t in pbar:
            if self.enable_pbar:
                pbar.set_description(desc=f"Map matching trajectory: {i}/{len(T2)}")
            selected_edges = self.map_match_trajectory(t)
            if len(selected_edges) == 0:
                no_matches.append(t)
            
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
        
        # Print out number of mismatched trajectories
        print(f'No matches for {len(no_matches)}/{len(T2)} trajectories')
        self.no_matches = no_matches
        
        return self.G1
    
    def map_match_trajectory(self, t):
        """
        Matches a single trajectory to a set of selected edges
        """
        if len(t) < 6:
            return []
        path = self.format_traj_hmm(t[2:-2])
        if self.use_latlon:
            states, num = self.hmm_matcher.match(path)
        else:
            path = [self.hmm_map.latlon2yx(*p) for p in path]
            states, num = self.hmm_matcher.match(path)

            
        return states