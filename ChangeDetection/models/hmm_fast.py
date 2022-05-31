from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from models.hmm import HMMChangeDetector
import os
import numpy as np

"""
This class implements the parallelized version of the HMM change detector
"""

class HMMChangeDetectorFast(object):
    def __init__(self, G1, use_latlon=False, obs_noise=40, obs_noise_ne=40, max_dist_init=400,
                max_dist=400, min_prob_norm=0.001, non_emitting_states=True, non_emitting_length_factor=0.75,
                max_lattice_width=None, dist_noise=40, dist_noise_ne=40, restrained_ne=True, avoid_goingback=True,
                num_cpu=12):

        """
        Initializes the parallelized HMM change detector

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
        
        if num_cpu is None:
            self.num_cpu = os.cpu_count()
        else:
            self.num_cpu = num_cpu
            if self.num_cpu > os.cpu_count():
                raise ValueError(f"num_cpu argument ({self.num_cpu}) is higher than number of cpus on this machine ({os.cpu_count()})")
        
        # Setup HMM map and matcher
        self.G1 = G1.copy()

    def parallel_forward(self, T2):
        """
        Function which is parallelized over multiple threads
        """
        hmm_det = HMMChangeDetector(self.G1, use_latlon=self.use_latlon, obs_noise=self.obs_noise, obs_noise_ne=self.obs_noise_ne,
                                    max_dist_init=self.max_dist_init, max_dist=self.max_dist, min_prob_norm=self.min_prob_norm,
                                    non_emitting_states=self.non_emitting_states, non_emitting_length_factor=self.non_emitting_length_factor,
                                    max_lattice_width=self.max_lattice_width, dist_noise=self.dist_noise, dist_noise_ne=self.dist_noise_ne,
                                    restrained_ne=self.restrained_ne, avoid_goingback=self.avoid_goingback, enable_pbar=False)
        return hmm_det.forward(T2)

    def forward(self, T2):
        """
        Infers the weights/scores for each edge in the map self.G1, given trajectories T2

        -------
        Params
        -------
        T2 : list
            List of trajectories
        """

        # Split trajectories into equal sized chunks given cpu count
        T2 = np.array_split(T2, self.num_cpu)

        # Create processes pool and input
        pool = Pool(self.num_cpu)

        # Run pool map
        results = []
        pbar = tqdm(pool.imap_unordered(self.parallel_forward, T2), total=len(T2))
        pbar.set_description('Map matching trajectories')
        for result in pbar:
            results.append(result)
        pool.close()
        pool.join()
        pbar.close()
        
        combined_weights = {}
        for G1 in results:
            for edge in G1.edges(data=True):
                edge_key = edge[:2]
                curr_weight = edge[2]['weight']
                if edge_key not in combined_weights.keys():
                    combined_weights[edge_key] = curr_weight
                else:
                    combined_weights[edge_key] += curr_weight

        nx.set_edge_attributes(self.G1, combined_weights, name='weight')
        return self.G1

if __name__ == '__main__':
    from SHDataset import SHDataset
    from utils import *

    dataset = SHDataset(noise=False)
    G1,T1,G2,T2 = dataset.read_snapshots(0, bbox=(52.355, 52.365, 4.860, 4.900))

    T1['T'] = random.sample(T1['T'], k=100)
    T2['T'] = random.sample(T2['T'], k=100)

    hmm_det = HMMChangeDetectorFast(G1)
    results = hmm_det.forward(T2['T'])
    print('break')