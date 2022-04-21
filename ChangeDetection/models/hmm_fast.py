from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from models.hmm import HMMChangeDetector
import os
import numpy as np

class HMMChangeDetectorFast(object):
    def __init__(self, G1, num_cpu=None, use_latlon=True, max_dist=1000, max_dist_init=1000, 
                min_prob_norm=0.001, non_emitting_states=True,
                non_emitting_length_factor=0.75, obs_noise=50, 
                obs_noise_ne=75, dist_noise=10, non_emitting_edgeid=False):
        
        # Initialize global variables/parameters
        self.use_latlon = use_latlon
        self.max_dist, self.max_dist_init, self.min_prob_norm, self.non_emitting_length_factor, \
            self.obs_noise, self.obs_noise_ne, self.dist_noise, self.non_emitting_edgeid, self.non_emitting_states, self.avoid_goingback \
             = max_dist, max_dist_init, min_prob_norm, non_emitting_length_factor, obs_noise, obs_noise_ne, \
                 dist_noise, non_emitting_edgeid, non_emitting_states, avoid_goingback
        
        if num_cpu is None:
            self.num_cpu = os.cpu_count()
        else:
            self.num_cpu = num_cpu
            if self.num_cpu > os.cpu_count():
                raise ValueError(f"num_cpu argument ({self.num_cpu}) is higher than number of cpus on this machine ({os.cpu_count()})")
        
        # Setup HMM map and matcher
        self.G1 = G1.copy()

    def parallel_forward(self, T2):
        hmm_det = HMMChangeDetector(self.G1, use_latlon=self.use_latlon, max_dist=self.max_dist, max_dist_init=self.max_dist_init,
                                    min_prob_norm=self.min_prob_norm, non_emitting_states=self.non_emitting_states,
                                    non_emitting_length_factor=self.non_emitting_length_factor, obs_noise=self.obs_noise,
                                    obs_noise_ne=self.obs_noise_ne, dist_noise=self.dist_noise, non_emitting_edgeid=self.non_emitting_edgeid
                                    avoid_goingback=self.avoid_goingback, enable_pbar=False)
        return hmm_det.forward(T2)

    def forward(self, T2):
        # Split trajectories into equal sized chunks given cpu count
        T2 = np.array_split(T2, self.num_cpu)

        # Create processes pool and input
        pool = Pool(self.num_cpu)

        # Run map
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