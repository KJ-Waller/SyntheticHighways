import numpy as np
import networkx as nx

class RandomDetector(object):
    def __init__(self, G1):
        self.G1 = G1.copy()
    
    def forward(self, T2):
        edge_scores = {edge: np.random.random() for edge in self.G1.edges}
        nx.set_edge_attributes(self.G1, edge_scores, name='weight')
        return self.G1