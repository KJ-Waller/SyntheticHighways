from tqdm import tqdm
from bresenham import bresenham
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class HistogramDetector(object):
    def __init__(self, G1, bbox, hist_dims=(200,200)):
        self.G1 = G1.copy()
        self.bbox = bbox
        self.hist_dims = hist_dims
    
    def forward(self, T2):
        hist, lat_bins, lon_bins = self.init_hist(self.bbox, self.hist_dims)
        for t in tqdm(T2):
            for i, p in enumerate(t):
                if i == 0:
                    continue
                edge = [t[i-1], p]
                hist = self.edge_to_hist(edge, hist, lat_bins, lon_bins)
                
        G_edge_scores = self.hist_to_scores(hist, lat_bins, lon_bins)
        
        nx.set_edge_attributes(self.G1, G_edge_scores, name='weight')
        
        return self.G1
    
    def init_hist(self, bbox, dims=(500,500)):
        lat_min, lat_max, lon_min, lon_max = bbox

        lat_bins = np.linspace(lat_min, lat_max, num=dims[1]-1)
        lon_bins = np.linspace(lon_min, lon_max, num=dims[0]-1)

        canvas = np.zeros(dims)
        return canvas, lat_bins, lon_bins

    def edge_to_hist(self, edge, hist, lat_bins, lon_bins):
        n1_lat, n1_lon = edge[0]['lat'], edge[0]['lon']
        n2_lat, n2_lon = edge[1]['lat'], edge[1]['lon']

        lats_binned = np.digitize((n1_lat, n2_lat), lat_bins)
        lons_binned = np.digitize((n1_lon, n2_lon), lon_bins)
        p1 = (lons_binned[0], lats_binned[0])
        p2 = (lons_binned[1], lats_binned[1])

        for lat, lon in zip(lats_binned, lons_binned):
            hist[lon][lat] = -1.0

        points = list(bresenham(*p1, *p2))
        for (lon, lat) in points:
            hist[lon][lat] = -1.0


        return hist
    
    def hist_to_scores(self, hist, lat_bins, lon_bins):
        edge_scores = {}
        for edge in self.G1.edges(data=True):
            edge_id = edge[:2]
            n1_lat, n1_lon = edge[2]['endpoints']['lat1'], edge[2]['endpoints']['lon1']
            n2_lat, n2_lon = edge[2]['endpoints']['lat2'], edge[2]['endpoints']['lon2']

            lats_binned = np.digitize((n1_lat, n2_lat), lat_bins)
            lons_binned = np.digitize((n1_lon, n2_lon), lon_bins)

            p1 = (lons_binned[0], lats_binned[0])
            p2 = (lons_binned[1], lats_binned[1])

            points = list(bresenham(*p1, *p2))

            edge_score = []
            for (lon, lat) in points:
                edge_score.append(hist[lon][lat])

                edge_score.append(hist[lon-1][lat])
                edge_score.append(hist[lon][lat-1])
                edge_score.append(hist[lon+1][lat])
                edge_score.append(hist[lon][lat+1])
                edge_score.append(hist[lon+1][lat+1])
                edge_score.append(hist[lon-1][lat-1])
                
            edge_score = np.mean(edge_score)
            edge_scores[edge_id] = edge_score

        return edge_scores
    
    def find_threshold(self):
        scores = nx.get_edge_attributes(self.G1, name='weight')

        possible_thresholds = []
        
        n, bins, _ = plt.hist(list(scores.values()))
        for i, (num, b) in enumerate(zip(n, bins)):
            if i == 0 or i == (len(n)-1):
                continue
            if num < n[i-1] and num < n[i+1]:
                possible_thresholds.append(b)
            
        if len(possible_thresholds) == 1:
            threshold = possible_thresholds[0]
        elif len(possible_thresholds) == 0:
            threshold = -0.4
        else:
            threshold = possible_thresholds[-1]
        return threshold

