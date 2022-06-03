from tqdm import tqdm
from bresenham import bresenham
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

"""
This class implements the histogram change detector
"""

class HistogramDetector(object):
    def __init__(self, G1, bbox, cell_res=1e-5, score_calc_method='intersect', accumulate_scores_hist=False):
        
        """
        Initializes the histogram change detector

        -------
        Params
        -------
        G1 : NetworkX Graph
            The graph of the map in the first snapshot (ie, before changes have occured)
        bbox : 4-tuple of floats
            The bounding box that is used for the map, used to bin latitude and longitudes
        hist_dims : 2-tuple of ints
            The dimension/discretization to use when creating the 2D histogram
        """

        # Initialize global variables/parameters
        self.G1 = G1.copy()
        self.bbox = bbox
        self.cell_res = cell_res
        self.score_calc_method = score_calc_method
        self.accumulate_scores_hist = accumulate_scores_hist
    
    def forward(self, T2):
        """
        Infers the weights/scores for each edge in the map self.G1, given trajectories T2

        -------
        Params
        -------
        T2 : list
            List of trajectories
        """

        # Initialize the histogram
        self.hist, lat_bins, lon_bins = self.init_hist(self.bbox, self.cell_res)

        # For every trajectory, fill in intersecting grid cells in the histogram
        for t in tqdm(T2, desc='Running Histogram Change Detector'):
            for i, p in enumerate(t):
                if i == 0:
                    continue
                edge = [t[i-1], p]
                self.hist = self.edge_to_hist(edge, self.hist, lat_bins, lon_bins)

        # Extract edge scores from the histogram
        G_edge_scores = self.hist_to_scores(self.hist, lat_bins, lon_bins)
        
        # Set the scores as weights in the graph
        nx.set_edge_attributes(self.G1, G_edge_scores, name='weight')
        
        # Return predicted graph
        return self.G1
    
    def init_hist(self, bbox, cell_res=0.0001):
        """
        Initializes an empty 2D histogram
        """
        lat_min, lat_max, lon_min, lon_max = bbox

        lat_bins = np.arange(lat_min, lat_max, cell_res)
        lon_bins = np.arange(lon_min, lon_max, cell_res)

        # dims = (lat_bins.size, lon_bins.size)
        dims = (lon_bins.size+1, lat_bins.size+1)

        hist = np.zeros(dims)
        return hist, lat_bins, lon_bins

    def edge_to_hist(self, edge, hist, lat_bins, lon_bins):
        """
        Adds a single trajectory edge to the 2D histogram
        """
        n1_lat, n1_lon = edge[0]['lat'], edge[0]['lon']
        n2_lat, n2_lon = edge[1]['lat'], edge[1]['lon']

        lats_binned = np.digitize((n1_lat, n2_lat), lat_bins)
        lons_binned = np.digitize((n1_lon, n2_lon), lon_bins)
        p1 = (lons_binned[0], lats_binned[0])
        p2 = (lons_binned[1], lats_binned[1])

        for lat, lon in zip(lats_binned, lons_binned):
            if self.accumulate_scores_hist:
                hist[lon][lat] += -1.0
            else:
                hist[lon][lat] = -1.0

        points = list(bresenham(*p1, *p2))
        for (lon, lat) in points:
            if self.accumulate_scores_hist:
                hist[lon][lat] += -1.0
            else:
                hist[lon][lat] = -1.0

        return hist
    
    def hist_to_scores(self, hist, lat_bins, lon_bins):
        """
        Extracts the scores out of the 2D histogram by taking the average 
        grid cell values where each edge intersects
        """
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

            if self.score_calc_method == 'center':
                points = points[int(len(points)*0.25):-int(len(points)*0.25)]

            edge_score = []
            for (lon, lat) in points:
                edge_score.append(hist[lon][lat])

                edge_score.append(hist[lon-1][lat])
                edge_score.append(hist[lon][lat-1])
                edge_score.append(hist[lon+1][lat])
                edge_score.append(hist[lon][lat+1])
                edge_score.append(hist[lon+1][lat+1])
                edge_score.append(hist[lon-1][lat-1])
            
            if len(edge_score) == 0:
                edge_score = 0
            else:
                edge_score = np.mean(edge_score)
            edge_scores[edge_id] = edge_score

        return edge_scores
    
    def find_threshold(self):
        """
        Finds a threshold based on the (1D) histogram of the scores. Tries
        to find a low point between peaks
        """
        scores = nx.get_edge_attributes(self.G1, name='weight')

        possible_thresholds = []
        
        n, bins, _ = plt.hist(list(scores.values()))
        for i, (num, b) in enumerate(zip(n, bins)):
            if i == 0 or i == (len(n)-1):
                continue
            if (num <= n[i-1] and num < n[i+1]) or (num < n[i-1] and num <= n[i+1]):
                possible_thresholds.append((b, num))
            
        if len(possible_thresholds) == 1:
            threshold = possible_thresholds[0][0]
        elif len(possible_thresholds) == 0:
            threshold = -0.4
        else:
            threshold = possible_thresholds[0][0]
        return threshold