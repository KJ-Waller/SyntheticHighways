import os
from tracemalloc import start
from turtle import back
import xml.etree.ElementTree as ET
from utils import *
import pandas as pd
from datetime import datetime
import pickle5 as pickle
from tqdm import tqdm
import numpy as np
import pyproj
from geographiclib.geodesic import Geodesic
geodesic = pyproj.Geod(ellps='WGS84')

class SHDataset(object):
    def __init__(self, dataset_dir='./dataset/', split_threshold=400, ref_coord=(52.356758, 4.894004),
                coord_scale_factor=1, noise=True):
        """
        Initializes the dataset

        -------
        Params
        -------
        dataset_dir : str
            Directory of the dataset
        split_threshold : number
            What threshold to use when splitting/cleaning up trajectories.
        """
        
        # Initialize global variables
        self.split_threshold = split_threshold
        self.ref_coord = ref_coord
        self.coord_scale_factor = coord_scale_factor
        self.noise = noise
        self.noise_mu = 0.0
        self.noise_sigma = 0.0001
        self.dataset_dir = dataset_dir
        self.rawdata_dir = os.path.join(self.dataset_dir, 'raw_data')
        
        # Read raw data XML filenames
        self.files = [os.path.join(self.rawdata_dir, fname) for fname in sorted(os.listdir(self.rawdata_dir)) if '.xml' in fname]
        self.maps = []
        for i in range(len(self.files)):
            map_name = self.files[i]
            if i%7 == 0:
                map_name = map_name[map_name.rfind('/')+1:map_name.rfind('_map')-2]
                curr_map = {
                    'map_name': map_name,
                    'snapshot1': {
                        'map': self.files[i]
                    }
                }
            if i%7 == 1:
                curr_map['snapshot1']['paths'] = self.files[i]
            if i%7 == 2:
                curr_map['snapshot1']['trajectories'] = self.files[i]
            if i%7 == 3:
                curr_map['snapshot2'] = {
                    'map': self.files[i]
                }
            if i%7 == 4:
                curr_map['snapshot2']['paths'] = self.files[i]
            if i%7 == 5:
                curr_map['snapshot2']['trajectories'] = self.files[i]
            if i%7 == 6:
                curr_map['changes'] = self.files[i]
                self.maps.append(curr_map)
                
                
        # Create cleaned dataset file in "clean_data" directory if it doesn't exist yet
        # Otherwise, load it from file
        self.cleandata_dir = os.path.join(self.dataset_dir, 'clean_data')
        if not os.path.exists(self.cleandata_dir):
            os.mkdir(self.cleandata_dir)
            
        if self.noise:
            self.dataset_fname = os.path.join(self.cleandata_dir, 'dataset_cleaned_noise.hdf5')
        else:
            self.dataset_fname = os.path.join(self.cleandata_dir, 'dataset_cleaned.hdf5')

        if not os.path.isfile(self.dataset_fname):
            self.data = []
            pbar = tqdm(range(self.__len__()))
            for i in pbar:
                pbar.set_description(desc=f"Loading map: {self.maps[i]['map_name']}")
                G1,T1,G2,T2 = self.load_snapshots(i)

                self.data.append((G1,T1,G2,T2))
        
            self.data = np.array(self.data)
            with open(self.dataset_fname, 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:

            with open(self.dataset_fname, 'rb') as handle:
                self.data = pickle.load(handle)

    def read_snapshots(self, i):
        return self.data[i]
                
    def load_snapshots(self, i):
        """
        Loads the snapshots of city with the given index
        ------
        Params
        ------
        i : int
            The index of which city/savegame to load
        """

        # Get the filenames containing the map and trajectories of the two snapshots
        map1_fname = self.maps[i]['snapshot1']['map']
        traj1_fname = self.maps[i]['snapshot1']['trajectories']
        path1_fname = self.maps[i]['snapshot1']['paths']
        map2_fname = self.maps[i]['snapshot2']['map']
        traj2_fname = self.maps[i]['snapshot2']['trajectories']
        path2_fname = self.maps[i]['snapshot2']['paths']
        
        # Read them into lists
        G1 = self.parse_map(map1_fname)
        T1 = self.parse_trajectories(traj1_fname)
        P1 = self.parse_paths(path1_fname)
        G2 = self.parse_map(map2_fname)
        T2 = self.parse_trajectories(traj2_fname)
        P2 = self.parse_paths(path2_fname)
        
        return G1, {'T': T1, 'P': P1}, G2, {'T': T2, 'P': P2}
    
    def parse_map(self, xml_fname, service='Road'):
        """
        Reads a map from the given filename, only retrieving the specified road types, returning
        a dictionary with two entries (nodes and edges), which contain lists of objects
        ------
        Params
        ------
        xml_fname : str
            XML file which to open
        service : str
            The service type (or segment/node type) which to extract
        """
        
        # Read XML file
        map_tree = ET.parse(xml_fname)
        map_root = map_tree.getroot()
        nodes = map_root[0]
        segments = map_root[1]
        
        # Initialize NetworkX graph
        G = nx.Graph()
        
        # Add nodes with service specified by 'service' argument
        for node in nodes:
            if node.attrib['Service'] == service:
                coords = cart_to_wgs84(self.ref_coord, 
                                       float(node[0].attrib['x']), float(node[0].attrib['z']),
                                      scale_factor=self.coord_scale_factor)
                G.add_node(
                    int(node.attrib['Id']),
                    lat=coords[0],
                    lon=coords[1],
                    elev=float(node.attrib['Elevation']),
                    road_service=node.attrib['Service'],
                    road_subservice=node.attrib['SubService'],
                    color='blue'
                )
                
        nodes = G.nodes(data=True)

        # Add edges if both nodes are present in graph
        for i, segment in enumerate(segments):
            
            start_nodeid = int(segment[0].attrib['NodeId'])
            end_nodeid = int(segment[1].attrib['NodeId'])
            
            if start_nodeid in G.nodes and end_nodeid in G.nodes:
                n1, n2 = nodes[start_nodeid], nodes[end_nodeid]
                lat1, lon1, lat2, lon2 = n1['lat'], n1['lon'], n2['lat'], n2['lon']

                # Calculate forward and backward azimuth/heading and edge length
                fwd_azimuth, back_azimuth, dist = geodesic.inv(lon1, lat1, lon2, lat2)
                fwd_azimuth = 360 + fwd_azimuth if fwd_azimuth < 0 else fwd_azimuth
                back_azimuth = 360 + back_azimuth if back_azimuth < 0 else back_azimuth

                # Calculate middlepoint of edge
                l = Geodesic.WGS84.InverseLine(lat1, lon1, lat2, lon2)

                # Compute the midpoint
                m = l.Position(0.5 * l.s13)

                G.add_edge(
                    start_nodeid, end_nodeid,
                    segment_id=int(segment.attrib['SegmentId']),
                    prefabid=int(segment.attrib['PrefabId']),
                    prefabname=segment.attrib['PrefabName'],
                    fwd_azimuth=fwd_azimuth,
                    back_azimuth=back_azimuth,
                    length=dist,
                    middle_coordinate={
                        'lat': m['lat2'],
                        'lon': m['lon2']
                    },
                    endpoints={
                        'lat1': lat1,
                        'lon1': lon1,
                        'lat2': lat2,
                        'lon2': lon2,
                    },
                    color='blue'
                )


                
        return G
    
    def parse_trajectories(self, xml_fname):
        """
        Reads the trajectories from the given XML filename
        ------
        Params
        ------
        xml_fname : str
            The XML filename which to read
        """
        # Read the XML file
        traj_tree = ET.parse(xml_fname)
        traj_root = traj_tree.getroot()
        vehicles = traj_root[0]

        # Initialize NetworkX graph
        G = nx.Graph()
        
        # Convert XML formatted trajectories to list of lists of objects
        trajectories = []
        for vehicle in vehicles:
            if vehicle.attrib['VehicleType'] != 'Car':
                continue
                
            trajectory = []
            for pos in vehicle:
                lat, lon, speed = float(pos.attrib['x']), float(pos.attrib['z']), float(pos.attrib['speed'])
                n1, n2, n3 = float(pos[0].attrib['n1']), float(pos[0].attrib['n2']), float(pos[0].attrib['n3'])
                x, y, z = float(pos[1].attrib['x']), float(pos[1].attrib['y']), float(pos[1].attrib['z']) 

                # Skip point if it's velocity vector == 0 and heading (xyz) == [0,0,1]
                if [n1,n2,n3] == [0,0,0] and [x,y,z] == [0,0,1]:
                    continue
                point = {
                    'lat': lat,
                    'lon': lon,
                    'speed': speed,
                    'x': x,
                    'y': y,
                    'z': z,
                    'heading': get_heading(float(pos[1].attrib['z']), float(pos[1].attrib['x'])),
                    'timestamp': pos.attrib['timestamp'],
                    'gt_segment': int(pos[2].attrib['Segment']),
                    'path_id': int(pos[2].attrib['PathId']),
                    'pathpos_idx': int(pos[2].attrib['PathPosIdx']),
                }
                trajectory.append(point)

            if len(trajectory) > 0:
                trajectories.append(trajectory)

        # Split trajectories into sub trajectories for some trajectories which are merged
        trajectories = self.clean_t(trajectories, self.split_threshold)
        
        # Convert coordinates to wgs84
        x,y = zip(*[(t['lat'], t['lon']) for traj in trajectories for t in traj])
        coords = cart_to_wgs84(self.ref_coord, x, y, scale_factor=self.coord_scale_factor)
        if self.noise:
            coords = self.add_noise(coords, self.noise_mu, self.noise_sigma)
        new_T = []
        i = 0
        for traj in trajectories:
            curr_t = []
            for t in traj:
                curr_t.append({
                    'lat': coords[i][0],
                    'lon': coords[i][1],
                    'speed': t['speed'],
                    'x': t['x'],
                    'y': t['y'],
                    'z': t['z'],
                    'heading': t['heading'],
                    'timestamp': t['timestamp'],
                    'gt_segment': t['gt_segment'],
                    'path_id': t['path_id'],
                    'pathpos_idx': t['pathpos_idx'],
                })
                i+=1

            new_T.append(curr_t)
        
        return new_T

    def parse_paths(self, xml_fname):
        """
        Reads the ground truth paths from the given XML filename
        ------
        Params
        ------
        xml_fname : str
            The XML filename which to read
        """
        # Read the XML file
        paths_tree = ET.parse(xml_fname)
        paths_root = paths_tree.getroot()

        paths = {}

        for path in paths_root:
            path_id = int(path.attrib['PathId'])
            segments = [int(segment.attrib['SegmentId']) for segment in path]
            paths[path_id] = segments

        return paths
        
    
    def clean_t(self, T, thresh):
        """
        Clean the trajectories using the given threshold. Many trajectories exported should be 
        split up into separate trajectories. This function takes the trajectories, calculating distance
        between each subsequent point and splits trajectories up if the distance between points is 
        higher than the given threshold
        ------
        Params
        ------
        T : list of trajectories
            The list of trajectory objects containing lat and lon coordinates at least
        tresh : number
            Theshold for splitting up trajectories
        """
        new_T = []
        for traj in T:
            splits = self.split_traj(traj, thresh=thresh)
            new_T.extend(splits)
        return new_T

    def add_noise(self, coords, mu, sigma):
        noise_lat = np.random.normal(mu, sigma, len(coords))
        noise_lon = np.random.normal(mu, sigma, len(coords))
        noise = np.stack((noise_lat, noise_lon), axis=1)
        coords = coords + noise
        return coords

    def split_traj(self, traj, thresh):
        if len(traj) == 1:
            return [traj]
        
        t_pos = np.array([(t['lat'], t['lon']) for t in traj])
        speeds = np.linalg.norm(np.diff(t_pos, axis=0), axis=1)
        idxs = np.argwhere(speeds > thresh)[:,0] + 1
        traj = np.array(traj)
        splits = np.split(traj, idxs, axis=0)
        splits = [split.tolist() for split in splits]
        return splits
        
    def __len__(self):
        return len(self.maps)

# dataset = SHDataset()
# dataset = SHDataset(noise=True, dataset_dir='./dataset_250/')
# G1,T1,G2,T2 = dataset.read_snapshots(0)
# bbox = (52.355, 52.365, 4.860, 4.900)
# G1,T1,G2,T2 = filter_bbox_snapshots(G1,T1,G2,T2,bbox)