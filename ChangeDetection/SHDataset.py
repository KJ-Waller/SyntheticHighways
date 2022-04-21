import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from importlib_metadata import itertools
from utils import *
import pickle5 as pickle
from tqdm import tqdm
import numpy as np
import pyproj
from geographiclib.geodesic import Geodesic
import re
import argparse
geodesic = pyproj.Geod(ellps='WGS84')

"""
This class implements Ornstein-Uhlenbeck process, for adding noise to trajectories
"""
class OUActionNoise(object):
    def __init__(self, mu, sigma=3.0, theta=1.0, dt=.0001, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

"""
This class implements the main Synthetic Highways dataset. It reads raw datafiles from the
dataset directory and processes them to pickle files per map and per noise configuration
"""
class SHDataset(object):
    def __init__(self, dataset_dir='./dataset/', split_threshold=200, ref_coord=(52.356758, 4.894004),
                coord_scale_factor=1, noise=True, noise_config=0, save_cleaned_data=True, min_traj_len=4,
                traj_trim_size=2):
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
        if noise_config > 3:
            raise ValueError(f"There are only 4 noise configs. Please specify a noise configuration in the range[0,1,2,3]")
        
        # Initialize global variables
        self.save_cleaned_data = save_cleaned_data
        self.split_threshold = split_threshold
        self.ref_coord = ref_coord
        self.coord_scale_factor = coord_scale_factor
        self.min_traj_len = min_traj_len
        self.traj_trim_size = traj_trim_size
        self.noise = noise
        self.pnoise_shift = 0.05
        sigmas = [0.0000125+(i*0.000025) for i in range(4)]
        thetas = sigmas
        
        self.noise_mu = np.array([0,0])
        self.noise_sigma = sigmas[noise_config]
        self.noise_theta = thetas[noise_config]
        self.noise_dt = .1
        self.noise_function = OUActionNoise(self.noise_mu, self.noise_sigma, self.noise_theta, self.noise_dt, x0=[0,0])
        self.dataset_dir = dataset_dir
        self.rawdata_dir = os.path.join(self.dataset_dir, 'raw_data')
        
        # Organize the xml files in the raw data directory
        self.organize_xml_filenames()

        # Check if clean data directory exists
        self.cleandata_dir = os.path.join(self.dataset_dir, 'clean_data')
        if not os.path.exists(self.cleandata_dir):
            os.mkdir(self.cleandata_dir)

        # For each map, if cleaned file doesn't exist, save it
        for i, fnames in enumerate(self.maps):
            map_name = fnames['map_name']
            cleaned_fname = os.path.join(self.cleandata_dir, f'{map_name}_cleaned_nonoise.hdf5')
            
            # Create cleaned dataset w/o noise and save to pickle if it doesn't exist
            if not os.path.isfile(cleaned_fname) and self.save_cleaned_data:
                G1,T1,G2,T2 = self.load_snapshots(i)

                with open(cleaned_fname, 'wb') as handle:
                    print(f'Writing snapshots to pickle file: {cleaned_fname}')
                    pickle.dump((G1,T1,G2,T2), handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Add noise to trajectories for current configuration, if noise is selected/enabled
            if self.noise:
                cleaned_fname_noise = os.path.join(self.cleandata_dir, f'{map_name}_cleaned_noise_config{noise_config+1}.hdf5')
                if not os.path.isfile(cleaned_fname_noise):
                    self.maps[i]['cleaned_data'] = cleaned_fname
                    G1,T1,G2,T2 = self.read_snapshots(i)
                    T1['T'] = self.add_noise_parallel(T1['T'])
                    T2['T'] = self.add_noise_parallel(T2['T'])

                    with open(cleaned_fname_noise, 'wb') as handle:
                        print(f'Writing snapshots to pickle file: {cleaned_fname}')
                        pickle.dump((G1,T1,G2,T2), handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                self.maps[i]['cleaned_data'] = cleaned_fname

    def organize_xml_filenames(self):
        # Read raw data XML filenames
        self.files = [os.path.join(self.rawdata_dir, fname) for fname in sorted(os.listdir(self.rawdata_dir)) if '.xml' in fname]

        # First find all unique map names
        map_names = []
        for file in self.files:
            match = re.findall(r'/([A-Z0-9 .]+)_1_map.xml', file, flags=re.IGNORECASE)
            if len(match) == 1:
                map_names.append(match[0])

        # Then for each map name, find all batches for trajectories and path filenames
        # and organize them into self.maps
        self.maps = []
        map_names = sorted(list(set(map_names)))
        for map_name in map_names:
            T1_fnames = []
            T2_fnames = []
            P1_fnames = []
            P2_fnames = []
            G1_fname = None
            G2_fname = None
            changes_fname = None
            for file in self.files:
                if map_name in file and 'batch' in file and '1_trajectories' in file:
                    T1_fnames.append(file)
                if map_name in file and 'batch' in file and '2_trajectories' in file:
                    T2_fnames.append(file)
                if map_name in file and 'batch' in file and '1_paths' in file:
                    P1_fnames.append(file)
                if map_name in file and 'batch' in file and '2_paths' in file:
                    P2_fnames.append(file)
                if map_name in file and 'batch' not in file and '1_map.xml' in file:
                    G1_fname = file
                if map_name in file and 'batch' not in file and '2_map.xml' in file:
                    G2_fname = file
                if map_name in file and 'batch' not in file and 'changes' in file:
                    changes_fname = file
            self.maps.append({
                'map_name': map_name,
                'snapshot1': {
                    'map': G1_fname,
                    'trajectories': T1_fnames,
                    'paths': P1_fnames
                },
                'snapshot2': {
                    'map': G2_fname,
                    'trajectories': T2_fnames,
                    'paths': P2_fnames
                },
                'changes': changes_fname
            })


    def read_snapshots(self, i, bbox=None):
        """ 
        Main function for fetching data from a single snapshot
        """
        if self.save_cleaned_data:
            cleaned_fname = self.maps[i]['cleaned_data']
            with open(cleaned_fname, 'rb') as handle:
                G1,T1,G2,T2 = pickle.load(handle)
        else:
            G1,T1,G2,T2 = self.load_snapshots(i)

        if bbox is not None:
            G1,T1,G2,T2 = filter_bbox_snapshots(G1,T1,G2,T2, bbox)

        return G1,T1,G2,T2

    def parse_trajectories_parallel(self, traj_fnames):
        """
        This function takes a list of trajectory filenames (batches) and processes them in parallel
        """
        if len(traj_fnames) == os.cpu_count():
            pool = Pool(os.cpu_count())
        elif len(traj_fnames) < os.cpu_count():
            pool = Pool(len(traj_fnames))
        elif len(traj_fnames) > os.cpu_count():
            raise NotImplementedError(f'There are more trajectory batches than there are cpu cores. This feature has not yet been implemented')

        results = []
        pbar = tqdm(pool.imap_unordered(self.parse_trajectories, traj_fnames), total=len(traj_fnames))
        pbar.set_description('Reading trajectories')
        for result in pbar:
            results.append(result)
        pool.close()
        pool.join()
        pbar.close()

        trajectories = list(itertools.chain.from_iterable(results))
        return trajectories

    def parse_paths_parallel(self, path_fnames):
        """
        This function takes a list of path filenames (batches) and processes them in parallel
        """
        if len(path_fnames) == os.cpu_count():
            pool = Pool(os.cpu_count())
        elif len(path_fnames) < os.cpu_count():
            pool = Pool(len(path_fnames))
        elif len(path_fnames) > os.cpu_count():
            raise NotImplementedError(f'There are more path batches than there are cpu cores. This feature has not yet been implemented')

        results = []
        pbar = tqdm(pool.imap_unordered(self.parse_paths, path_fnames), total=len(path_fnames))
        pbar.set_description('Reading paths')
        for result in pbar:
            results.append(result)
        pool.close()
        pool.join()
        pbar.close()

        paths = dict(itertools.chain.from_iterable(map(dict.items, results)))
        return paths

                
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
        traj1_fnames = self.maps[i]['snapshot1']['trajectories']
        path1_fnames = self.maps[i]['snapshot1']['paths']
        map2_fname = self.maps[i]['snapshot2']['map']
        traj2_fnames = self.maps[i]['snapshot2']['trajectories']
        path2_fnames = self.maps[i]['snapshot2']['paths']
        
        # Read the files in (this will take a while)
        G1 = self.parse_map(map1_fname)
        T1 = self.parse_trajectories_parallel(traj1_fnames)
        P1 = self.parse_paths_parallel(path1_fnames)
        G2 = self.parse_map(map2_fname)
        T2 = self.parse_trajectories_parallel(traj2_fnames)
        P2 = self.parse_paths_parallel(path2_fnames)
        
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
        
        # Convert XML formatted trajectories to list of lists of objects
        trajectories = []
        for vehicle in vehicles:
            if vehicle.attrib['VehicleType'] != 'Car':
                continue
                
            trajectory = []
            for pos in vehicle:

                # Check if vehicle type still is consistent
                vehicle_type = pos.attrib['VehicleType']
                if vehicle_type != 'Car':
                    # If current position isn't from a car
                    if len(trajectory) < self.min_traj_len:
                        # If trajectory is too small, skip over it
                        trajectory = []
                        continue
                    # Otherwise, add to trajectories
                    else:
                        trajectories.append(trajectory)
                        continue

                # Process current position info
                lat, lon, speed = float(pos.attrib['x']), float(pos.attrib['z']), float(pos.attrib['speed'])
                n1, n2, n3 = float(pos[0].attrib['n1']), float(pos[0].attrib['n2']), float(pos[0].attrib['n3'])
                x, y, z = float(pos[1].attrib['x']), float(pos[1].attrib['y']), float(pos[1].attrib['z']) 

                # Skip point if it's velocity vector == 0 and heading (xyz) == [0,0,1]
                if [n1,n2,n3] == [0,0,0] and [x,y,z] == [0,0,1]:
                    continue
                
                # Aggregate information into object and append to trajectory
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
        trajectories = self.split_trajectories(trajectories, self.split_threshold)
        
        # Convert coordinates to wgs84
        trajectories = self.to_wgs84(trajectories)

        # Finally, clean trajectories by trimming and filtering on minimum length
        trajectories = self.clean_trajectories(trajectories)
        
        return trajectories

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
        
    def split_trajectories(self, T, thresh):
        """
        Splits the trajectories using the given threshold. Many trajectories exported should be 
        split up into separate trajectories. This function takes the trajectories, calculating distance
        between each subsequent point and splits trajectories up if the distance between points is 
        higher than the given threshold
        ------
        Params
        ------
        T : list of trajectories
            The list of trajectory objects containing lat and lon coordinates at least
        tresh : numberwriter
            Theshold for splitting up trajectories
        """
        new_T = []
        for traj in T:
            splits = self.split_t(traj, thresh=thresh)
            new_T.extend(splits)
        return new_T

    def clean_trajectories(self, T):
        """
        Clean the trajectories by trimming them on both ends and filtering by minimum length
        """
        
        new_T = []
        for t in T:
            new_t = t[self.traj_trim_size:-self.traj_trim_size]
            if len(new_t) < self.min_traj_len:
                continue
            else:
                new_T.append(new_t)
        return new_T

    def to_wgs84(self, T):

        i = 0
        x,y = zip(*[(t['lat'], t['lon']) for traj in T for t in traj])
        coords = cart_to_wgs84(self.ref_coord, x, y, scale_factor=self.coord_scale_factor)
        new_T = []
        for traj in T:
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


    def add_noise_parallel(self, T):
        """
        Adds noise to trajectories using multiple threads
        """
        T_batches = np.array_split(T, os.cpu_count())
        pool = Pool(os.cpu_count())

        results = []
        pbar = tqdm(pool.imap_unordered(self.add_noise, T_batches), total=len(T_batches))
        pbar.set_description('Adding noise to trajectories')
        for result in pbar:
            results.append(result)

        pool.close()
        pool.join()
        pbar.close()
        trajectories_w_noise = list(itertools.chain.from_iterable(results))
        return trajectories_w_noise

    def add_noise(self, T):
        """
        Adds noise to trajectories
        """
        new_T = []
        
        for t in T:
            new_T.append(self.add_noise_t(t))
        return new_T

    def add_noise_t(self, t):
        """
        Adds noise to a single trajectory
        """
        ou_noise = np.array([self.noise_function() for i in range(len(t))])
        old_coords = np.array([(p['lat'], p['lon']) for p in t])
        new_coords = old_coords + ou_noise

        shift_probs = np.expand_dims(np.random.binomial(1, self.pnoise_shift, len(t)), axis=1)
        shift_noise_lat = np.random.normal(0, self.noise_sigma, len(t))
        shift_noise_lon = np.random.normal(0, self.noise_sigma, len(t))
        shift_noise = np.stack((shift_noise_lat, shift_noise_lon), axis=1)
        shift_noise = shift_noise * shift_probs

        final_noise = new_coords + shift_noise

        new_t = []
        for i, p in enumerate(t):
            new_t.append({
                'lat': final_noise[i,0],
                'lon': final_noise[i,1],
                'speed': p['speed'],
                'x': p['x'],
                'y': p['y'],
                'z': p['z'],
                'heading': p['heading'],
                'timestamp': p['timestamp'],
                'gt_segment': p['gt_segment'],
                'path_id': p['path_id'],
                'pathpos_idx': p['pathpos_idx'],
            })
        
        self.noise_function.reset()

        return new_t

    def split_t(self, t, thresh):
        """
        Splits up trajectories given some threshold
        """
        if len(t) == 1:
            return [t]
        
        t_pos = np.array([(p['lat'], p['lon']) for p in t])
        speeds = np.linalg.norm(np.diff(t_pos, axis=0), axis=1)
        idxs = np.argwhere(speeds > thresh)[:,0] + 1
        t = np.array(t)
        splits = np.split(t, idxs, axis=0)
        splits = [split.tolist() for split in splits]
        return splits
        
    def __len__(self):
        return len(self.maps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_dir', default='./dataset/', type=str, help='Dataset root directory')
    parser.add_argument('--noise', default=False, action='store_true', help='Add noise to trajectories')
    parser.add_argument('--noise_config', default=0, type=int, help='Which noise configuration to use')
    parser.add_argument('--split_threshold', default=200, type=int, help='What threshold to use when splitting up trajectories')
    parser.add_argument('--preprocess_dataset', default=True, action='store_true', help='Whether to preprocess the entire dataset')

    args = parser.parse_args()

    if args.preprocess_dataset:
        print('Processing no noise dataset')
        dataset = SHDataset(noise=False, dataset_dir=args.dataset_dir, noise_config=0)
        for i in range(4):
            print(f'Processing dataset w/ noise config {i}')
            dataset = SHDataset(noise=True, dataset_dir=args.dataset_dir, noise_config=i)

    dataset = SHDataset(noise=args.noise, dataset_dir=args.dataset_dir, noise_config=args.noise_config)
    G1,T1,G2,T2 = dataset.read_snapshots(0, bbox=(52.355, 52.365, 4.860, 4.900))

    T1['T'] = random.sample(T1['T'], k=1000)
    T2['T'] = random.sample(T2['T'], k=1000)

    plot_graph(snapshot_to_nxgraph(G1, T2['T']), figsize=(10,10))