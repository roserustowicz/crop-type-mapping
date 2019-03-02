"""

File that houses the dataset wrappers we have.

"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
import h5py
import numpy as np
import os

from skimage.transform import resize as imresize

import preprocess
from constants import *
from random import shuffle


def get_Xy(dl):
    """ 
    Constructs data (X) and labels (y) for pixel-based methods. 
    Args: 
      dl - pytorch data loader
    Returns: 
      X - matrix of data of shape [examples, features] 
      y - vector of labels of shape [examples,] 
    """
    # Populate data and labels of classes we care about
    X = []
    y = []
    for inputs, targets, cloudmasks in dl:
        X, y = get_Xy_batch(inputs, targets, X, y)
    X = np.vstack(X)
    y = np.squeeze(np.vstack(y))

    # shuffle
    indices = np.array(list(range(y.shape[0])))
    indices = np.random.shuffle(indices)
    X = np.squeeze(X[indices, :])
    y = np.squeeze(y[indices])
    return X, y

def get_Xy_batch(inputs, targets, X, y):
    """ 
    Constructs necessary pixel array for pixel based methods. The 
    function takes one case of inputs, targets each time it is called
    and builds up X, y as the dataloader goes through all batches 
    """
    # For each input example and corresponding target,
    for ex_idx in range(inputs.shape[0]):
        for crop_idx in range(targets.shape[1]):
            cur_inputs = np.transpose(np.reshape(inputs[ex_idx, :, :, :, :], (-1, 64*64)), (1, 0))
            cur_targets = np.squeeze(np.reshape(targets[ex_idx, crop_idx, :, :], (-1, 64*64)))
            # Index pixels of desired crop
            valid_inputs = cur_inputs[cur_targets == 1, :]
            if valid_inputs.shape[0] == 0:
                pass
            else:
                # Append valid examples to X
                X.append(valid_inputs)
                # Append valid labels to y
                labels = torch.ones((int(torch.sum(cur_targets).numpy()), 1)) * crop_idx
                y.append(labels)
    return X, y 

def split_and_aggregate(arr, doys, ndays, reduction='avg'):
    """
    Aggregates an array along the time dimension, grouping by every ndays
    
    Args: 
      arr - array of images of dimensions
      doys - vector / list of days of year associated with images stored in arr
      ndays - number of days to aggregate together
    """
    total_days = 364
    
    # Get index of observations corresponding to time bins
    # // ndays gives index from 0 to total_days // ndays
    obs_idxs = list(doys.astype(int) // ndays)
    split_idxs = []
    # Get the locations within the obs_idxs vector that change
    # to a new time bin (will be used for splitting)
    for idx in range(1, int(total_days//ndays)):
        if idx in obs_idxs:
            split_idxs.append(obs_idxs.index(idx))
        # if no observations are in bin 1, append the 0 index
        # to indicate that the bin is empty
        elif idx == 1:
            split_idxs.append(0)
        # if observations are not in the bin, use the same index 
        #  as the previous bin, to indicate the bin is empty
        else:
            split_idxs.append(prev_split_idx)
        prev_split_idx = split_idxs[-1]
        
    # split the array according to the indices of the time bins
    split_arr = np.split(arr, split_idxs, axis=3)
 
    # For each bin, create a composite according to a "reduction"
    #  and append for concatenation into a new reduced array
    composites = []
    for a in split_arr:
        if a.shape[3] == 0:
            a = np.zeros((a.shape[0], a.shape[1], a.shape[2], 1))
        if reduction == 'avg':
            cur_agg = np.mean(a, axis=3)
        elif reduction == 'min':
            cur_agg = np.min(a, axis=3)
        elif reduction == 'max':
            cur_agg = np.max(a, axis=3)
        elif reduction == 'median':
            cur_agg = np.median(a, axis=3)
        composites.append(np.expand_dims(cur_agg, axis=3))

    new_arr = np.concatenate(composites, axis=3)
    new_doys = np.asarray(list(range(0, total_days-ndays+1, ndays)))
    return new_arr, new_doys

class CropTypeDS(Dataset):

    def __init__(self, args, grid_path, split):
        self.model_name = args.model_name
        # open hdf5 file
        self.hdf5_filepath = HDF5_PATH[args.country]

        with open(grid_path, "rb") as f:
            self.grid_list = list(pickle.load(f))
        
        # Rose debugging line to ignore missing S2 files for Tanzania
        #for my_item in ['004125', '004070', '003356', '004324', '004320', '004322', '003706', '004126', '003701', '003700', '003911', '003716', '004323', '004128', '003485', '004365', '004321', '003910', '004129', '003704', '003486', '003488', '003936', '003823']:
        #    if my_item in self.grid_list:
        #        self.grid_list.remove(my_item)

        self.country = args.country
        self.num_grids = len(self.grid_list)
        self.grid_size = GRID_SIZE[args.country]
        self.agg_days = args.agg_days
        # s1 args
        self.use_s1 = args.use_s1
        self.s1_agg = args.s1_agg
        # s2 args 
        self.use_s2 = args.use_s2
        self.s2_agg = args.s2_agg
        # planet args
        self.resize_planet = args.resize_planet
        self.use_planet = args.use_planet
        self.planet_agg = args.planet_agg
        
        self.num_classes = NUM_CLASSES[args.country]
        self.split = split
        self.apply_transforms = args.apply_transforms
        self.normalize = args.normalize
        self.sample_w_clouds = args.sample_w_clouds
        self.include_clouds = args.include_clouds
        self.include_doy = args.include_doy
        self.include_indices = args.include_indices
        self.num_timesteps = args.num_timesteps
        self.all_samples = args.all_samples
        
        ## Timeslice for FCN
        self.timeslice = args.time_slice
        self.seed = args.seed
        self.least_cloudy = args.least_cloudy
        self.s2_num_bands = args.s2_num_bands
        
        with h5py.File(self.hdf5_filepath, 'r') as data:
            self.s1_lengths = data['s1_lengths']
            self.s2_lengths = data['s2_lengths']

    def __len__(self):
        return self.num_grids

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_filepath, 'r') as data:
            sat_properties = { 's1': {'data': None, 'doy': None, 'use': self.use_s1, 'agg': self.s1_agg,
                                      'agg_reduction': 'avg', 'cloudmasks': None },
                               's2': {'data': None, 'doy': None, 'use': self.use_s2, 'agg': self.s2_agg,
                                      'agg_reduction': 'min', 'cloudmasks': None, 'num_bands': self.s2_num_bands },
                               'planet': {'data': None, 'doy': None, 'use': self.use_planet, 'agg': self.planet_agg,
                                          'agg_reduction': 'median', 'cloudmasks': None } }
  
            for sat in ['s1', 's2', 'planet']:
                sat_properties = self.setup_data(data, idx, sat, sat_properties)

            transform = self.apply_transforms and np.random.random() < .5 and self.split == 'train'
            rot = np.random.randint(0, 4)
            grid = preprocess.concat_s1_s2_planet(sat_properties['s1']['data'],
                                                  sat_properties['s2']['data'], 
                                                  sat_properties['planet']['data'])
            grid = preprocess.preprocess_grid(grid, self.model_name, self.timeslice, transform, rot)
            label = data['labels'][self.grid_list[idx]][()]
            label = preprocess.preprocess_label(label, self.model_name, self.num_classes, transform, rot) 
        
        if sat_properties['s2']['cloudmasks'] is None:
            cloudmasks = False
        else:
            cloudmasks = sat_properties['s2']['cloudmasks']

#         if self.split == 'train':
#             x_start = np.random.randint(0, 32)
#             y_start = np.random.randint(0, 32)
# #             while torch.sum(label[:, x_start:x_start+32, y_start:y_start+32]) == 0:
# #                 x_start = np.random.randint(0, 32)
# #                 y_start = np.random.randint(0, 32)
#             label = label[:, x_start:x_start+32, y_start:y_start+32]
#             grid = grid[:, :, x_start:x_start+32, y_start:y_start+32]

#             if cloudmasks is not None:
#                 cloudmasks = cloudmasks[:, x_start:x_start+32, y_start:y_start+32, :]
        return grid, label, cloudmasks
    

    def setup_data(self, data, idx, sat, sat_properties):
        if sat_properties[sat]['use']:
            sat_properties[sat]['data'] = data[sat][self.grid_list[idx]]       
                    
            if sat in ['planet']:
                sat_properties[sat]['data'] = sat_properties[sat]['data'][:, :, :, :].astype(np.double)  
                if self.resize_planet:
                    sat_properties[sat]['data'] = imresize(sat_properties[sat]['data'], 
                                                           (sat_properties[sat]['data'].shape[0], self.grid_size, self.grid_size, sat_properties[sat]['data'].shape[3]), 
                                                           anti_aliasing=True, mode='reflect')
                else:
                    # upsample to 256 x 256 to fit into model
                    sat_properties[sat]['data'] = imresize(sat_properties[sat]['data'],
                                                           (sat_properties[sat]['data'].shape[0], 256, 256, sat_properties[sat]['data'].shape[3]),
                                                           anti_aliasing=True, mode='reflect')

            if sat in ['s2']:
                if sat_properties[sat]['num_bands'] == 4:
                    sat_properties[sat]['data'] = sat_properties[sat]['data'][[0, 1, 2, 6], :, :, :] #B, G, R, NIR
                elif sat_properties[sat]['num_bands'] == 10:
                    sat_properties[sat]['data'] = sat_properties[sat]['data'][:10, :, :, :]
                elif sat_properties[sat]['num_bands'] != 10:
                    raise ValueError('s2_num_bands must be 4 or 10')

                if self.include_clouds:
                    sat_properties[sat]['cloudmasks'] = data['cloudmasks'][self.grid_list[idx]][()]

            if self.include_doy:
                sat_properties[sat]['doy'] = data[f'{sat}_dates'][self.grid_list[idx]][()]
 
            if sat_properties[sat]['agg']:
                sat_properties[sat]['data'], sat_properties[sat]['doy'] = split_and_aggregate(sat_properties[sat]['data'], 
                                                                                          sat_properties[sat]['doy'],
                                                                                          self.agg_days, 
                                                                                          reduction=sat_properties[sat]['agg_reduction'])
                
                # replace the VH/VV band with a cleaner band after aggregation??
                if sat in ['s1']:
                    sat_properties[sat]['data'][2,:,:,:] = sat_properties[sat]['data'][1,:,:,:] / sat_properties[sat]['data'][0,:,:,:]

            #TODO: include NDVI and GCVI for s2 and planet, calculate before normalization
            if sat in ['s2', 'planet'] and self.include_indices:
                if (sat in ['s2'] and sat_properties[sat]['num_bands'] == 4) or (sat in ['planet']):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndvi = (sat_properties[sat]['data'][3, :, :, :] - sat_properties[sat]['data'][2, :, :, :]) / (sat_properties[sat]['data'][3, :, :, :] + sat_properties[sat]['data'][2, :, :, :])
                        gcvi = (sat_properties[sat]['data'][3, :, :, :] / sat_properties[sat]['data'][1, :, :, :]) - 1 

                    ndvi[(sat_properties[sat]['data'][3, :, :, :] + sat_properties[sat]['data'][2, :, :, :]) == 0] = 0
                    gcvi[sat_properties[sat]['data'][1, :, :, :] == 0] = 0

                elif sat_properties[sat]['num_bands'] == 10:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndvi = (sat_properties[sat]['data'][6, :, :, :] - sat_properties[sat]['data'][2, :, :, :]) / (sat_properties[sat]['data'][6, :, :, :] + sat_properties[sat]['data'][2, :, :, :])
                        gcvi = (sat_properties[sat]['data'][6, :, :, :] / sat_properties[sat]['data'][1, :, :, :]) - 1

                    ndvi[(sat_properties[sat]['data'][6, :, :, :] + sat_properties[sat]['data'][2, :, :, :]) == 0] = 0
                    gcvi[sat_properties[sat]['data'][1, :, :, :] == 0] = 0

            #TODO: Clean this up a bit. No longer include doy/clouds if data is aggregated? 
                
            if self.normalize:
                sat_properties[sat]['data'] = preprocess.normalization(sat_properties[sat]['data'], sat, self.country)
            
            # Concatenate vegetation indices after normalization, before temporal sample
            if sat in ['planet', 's2'] and self.include_indices:
                #print('data: ' , sat_properties[sat]['data'].shape)
                #print('ndvi: ', np.expand_dims(ndvi, axis=0).shape)
                sat_properties[sat]['data'] = np.concatenate(( sat_properties[sat]['data'], np.expand_dims(ndvi, axis=0)), 0)
                sat_properties[sat]['data'] = np.concatenate(( sat_properties[sat]['data'], np.expand_dims(gcvi, axis=0)), 0)

            if not sat_properties[sat]['agg']:
                sat_properties[sat]['data'], sat_properties[sat]['doy'], sat_properties[sat]['cloudmasks'] = preprocess.sample_timeseries(sat_properties[sat]['data'],
                                                                                                               self.num_timesteps, sat_properties[sat]['doy'],
                                                                                                               cloud_stack = sat_properties[sat]['cloudmasks'],
                                                                                                               seed=self.seed, least_cloudy=self.least_cloudy,
                                                                                                               sample_w_clouds=self.sample_w_clouds, 
                                                                                                               all_samples=self.all_samples)

            # Concatenate cloud mask bands
            if sat_properties[sat]['cloudmasks'] is not None and self.include_clouds:
                sat_properties[sat]['cloudmasks'] = preprocess.preprocess_clouds(sat_properties[sat]['cloudmasks'], self.model_name, self.timeslice)
                sat_properties[sat]['data'] = np.concatenate(( sat_properties[sat]['data'], sat_properties[sat]['cloudmasks']), 0)

            # Concatenate doy bands
            if sat_properties[sat]['doy'] is not None and self.include_doy:
                doy_stack = preprocess.doy2stack(sat_properties[sat]['doy'], sat_properties[sat]['data'].shape)
                sat_properties[sat]['data'] = np.concatenate((sat_properties[sat]['data'], doy_stack), 0)
        return sat_properties

    

class CropTypeBatchSampler(Sampler):
    """
        Groups sequences of similiar length into the same batch to prevent unnecessary computation.
    """
    def __init__(self, dataset, batch_size):
        super(CropTypeBatchSampler, self).__init__(dataset)
        batches = []
        count = 1
        cur_list = []
        
        buckets = defaultdict(list)
        
        grid_lengths = dataset.grid_lengths
        
        # shuffle dataset
        
        # create buckets for grid lengths (maybe %10) 
        
        # for each grid, add it to the a0ppropriate bucket 
        
        # if a bucket is too large, trim to max_batch_size length
        
        # later need to pad to the same length
        
        for i in range(len(dataset)):
            if count % batch_size != 0:
                cur_list.append(i)
            else:
                batches.append(cur_list)
                cur_list = []
            count += 1
#           for i in range(len(dataset)):
#             grid, label, cloudmasks = dataset[i]
#             batches.append(dataset[i])
        print(len(batches))
        self.batches = batches
        
    def __iter__(self):
        for b in self.batches:
            yield(b)
        
    def __len__(self):
        return len(self.batches)



class GridDataLoader(DataLoader):

    def __init__(self, args, grid_path, split):
        dataset = CropTypeDS(args, grid_path, split)
        super(GridDataLoader, self).__init__(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=args.shuffle,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

def get_dataloaders(country, dataset, args):
    dataloaders = {}
    for split in SPLITS:
        grid_path = os.path.join(GRID_DIR[country], f"{country}_{dataset}_{split}")
        dataloaders[split] = GridDataLoader(args, grid_path, split)

    return dataloaders
