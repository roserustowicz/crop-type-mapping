"""

File that houses the dataset wrappers we have.

"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import h5py
import numpy as np
import os

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
    new_doys = np.asarray(list(range(0, total_days-ndays, ndays)))
    return new_arr, new_doys

class CropTypeDS(Dataset):

    def __init__(self, args, grid_path, split):
        self.model_name = args.model_name
        # open hdf5 file
        self.hdf5_filepath = args.hdf5_filepath

        with open(grid_path, "rb") as f:
            self.grid_list = list(pickle.load(f))
        
        # Rose debugging line to ignore missing S2 files for Tanzania
        #for my_item in ['004125', '004070', '003356', '004324', '004320', '004322', '003706', '004126', '003701', '003700', '003911', '003716', '004323', '004128', '003485', '004365', '004321', '003910', '004129', '003704', '003486', '003488', '003936', '003823']:
        #    if my_item in self.grid_list:
        #        self.grid_list.remove(my_item)

        self.country = args.country
        self.num_grids = len(self.grid_list)
        self.use_s1 = args.use_s1
        self.use_s2 = args.use_s2
        self.s1_agg = args.s1_agg
        self.s2_agg = args.s2_agg
        self.agg_days = args.agg_days
        self.num_classes = args.num_classes
        self.split = split
        self.apply_transforms = args.apply_transforms
        self.normalize = args.normalize
        self.sample_w_clouds = args.sample_w_clouds
        self.include_clouds = args.include_clouds
        self.include_doy = args.include_doy
        self.all_samples = args.all_samples
        ## Timeslice for FCN
        self.timeslice = args.time_slice
        self.seed = args.seed
        self.least_cloudy = args.least_cloudy
        self.s2_num_bands = args.s2_num_bands

    def __len__(self):
        return self.num_grids

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_filepath, 'r') as data:
            s1 = None
            s2 = None
            cloudmasks = None
            s1_doy = None
            s2_doy = None
            if self.use_s1:
                s1 = data['s1'][self.grid_list[idx]]

                if self.include_doy:
                    s1_doy = data['s1_dates'][self.grid_list[idx]][()]

                if self.s1_agg:
                    s1, s1_doy = split_and_aggregate(s1, s1_doy, self.agg_days, reduction='avg')
 
                #TODO: compute VH / VV from aggregated VV, VH
                #TODO: Clean this up a bit. No longer include doy/clouds if data is aggregated? 
                if self.normalize:
                    s1 = preprocess.normalization(s1, 's1', self.country)
                
                if not self.s1_agg:
                    s1, s1_doy, _ = preprocess.sample_timeseries(s1, MIN_TIMESTAMPS, s1_doy, seed=self.seed, all_samples=self.all_samples)

                # Concatenate DOY bands
                if s1_doy is not None and self.include_doy:
                    doy_stack = preprocess.doy2stack(s1_doy, s1.shape)
                    s1 = np.concatenate((s1, doy_stack), 0)

            if self.use_s2:
                s2 = data['s2'][self.grid_list[idx]]
                if self.s2_num_bands == 4:
                    s2 = s2[[0, 1, 2, 6], :, :, :] #B, G, R, NIR
                elif self.s2_num_bands == 10:
                    s2 = s2[:10, :, :, :]
                elif self.s2_num_bands != 10:
                    print('s2_num_bands must be 4 or 10')

                if self.include_doy:
                    s2_doy = data['s2_dates'][self.grid_list[idx]][()]

                if self.s2_agg:
                    s2, s2_doy = split_and_aggregate(s2, s2_doy, self.agg_days, reduction='min')

                if self.normalize:
                    s2 = preprocess.normalization(s2, 's2', self.country)
 
                #TODO: include NDVI and GCVI
                
                if self.include_clouds:
                    cloudmasks = data['cloudmasks'][self.grid_list[idx]][()]
                
                if not self.s2_agg:
                    s2, s2_doy, cloudmasks = preprocess.sample_timeseries(s2, MIN_TIMESTAMPS, s2_doy, cloud_stack=cloudmasks, seed=self.seed, least_cloudy=self.least_cloudy, sample_w_clouds=self.sample_w_clouds, all_samples=self.all_samples)

                # Concatenate cloud mask bands
                if cloudmasks is not None and self.include_clouds:
                    cloudmasks = preprocess.preprocess_clouds(cloudmasks, self.model_name, self.timeslice)
                    s2 = np.concatenate((s2, cloudmasks), 0)

                # Concatenate DOY bands
                if s2_doy is not None and self.include_doy:
                    doy_stack = preprocess.doy2stack(s2_doy, s2.shape)
                    s2 = np.concatenate((s2, doy_stack), 0)

            transform = self.apply_transforms and np.random.random() < .5 and self.split == 'train'
            rot = np.random.randint(0, 4)
            grid = preprocess.concat_s1_s2(s1, s2)
            grid = preprocess.preprocess_grid(grid, self.model_name, self.timeslice, transform, rot)
            label = data['labels'][self.grid_list[idx]][()]
            label = preprocess.preprocess_label(label, self.model_name, self.num_classes, transform, rot) 
        
        if cloudmasks is None:
            cloudmasks = False
        return grid, label, cloudmasks
      
class GridDataLoader(DataLoader):

    def __init__(self, args, grid_path, split):
        dataset = CropTypeDS(args, grid_path, split)
        super(GridDataLoader, self).__init__(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=args.shuffle,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

def get_dataloaders(grid_dir, country, dataset, args):
    dataloaders = {}
    for split in SPLITS:
        grid_path = os.path.join(grid_dir, f"{country}_{dataset}_{split}")
        dataloaders[split] = GridDataLoader(args, grid_path, split)

    return dataloaders
