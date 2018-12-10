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

class CropTypeDS(Dataset):

    def __init__(self, args, grid_path, split):
        self.model_name = args.model_name
        # open hdf5 file
        self.hdf5_filepath = args.hdf5_filepath

        with open(grid_path, "rb") as f:
            self.grid_list = list(pickle.load(f))
            self.grid_list = np.random.choice(self.grid_list, int(len(self.grid_list) * args.percent_of_dataset))

        self.num_grids = len(self.grid_list)
        self.use_s1 = args.use_s1
        self.use_s2 = args.use_s2
        self.num_classes = args.num_classes
        self.split = split
        self.apply_transforms = args.apply_transforms
        self.sample_w_clouds = args.sample_w_clouds
        self.include_clouds = args.include_clouds
        self.include_doy = args.include_doy
        ## Timeslice for FCN
        self.timeslice = args.time_slice
        self.seed = args.seed
        self.least_cloudy = args.least_cloudy

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
                s1 = preprocess.normalization(s1, 's1')
                if self.include_doy:
                    s1_doy = data['s1_dates'][self.grid_list[idx]][()]
                s1, s1_doy, _ = preprocess.sample_timeseries(s1, MIN_TIMESTAMPS, s1_doy, seed=self.seed)
                # Concatenate DOY bands
                if s1_doy is not None and self.include_doy:
                    doy_stack = preprocess.doy2stack(s1_doy, s1.shape)
                    s1 = np.concatenate((s1, doy_stack), 0)

            if self.use_s2:
                s2 = data['s2'][self.grid_list[idx]][()]
                s2 = preprocess.normalization(s2, 's2')
                if self.include_clouds:
                    cloudmasks = data['cloudmasks'][self.grid_list[idx]][()]
                if self.include_doy:
                    s2_doy = data['s2_dates'][self.grid_list[idx]][()]
                s2, s2_doy, cloudmasks = preprocess.sample_timeseries(s2, MIN_TIMESTAMPS, s2_doy, cloud_stack=cloudmasks, seed=self.seed, least_cloudy=self.least_cloudy, sample_w_clouds=self.sample_w_clouds)

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
