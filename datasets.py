"""

File that houses the dataset wrappers we have.

"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import h5py
import numpy as np
import os
from preprocess import *
from constants import *
from random import shuffle

class CropTypeDS(Dataset):

    def __init__(self, args, grid_path, split):
        self.model_name = args.model_name
        # open hdf5 file
        self.hdf5_filepath = args.hdf5_filepath

        with open(grid_path, "rb") as f:
            self.grid_list = list(pickle.load(f))

        self.num_grids = len(self.grid_list)
        self.use_s1 = args.use_s1
        self.use_s2 = args.use_s2
        self.num_classes = args.num_classes
        self.split = split
        self.apply_transforms = args.apply_transforms
        ## Timeslice for FCN
        self.timeslice = args.time_slice

    def __len__(self):
        return self.num_grids

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_filepath, 'r') as data:
            s1 = None
            s2 = None
            if self.use_s1:
                s1 = data['s1'][self.grid_list[idx]][:2, :, :]
                s1 = normalization(s1, 's1')
            if self.use_s2:
                s2 = data['s2'][self.grid_list[idx]][()]
                s2 = normalization(s2, 's2')
            
            transform = self.apply_transforms and np.random.random() < .5 and self.split == 'train'
            rot = np.random.randint(0, 4)
            grid = concat_s1_s2(s1, s2)
            grid = preprocess_grid(grid, self.model_name, self.timeslice, transform, rot)
            label = data['labels'][self.grid_list[idx]][()]
            label = preprocess_label(label, self.model_name, self.num_classes, transform, rot) 

        return grid, label
      
class GridDataLoader(DataLoader):

    def __init__(self, args, grid_path, split):
        dataset = CropTypeDS(args, grid_path, split)
        super(GridDataLoader, self).__init__(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=args.shuffle,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=truncateToSmallestLength)


def get_dataloaders(grid_dir, country, dataset, args):
    dataloaders = {}
    for split in SPLITS:
        grid_path = os.path.join(grid_dir, f"{country}_{dataset}_{split}")
        dataloaders[split] = GridDataLoader(args, grid_path, split)

    return dataloaders
