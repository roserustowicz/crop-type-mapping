"""

File that houses the dataset wrappers we have.

"""

import torch
import torch.utils.data import Dataset, Dataloader
from random import shuffle
import pickle
import h5py
from preprocess import *
import numpy as np

class CropTypeDS(Dataset):

    def __init__(self, args, grid_path):
        self.model_name = args.model_name
        # open hdf5 file
        self.hdf5_filepath = args.hdf5_filepath

        with open(grid_path, "rb") as f:
            self.grid_list = list(pickle.load(f))
        self.num_grids = len(self.grid_list)
        self.use_s1 = args.use_s1
        self.use_s2 = args.use_s2
        self.num_classes = args.num_classes

    def __len__(self):
        return self.num_grids

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_filepath, 'r') as data:
            s1 = None
            s2 = None
            if self.use_s1:
                s1 = data['s1'][grid_num][:2, :, :]
            if self.use_s2:
                s2 = data['s2'][grid_num][()]
            
            grid = concat_s1_s2(s1, s2)
            grid = preprocess_grid(grid, self.model_name)
            label = data['labels'][grid_num][()]
            label = preprocess_label(label, self.model_name, self.num_classes) 
    
        return grid, label

class GridDataLoader(DataLoader):

    def __init__(self, args, grid_path):
        dataset = CropTypeDS(args, grid_path)
        super(GridDataLoader, self).__init__(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=args.shuffle,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=padToEqualLength)

