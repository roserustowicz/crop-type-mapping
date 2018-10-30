"""

File that houses the dataset wrappers we have.

"""

from keras.utils import Sequence
from random import shuffle
import pickle
import h5py
from preprocess import *
import numpy as np

class CropTypeSequence(Sequence):

    def __init__(self, model_name, hdf5_filepath, grid_path, batch_size, use_s1, use_s2, num_classes):
        self.model_name = model_name
        self.batch_size = batch_size
        # open hdf5 file
        self.hdf5_filepath = hdf5_filepath

        with open(grid_path, "rb") as f:
            self.grid_list = list(pickle.load(f))
        self.num_grids = len(self.grid_list)
        self.use_s1 = use_s1
        self.use_s2 = use_s2
        self.num_classes = num_classes

    def __len__(self):
        return int(self.num_grids / self.batch_size)

    def __getitem__(self, idx):
        grid_nums = self.grid_list[idx:idx + self.batch_size]
        batch_X = []
        batch_Y = []

        with h5py.File(self.hdf5_filepath, 'r') as data:
            # iterate through and append these together
            for grid_num in grid_nums:
                s1 = None
                s2 = None
                if self.use_s1:
                    s1 = data['s1'][grid_num][()]
                if self.use_s2:
                    s2 = data['s2'][grid_num][()]
                
                grid = concat_s1_s2(s1, s2)
                grid = preprocess_grid(grid, self.model_name)
                batch_X.append(grid)
                label = data['labels'][grid_num][()]
                label = preprocess_label(label, self.model_name, self.num_classes) 
                batch_Y.append(label)

        batch_X = padToEqualLength(batch_X)
        return np.array(batch_X), np.array(batch_Y)



