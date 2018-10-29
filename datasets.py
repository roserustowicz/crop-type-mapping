"""

File that houses the dataset wrappers we have.

"""

from keras.utils import Sequence
from random import shuffle
import pickle
import h5py

class CropTypeSequence(Sequence):

    def __init__(self, model_name, hdf5_filepath, grid_path, batch_size, use_s1, use_s2):
        self.model_name = model_name
        self.batch_size = batch_size
        # open hdf5 file
        with h5py.File(hdf5_filepath, 'r') as hdf5_file:
            self.data = hdf5_file

        with open(grid_path, "rb") as f:
            self.grid_list = list(pickle.load(f))

        self.num_grids = len(self.grid_list)
        self.use_s1 = use_s1
        self.use_s2 = use_s2

    def __len__(self):
        return int(self.num_grids / self.batch_size)

    def __getitem__(self, idx):
        grid_nums = self.grid_list[idx:idx + self.batch_size]
        batch_X = []
        batch_Y = []

        # iterate through and append these together
        for grid_num in grid_nums:
            if self.use_s1:
                s1 = self.data[f'/s1/{grid_num}']
            if self.use_s2:
                s2 = self.data[f'/s2/{grid_num}']

            batch_X.append(preprocess_grid(np.concatenate(s1, s2), self.model_name))
            batch_Y.append(self.data[f'/labels/{grid_num}'])

        return np.array(batch_X), np.array(batch_Y)
        # get self.batch_size examples and return



