"""

File that houses the dataset wrappers we have.

"""

from keras.utils import Sequence
from random import shuffle

class CropTypeSequence(Sequence):

    def __init__(self, model_name, hdf5_filepath, batch_size, shuffle, use_s1, use_s2):
        self.model_name = model_name
        self.batch_size = batch_size
        # open hdf5 file
        self.shuffle = shuffle
        with h5py.File(hdf5_filepath, 'r') as hdf5_file:
            self.data = hdf5_file

        self.grid_list = self.data['/masks'].keys()
        self.num_grids = len(self.grid_list)
        self.use_s1 = use_s1
        self.use_s2 = use_s2

    def __len__(self):
        return num_examples / self.batch_size

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

            batch_X.append(preprocess_grid(np.concatenate(s1, s2), model.name)
            batch_Y.append(self.data[f'/masks/{grid_num}'])

        return np.array(batch_X), np.array(batch_Y)
        # get self.batch_size examples and return


    def on_epoch_end(self):
        """ Randomizes ordering for next epoch. """
        if self.shuffle == True:
            shuffle(self.grid_list)



