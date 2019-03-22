"""
Run 

`python scripts/make_32x32_grids.py --country=X` 

to update hdf5, then set grid size to 32 in constants. Assumes info in constants is correct!!

"""
import h5py
import pickle
import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from constants import *
from skimage.transform import resize as imresize
from tqdm import tqdm

def load_splits(country):
    with open(os.path.join(GRID_DIR[country], f'{country}_full_train'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(GRID_DIR[country], f'{country}_full_val'), 'rb') as f:
        val = pickle.load(f)
    with open(os.path.join(GRID_DIR[country], f'{country}_full_test'), 'rb') as f:
        test = pickle.load(f)
    return train, val, test

def save_splits(country, new_splits, old_splits):
    with open(os.path.join(GRID_DIR[country], f'{country}_full_train_32'), 'wb') as f:
        pickle.dump(new_splits['train'], f)
    with open(os.path.join(GRID_DIR[country], f'{country}_full_old_train'), 'wb') as f:
        pickle.dump(old_splits['train'], f)
    with open(os.path.join(GRID_DIR[country], f'{country}_full_val_32'), 'wb') as f:
        pickle.dump(new_splits['val'], f)
    with open(os.path.join(GRID_DIR[country], f'{country}_full_old_val'), 'wb') as f:
        pickle.dump(old_splits['val'], f)
    with open(os.path.join(GRID_DIR[country], f'{country}_full_test_32'), 'wb') as f:
        pickle.dump(new_splits['test'], f)
    with open(os.path.join(GRID_DIR[country], f'{country}_full_old_test'), 'wb') as f:
        pickle.dump(old_splits['test'], f)

def del_new_grids(country, splits):
    all_grids = set()
    with h5py.File(HDF5_PATH[country], 'a') as f:
        for category in ["labels", "cloudmasks", "s1_lengths", "s2_lengths", "s1_dates", "s2_dates", "s1", "s2", "planet_lengths", "planet_dates", "planet"]:
            for grid in f[category].keys():
                if grid in all_grids:
                    print("overlap")
                all_grids.add(grid)
                if "_" not in grid:
#                     print(grid)
                    del f[category][grid]
    print(all_grids)
    
def split_grids(country, num_pixels=32):
    train, val, test = load_splits(country)
    new_splits = {'train': [], 'val': [], 'test': []}
    old_splits = {'train': train, 'val': val, 'test': test}
    NUM_PLANET_PIXELS = 128
    groups = ['s1', 's2', 'labels', 'cloudmasks', 's1_dates', 's2_dates', 'planet', 'planet_dates']
    hdf5_file = h5py.File(HDF5_PATH[country]+  "_32", 'a')
    for group_name in groups:
        if group_name not in hdf5_file:
            hdf5_file.create_group(f'/{group_name}')
            
    with h5py.File(HDF5_PATH[country], 'a') as f:
        for split_name, split in old_splits.items():
            for grid in tqdm(split):
                print("grid: {}".format(grid))
                print(HDF5_PATH[country])
                print(f['s2'][grid])
                if grid not in f['s2']: continue # hacky fix for tanzania
            
                s1_grid = f['s1'][grid]
                s2_grid = f['s2'][grid]
                s1_dates_grid = f['s1_dates'][grid]
                s2_dates_grid = f['s2_dates'][grid]
                cloudmasks_grid = f['cloudmasks'][grid]
                label = f['labels'][grid]
                
                planet = None
                planet_dates_grid = None
                if 'planet' in f.keys():
                    planet_grid = f['planet'][grid][:, :, :, :].astype(np.float)
                    planet_grid = imresize(planet_grid, (planet_grid.shape[0], 256, 256, planet_grid.shape[3]), anti_aliasing=True, mode='reflect')
                    planet_dates_grid = f['planet_dates'][grid]

                s2_sub_grids = []
                s1_sub_grids = []
                cloudmasks_sub_grids = []
                label_sub_grids = []
                planet_sub_grids = []

                valid = set()

                for i in range(0, 2):
                    for j in range(0, 2):
                        label_sub_grid = label[i*num_pixels: (i+1)*num_pixels, j*num_pixels: (j+1)*num_pixels]

                        if np.sum(label_sub_grid) == 0: continue # if grid contains no pixels, ignore

                        label_sub_grids.append(label_sub_grid)

                        s1_sub_grid = s1_grid[:, i*num_pixels: (i+1) * num_pixels, j*num_pixels: (j+1)*num_pixels, :]
                        s1_sub_grids.append(s1_sub_grid)

                        s2_sub_grid = s2_grid[:, i * num_pixels:(i+1) * num_pixels, j * num_pixels:(j+1)*num_pixels, :]
                        s2_sub_grids.append(s2_sub_grid)

                        cloudmasks_sub_grids.append(cloudmasks_grid[i*num_pixels:(i+1)*num_pixels, j*num_pixels: (j+1)*num_pixels, :])
                        planet_sub_grids.append(planet_grid[:, i*NUM_PLANET_PIXELS:(i+1) * NUM_PLANET_PIXELS, j*NUM_PLANET_PIXELS: (j+1) * NUM_PLANET_PIXELS, :])
                
                for i in range(len(label_sub_grids)):
                    sub_grid_name = grid + "_{}".format(i)
                    new_splits[split_name].append(sub_grid_name)
                    hdf5_file.create_dataset("labels/{}".format(sub_grid_name), data=label_sub_grids[i], dtype='i2', chunks=True)
                    hdf5_file.create_dataset("cloudmasks/{}".format(sub_grid_name), data=cloudmasks_sub_grids[i], dtype='i2', chunks=True)
                    hdf5_file.create_dataset("s1_lengths/{}".format(sub_grid_name), data=s1_grid.shape[-1], dtype='i2')
                    hdf5_file.create_dataset("s2_lengths/{}".format(sub_grid_name), data=s2_grid.shape[-1], dtype='i2')
                    hdf5_file.create_dataset("s1_dates/{}".format(sub_grid_name), data=s1_dates_grid, dtype='i2')
                    hdf5_file.create_dataset("s2_dates/{}".format(sub_grid_name), data=s2_dates_grid, dtype='i2')
                    hdf5_file.create_dataset("s1/{}".format(sub_grid_name), data=s1_sub_grids[i], dtype='i2', chunks=True)
                    hdf5_file.create_dataset("s2/{}".format(sub_grid_name), data=s2_sub_grids[i], dtype='i2', chunks=True)
                    if 'planet' in f.keys():
                        hdf5_file.create_dataset("planet/{}".format(sub_grid_name), data=planet_sub_grids[i], dtype='i2', chunks=True)
                        hdf5_file.create_dataset("planet_dates/{}".format(sub_grid_name), data=planet_dates_grid, dtype='i2')
                        hdf5_file.create_dataset("planet_lengths/{}".format(sub_grid_name), data=planet_grid.shape[-1], dtype='i2')
                        
    save_splits(country, new_splits, old_splits)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', type=str, choices=['ghana', 'southsudan', 'tanzania'],
                        help='country to work on')
    args = parser.parse_args()
#     del_new_grids('ghana', [])
    split_grids(args.country)
