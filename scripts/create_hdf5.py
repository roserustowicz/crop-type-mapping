"""

Script to create an hdf5 version of our data (more efficient).

"""
import h5py
import os
import rasterio
import argparse
import numpy as np
import json
import sys
import pickle

sys.path.insert(0, '../')
import util
from pprint import pprint
from tqdm import tqdm
from skimage.transform import resize as imresize

def get_grid_num(filename, ext, group_name):
    if ext == 'json' and group_name in ['s1_dates', 's2_dates']:
        grid_num = filename.split('_')[-1]
    elif ext == 'json' and group_name in ['planet_dates']:
        grid_num = filename.split('_')[-2]
    elif ext == 'npy' and group_name in ['s1', 's2', 'labels', 'planet'] and 'mask' not in filename:
        grid_num = filename.split('_')[-1] if group_name not in ['labels', 'planet'] else filename.split('_')[-2]
    elif ext == 'npy' and group_name == 'cloudmasks' and 'mask' in filename:
        grid_num = filename.split('_')[-2]
    else:
        grid_num = None
    return grid_num


def load_splits(data_dir, country):
    with open(os.path.join(data_dir, f'{country}_full_train'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(data_dir, f'{country}_full_val'), 'rb') as f:
        val = pickle.load(f)
    with open(os.path.join(data_dir, f'{country}_full_test'), 'rb') as f:
        test = pickle.load(f)
    return train, val, test

def save_splits(country, data_dir, new_splits, suffix):
    with open(os.path.join(data_dir, f'{country}_full_train_' + suffix), 'wb') as f:
        pickle.dump(new_splits['train'], f)
    with open(os.path.join(data_dir, f'{country}_full_val_' + suffix), 'wb') as f:
        pickle.dump(new_splits['val'], f)
    with open(os.path.join(data_dir, f'{country}_full_test_' + suffix), 'wb') as f:
        pickle.dump(new_splits['test'], f)

def create_hdf5(args, groups=None):
    """ Creates a hdf5 representation of the data.

    Args:
        data_dir - (string) path to directory containing data which has three subdirectories: s1, s2, masks
        output_dir - (string) path to output directory
    """

    data_dir = args.data_dir
    output_dir = args.output_dir
    country = args.country
    use_planet = args.use_planet
    out_fname = args.out_fname
    num_pixels = args.num_pixels
    num_planet_pixels = num_pixels * 4
    assert 64 % num_pixels == 0, "NUM PIXELS SHOULD DIVIDE 64 EVENLY"
    train, val, test = load_splits(data_dir, country)
    new_splits = {'train': [], 'val': [], 'test': []}
    old_splits = {'train': train, 'val': val, 'test': test}
    all_grids = set(old_splits['train'] + old_splits['val'] + old_splits['test'])
    all_new_grids = set()

    if groups is None:
        if country in ['germany']:
            groups = ['s2', 'labels', 's2_dates']
        else:
            # extremely important to iterate through labels first, allows us to know what 
            # grids to use later
            groups =['labels', 's1', 's2', 'cloudmasks', 's1_dates', 's2_dates']
        if use_planet:
            groups += ['planet', 'planet_dates']

    hdf5_file = h5py.File(os.path.join(output_dir, out_fname + "_{}".format(num_pixels)), 'a')
    # subdivide the hdf5 directory into grids and masks
    for group_name in groups:
        if group_name not in hdf5_file:
            hdf5_file.create_group(f'/{group_name}')

        actual_dir_name = None
        if group_name in ['s1', 's1_dates']:
            actual_dir_name = "s1_npy"
        elif group_name in ['s2', 's2_dates', 'cloudmasks']:
            actual_dir_name = "s2_npy"
        elif group_name in ['planet', 'planet_dates']:
            actual_dir_name = "planet_npy"
        elif group_name == 'labels':
            actual_dir_name = "raster_npy"

        for filepath in tqdm(os.listdir(os.path.join(data_dir, actual_dir_name))):
            print('filepath: ', filepath)
            filename, ext = filepath.split('.')
            print('fname: ', filename)
            print('ext: ', ext)
            print('group name: ', group_name)
            # get grid num to use as the object's file name
            grid_num = get_grid_num(filename, ext, group_name)
            print('grid_num: ', grid_num)
            if grid_num is None or grid_num not in all_grids: continue
            # load in data
            if ext == 'npy':
                data = np.load(os.path.join(data_dir, actual_dir_name, filepath))
            elif ext == 'json':
                # open json of dates
                with open(os.path.join(data_dir, actual_dir_name, filepath)) as f:
                    dates = json.load(f)['dates']
                data = util.dates2doy(dates)
            dtype = 'i2' if group_name not in ['s1', 's2', 'planet'] else 'f8' 
            for i in range(0, 64 // num_pixels):
                for j in range(0, 64 // num_pixels):
                    new_grid_name = grid_num + "_{}_{}".format(i, j)
                    print('new grid name: ', new_grid_name)
                    hdf5_filename = f'/{group_name}/{new_grid_name}'
                    if 'dates' not in group_name:
                        if group_name == "planet":
                            data = data.astype(np.float)
                            data = imresize(data, (data.shape[0], 256, 256, data.shape[3]), anti_aliasing=True, mode='reflect')
                            sub_grid = data[i*num_planet_pixels: (i+1)*num_planet_pixels, j*num_planet_pixels: (j+1) * num_planet_pixels]
                        else:
                            sub_grid = data[i*num_pixels: (i+1)*num_pixels, j*num_pixels: (j+1)*num_pixels]
                    else:
                        sub_grid = data

                    if group_name == 'labels':
                        if np.sum(sub_grid) > 0:
                            all_new_grids.add(new_grid_name)
                            found = False
                            for split_name in old_splits:
                                if grid_num in old_splits[split_name]:
                                    new_splits[split_name].append(new_grid_name)
                                    found = True
                                    break
                            assert found, "Grid num {} not found in any split".format(grid_num)
                            hdf5_file.create_dataset(hdf5_filename, data=data, dtype='i1', chunks=True)
                            print(f"Processed {os.path.join(group_name, filepath)} as {hdf5_filename}")
                    else:
                        if new_grid_name in all_new_grids:
                            print(f"Processed {os.path.join(group_name, filepath)} as {hdf5_filename} with dtype: {dtype}")
                            hdf5_file.create_dataset(hdf5_filename, data=data, chunks=True) 
                            if group_name in ['s1', 's2', 'planet']:
                                _, _, _, l = sub_grid.shape
                                length_group = group_name + "_length"
                                print(f"Processed {length_group}/{new_grid_name} with length {l}")
                                hdf5_file.create_dataset(f'/{length_group}/{new_grid_name}', data=l, dtype='i2')
    pprint(new_splits)
    save_splits(country, data_dir, new_splits, suffix=str(num_pixels))
    hdf5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='Path to directory containing data.',
                        default='/home/roserustowicz/croptype_data_local/data/ghana/')
    parser.add_argument('--output_dir', type=str,
                        help='Path to directory to output the hdf5 file.',
                        default='/home/roserustowicz/croptype_data_local/data/ghana/')
    parser.add_argument('--country', type=str,
                        help='Country to output the hdf5 file for.',
                        default='ghana')
    parser.add_argument('--use_planet', type=util.str2bool, default=False,
                        help='Include Planet in hdf5 file')
    parser.add_argument('--num_pixels', type=int, default=32)
    parser.add_argument('--out_fname', type=str, default='data.hdf5')
    args = parser.parse_args()

    groups = ['labels', 's2', 'cloudmasks', 's2_dates'] #None #['planet', 'planet_dates', 'labels']

    create_hdf5(args, groups)

