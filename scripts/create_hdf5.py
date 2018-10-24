"""

Script to create an hdf5 version of our data (more efficient).

"""
import h5py
import os
import rasterio
import argparse
import numpy as np


def create_hdf5(data_dir, output_dir):
    """ Creates a hdf5 representation of the data.

    Args:
        data_dir - (string) path to directory containing data which has three subdirectories: s1, s2, masks
        output_dir - (string) path to output directory
    """
    hdf5_file = h5py.File(os.path.join(output_dir, 'data.hdf5'), 'a')
    # subdivide the hdf5 directory into grids and masks
    for group_name in ['s1', 's2']:
        if group_name not in hdf5_file:
            hdf5_file.create_group(f'/{group_name}')

        for filepath in os.listdir(os.path.join(data_dir, group_name)):
            filename, ext = filepath.split('.')
            if ext != 'npy': continue
            # get grid num to use as the object's file name
            grid_num = filename.split('_')[-1]
            # load in data
            data = None
            if group_name == 'masks':
                with rasterio.open(os.path.join(data_dir, group_name, filepath)) as mask_data:
                    data = mask_data.read()
            else:
                data = np.load(os.path.join(data_dir, group_name, filepath))
            # create file name
            hdf5_filename = f'/{group_name}/{grid_num}'
            hdf5_file.create_dataset(hdf5_filename, data=data, dtype='i2', chunks=True)
            print(f"Processed {os.path.join(group_name, filepath)} as {hdf5_filename}")

    hdf5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='Path to directory containing data.',
                        default='/home/data/Ghana/raster_64x64/')
    parser.add_argument('--output_dir', type=str,
                        help='Path to directory to output the hdf5 file.',
                        default='/home/data/Ghana/s1_64x64_npy/')

    args = parser.parse_args()
    create_hdf5(args.data_dir, args.output_dir)

