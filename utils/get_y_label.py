import os
import numpy as np
import pickle
import rasterio
import pandas as pd
import argparse
"""
    Get y label for different set small/full, different type train/val/test
    
    Args:
      home - (str) the base directory of data

      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'

      data_set - (str) balanced 'small' or unbalanced 'full' dataset

      data_type - (str) 'train'/'val'/'test'

      satellite - (str) satellite to use 's1' 's2' 's1_s2'

      ylabel_dir - (str) dir to save ylabel

    Output: 
    ylabel_dir/..

    save as row*col*grid_nums 3D array

"""

def get_y_label(home, country, data_set, data_type, satellite, ylabel_dir):
    # gridded data
    gridded_dir = os.path.join(home, data_set, data_type, satellite)
    gridded_fnames = [f for f in os.listdir(gridded_dir) if (f.endswith('.npy')) and ('mask' not in f) ]
    grid_nums = [f.split('_')[-1].replace('.npy', '') for f in gridded_fnames]
    gridded_fnames = [gridded_fnames[ID] for ID in np.argsort(grid_nums)]
    grid_nums = np.array([grid_nums[ID] for ID in np.argsort(grid_nums)])

    # Match the Mask
    mask_dir = os.path.join(home, country, 'raster_64x64_npy')
    mask_fnames = [f for f in os.listdir(mask_dir) if f.endswith('.npy')]
    mask_ids = [f.split('_')[-1].replace('.npy', '') for f in mask_fnames]
    mask_fnames = [mask_fnames[ID] for ID in np.argsort(mask_ids)]
    mask_ids = np.array([mask_ids[ID] for ID in np.argsort(mask_ids)])

    # Find the corresponded ID
    grid_nums_idx = [np.where([grid_num==mask_id for mask_id in mask_ids])[0][0] for grid_num in grid_nums]
    mask_corresponded_fnames = [mask_fnames[grid_num_idx] for grid_num_idx in grid_nums_idx]

    # Geom_ID Mask Array
    mask_array = np.zeros((len(grid_nums),64,64))

    for i in range(len(grid_nums)): 
        fname = os.path.join(mask_dir,mask_corresponded_fnames[i])
        # Save Mask as one big array
        mask_array[i,:,:] = np.load(fname)[0:64,0:64]

    output_fname = "_".join([data_set, data_type, satellite, 'croptypemask', 'g'+str(len(grid_nums)),'r64', 'c64'+'.npy'])

    np.save(os.path.join(ylabel_dir,output_fname), mask_array)
    

if __name__ == '__main__':
    
    # Construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-hm", "--home", required=False, default='/home/data/')
    arg_parser.add_argument("-c", "--country", required=False, default='Ghana')
    arg_parser.add_argument("-ds", "--dataset", required=False, default='full')
    arg_parser.add_argument("-dt", "--datatype", required=False, default='train')
    arg_parser.add_argument("-s", "--satellite", required=False, default='s1')
    arg_parser.add_argument("-ydir", "--ylabel_dir", required=False, default='/home/data/ylabel')
    args = vars(arg_parser.parse_args())
    
    home = args['home']
    country = args['country']
    data_set = args['dataset']
    data_type = args['datatype']
    satellite = args['satellite']
    ylabel_dir = args['ylabel_dir']
        
    get_y_label(home, country, data_set, data_type, satellite, ylabel_dir)