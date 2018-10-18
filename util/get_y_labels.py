import os
import numpy as np
import pickle
import rasterio
import pandas as pd
import argparse


def get_y_labels(home, country, data_set, data_type, satellite, csv_source):
    """
    Creates a croptype array of dimensions grids x rows x columns for each country, each dataset: full/small,
    each data_type: train/val/test
    source combination i.e. Ghana + s1, Ghana + s2, etc. 
    """
    
    # gridded data
    gridded_dir = os.path.join(home, data_set, data_type, satellite)
    gridded_fnames = [f for f in os.listdir(gridded_dir) if f.endswith('.npy')]
    grid_nums = [f.split('_')[-1].replace('.npy', '') for f in gridded_fnames]

    # Match the Mask
    mask_dir = os.path.join(home, country, 'raster_64x64')
    mask_fnames = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    mask_ids = [f.split('_')[-1].replace('.tif', '') for f in mask_fnames]

    # Find the corresponded ID
    grid_nums_idx = [np.where([grid_num==mask_id for mask_id in mask_ids])[0][0] for grid_num in grid_nums]
    mask_corresponded_fnames = [mask_fnames[grid_num_idx] for grid_num_idx in grid_nums_idx]

    # Geom_ID Mask Array
    mask_array = np.zeros((len(grid_nums),64,64))

    for i in range(len(grid_nums)): 
        fname = os.path.join(mask_dir,mask_corresponded_fnames[i])
        # Save Mask as one big array
        with rasterio.open(fname) as src:
            mask_array[i,:,:] = src.read()[0,0:64,0:64]

    # Cluster geom_id by croptype
    fname = os.path.join(home, csv_source)
    crop_type = pd.read_csv(fname)

    # Crop name
    crop_names = pd.unique(crop_type['crop'])
    clustered_geom_id = [np.array(crop_type['geom_id'][crop_type['crop']==crop_name]) for crop_name in crop_names]

    # Crop Mask Array
    crop_dict = dict(zip(crop_names, np.arange(len(crop_names))+1))
    crop_mask_array = np.copy(mask_array)
    crop_mask_array = crop_mask_array.astype(object)

    for crop_name in crop_names:
        crop_idx = crop_dict[crop_name]
        for geom_id in clustered_geom_id[crop_idx-1]:
            crop_mask_array[mask_array==geom_id] = crop_name

    crop_mask_array[mask_array==0] = 'NonCrop'

    output_fname = "_".join([data_set, data_type, satellite, 'croptypemask', 'g'+str(len(grid_nums)),'r64', 'c64'+'.npy'])

    np.save(output_fname, crop_mask_array)
    
    
if __name__ == '__main__':
    
    # Construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-hm", "--home", required=False, default='/home/data/')
    arg_parser.add_argument("-c", "--country", required=False, default='Ghana')
    arg_parser.add_argument("-ds", "--dataset", required=False, default='full')
    arg_parser.add_argument("-dt", "--datatype", required=False, default='train')
    arg_parser.add_argument("-s", "--satellite", required=False, default='s1')
    arg_parser.add_argument("-csv", "--csvsource", required=False, default='ghana_crop.csv')
    args = vars(arg_parser.parse_args())
    
    home = args['home']
    country = args['country']
    data_set = args['dataset']
    data_type = args['datatype']
    satellite = args['satellite']
    csv_source = args['csvsource']
        
    get_y_labels(home, country, data_set, data_type, satellite, csv_source)