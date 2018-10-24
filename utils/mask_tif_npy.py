import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import rasterio
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
import json
import numpy.ma as ma
import argparse
import itertools
"""
    Transfer cropmask from .tif files by field_id to cropmask .npy by crop_id given crop_dict
    
    Args:
      home - (str) the base directory of data

      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'

      csv_source - (str) string for the csv field id file corresponding with the country

      crop_dict_dir - (str) string for the crop_dict dictionary {0: 'Unlabeled', 1: 'Groundnuts' ...}

    Outputs:
      ./raster_64X64_npy/..

"""

def mask_tif_npy(home, country, csv_source, crop_dict_dir):
    fname = os.path.join(home, csv_source)
    crop_csv = pd.read_csv(fname)

    mask_dir = os.path.join(home, country, 'raster_64x64')
    mask_dir_npy = os.path.join(home, country, 'raster_64x64_npy')
    mask_fnames = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    mask_ids = [f.split('_')[-1].replace('.tif', '') for f in mask_fnames]
    mask_fnames = [mask_fnames[ID] for ID in np.argsort(mask_ids)]
    mask_ids = np.array([mask_ids[ID] for ID in np.argsort(mask_ids)])

    crop_dict = np.load(os.path.join(home, crop_dict_dir))
    clustered_geom_id = [np.array(crop_csv['geom_id'][crop_csv['crop']==crop_name]) for crop_name in crop_dict.item().values()]

    for mask_fname in mask_fnames: 
        with rasterio.open(os.path.join(mask_dir,mask_fname)) as src:
            mask_array = src.read()[0,:,:]
            mask_array_geom_id = np.unique(mask_array)
            mask_array_crop_id = np.zeros(mask_array.shape)
            mask_array_crop_id[:] = np.nan
            for geom_id in mask_array_geom_id:
                if geom_id>0:
                    crop_num = np.where([geom_id in clustered_geom_id[i] for i in np.arange(len(clustered_geom_id))])[0][0]
                    mask_array_crop_id[mask_array==geom_id] = crop_num
                elif geom_id == 0:
                    mask_array_crop_id[mask_array==geom_id] = 0
            np.save(os.path.join(mask_dir_npy,mask_fname.replace('.tif', '.npy')), mask_array_crop_id)


if __name__ == '__main__':

    # Construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-hm", "--home", required=False, default='/home/data/')
    arg_parser.add_argument("-c", "--country", required=False, default='Ghana')
    arg_parser.add_argument("-csv", "--csvsource", required=False, default='ghana_crop.csv')
    arg_parser.add_argument("-cdir", "--cropdir", required=False, default='crop_dict.npy')
    args = vars(arg_parser.parse_args())
    
    home = args['home']
    country = args['country']
    csv_source = args['csvsource']
    crop_dict_dir = args['cropdir']