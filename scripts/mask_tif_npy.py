import pandas as pd
import pickle
import numpy as np
import json
import operator
import os
import sys
import rasterio
import random
import argparse
import pickle
import itertools
import time
from datetime import datetime


def mask_tif_npy(home, country, csv_source, crop_dict_dir, raster_dir):
    """
    Transfer cropmask from .tif files by field_id to cropmask .npy by crop_id given crop_dict
    
    Args:
      home - (str) the base directory of data
      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'
      csv_source - (str) string for the csv field id file corresponding with the country
      crop_dict_dir - (str) string for the crop_dict dictionary {0: 'unlabeled', 1: 'groundnuts' ...}
      raster_dir - (str) string for the mask raster dir 'raster' or 'raster_64x64'
    Outputs:
      ./raster_npy/..
    """
    fname = os.path.join(home, country, csv_source)
    crop_csv = pd.read_csv(fname)

    mask_dir = os.path.join(home, country, raster_dir)
    mask_dir_npy = os.path.join(home, country, raster_dir+'_npy')
    mask_fnames = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    mask_ids = [f.split('_')[-1].replace('.tif', '') for f in mask_fnames]
    mask_fnames = [mask_fnames[ID] for ID in np.argsort(mask_ids)]
    mask_ids = np.array([mask_ids[ID] for ID in np.argsort(mask_ids)])

    crop_dict = np.load(os.path.join(home, country, crop_dict_dir))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', type=str,
                        help='home dir',
                        default='/home/data')
    parser.add_argument('--country', type=str,
                        help='which country',
                        default='ghana')                    
    parser.add_argument('--csv_source', type=str,
                        help='crop field id csv source',
                        default='ghana_crop.csv')        
    parser.add_argument('--crop_dict_dir', type=str,
                        help='crop dictionary dir',
                        default='ghana_crop_dict.npy')
    parser.add_argument('--raster_dir', type=str,
                        help='raster tiff file dir',
                        default='raster')

    args = parser.parse_args()
    mask_tif_npy(args.home, args.country, args.csv_source, args.crop_dict_dir, args.raster_dir)

