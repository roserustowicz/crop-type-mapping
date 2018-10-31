import os
import numpy as np
import pickle
import rasterio
from collections import Counter
import json

def get_empty_grids(home, countries, sources, verbose, ext, lbl_dir):
    """
    Provides data from input .tif files depending on function input parameters. 
    
    Args:
      home - (str) the base directory of data

      countries - (list of str) list of strings that point to the directory names
                  of the different countries (i.e. ['ghana', 'tanzania', 'southsudan'])

      sources - 

      verbose - (boolean) prints outputs from function

      ext - 
   
      lbl_dir - 
    """

    valid_pixels_list = []
    empty_masks = []
    for country in countries:
        mask_fnames = [os.path.join(home, country, lbl_dir, f) for f in os.listdir(os.path.join(home, country, lbl_dir)) if f.endswith('.tif')]
        mask_ids = [f.split('_')[-1].replace('.tif', '') for f in mask_fnames]

        mask_fnames.sort()
        mask_ids.sort()
    
        assert len(mask_fnames) == len(mask_ids)

        for mask_fname, mask_id in zip(mask_fnames, mask_ids):
            with rasterio.open(mask_fname) as src:
                cur_mask = src.read()
                valid_pixels = np.sum(cur_mask > 0) 
                valid_pixels_list.append((mask_id, valid_pixels))
                if valid_pixels == 0:
                    empty_masks.append(mask_id)

        delete_me = []
        for source in sources:
            grid_numbers, source_files = get_grid_nums(home, country, source, ext)

            all_ids = set(empty_masks + grid_numbers)
            for el in all_ids:
                if el in empty_masks and el in grid_numbers:
                    delete_me.append(el)

        if verbose:
            print('valid pixels list: ', len(valid_pixels_list))
            print('empty masks: ', len(empty_masks))
            print('delete me length: ', len(delete_me))
            print('delete me: ', delete_me)
        
        return set(delete_me)

def get_grid_nums(home, country, source, ext):
    cur_path = os.path.join(home, country, source)
    if ext == 'tif':
        files = [os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.endswith('.tif')]
        files.sort()
        if country == 'ghana':
            grid_numbers = [f.split('_')[-2] for f in files]
        elif country == 'tanzania':
            grid_numbers = [f.split('_')[-3] for f in files]
    elif ext == 'npy':
        files = [os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.endswith('.npy') or f.endswith('.json')]
        files.sort()
        grid_numbers = [f.split('_')[-1].replace('.npy', '') for f in files]

    grid_numbers.sort()
    return grid_numbers, files

def remove_irrelevant_files(home, countries, sources, delete_list, dry_run, ext): 
    for country in countries:
        for source in sources:
            grid_nums, source_files = get_grid_nums(home, country, source, ext)
            for grid_to_rm in delete_list:
                files_to_rm = [f for f in source_files if ''.join(['_', grid_to_rm]) in f]

                if dry_run: 
                    print("grid to remove: ", grid_to_rm)
                    print("files to remove: ", files_to_rm)
                else:
                    if verbose:
                        print("grid to remove: ", grid_to_rm)
                        print("files to remove: ", files_to_rm)
                    # Remove files 
                    [os.remove(f) for f in files_to_rm]

if __name__ == '__main__':

    home = '/home/data'
    countries = ['tanzania']
    sources = ['s1', 's2']
    #sources = ['s1_64x64_npy', 's2_64x64_npy']
    lbl_dir = 'raster'
    #lbl_dir = 'raster_64x64'
    verbose = 1
    dry_run = 1
    ext = 'tif'
    #ext = 'npy'

    grids_to_delete = get_empty_grids(home, countries, sources, verbose, ext, lbl_dir)
    remove_irrelevant_files(home, countries, sources, grids_to_delete, dry_run, ext)

