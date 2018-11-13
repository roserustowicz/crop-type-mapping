import os
import numpy as np
import pickle
import rasterio
from collections import Counter
import json

def get_img_cube(home, countries, sources, verbose, out_format, lbl_dir, filter_s1):
    """
    Provides data from input .tif files depending on function input parameters. 
    
    Note: the current implementation is highly dependent on filenames. Mask filenames are assumed to be 
          stored in 'HOME/raster_64x64/', where HOME is the 'home' input argument. Files associated with 
          Sentinel-1 are assumed to be in a source directory that contains 's1'. 
          Sentinel-1 files assume the format 's1_COUNTRY_ORBIT_GRIDID_YYYY_MM_DD.tif' 
          Sentinel-2 files assume the format 's2_COUNTRY_GRIDID_YYYY_MM_DD.tif'
          Mask files assume the format 'COUNTRY_ROWSxCOLS_GRIDID.tif' 

    Args:
      home - (str) the base directory of data

      countries - (list of str) list of strings that point to the directory names
                  of the different countries (i.e. ['Ghana', 'Tanzania', 'SouthSudan'])

      sources - (list of str) list of strings that point to the directory names 
                of the different satellite sources (i.e. ['s1_64x64', 's2_64x64'])

      verbose - (boolean) prints outputs from function

      out_format - (str) takes "pickle", "5d_array", or "npy"
          for "pickle": Creates a data array of dimensions grids x bands x rows x columns x timestamps
                        for each country, source combination i.e. Ghana + s1, Ghana + s1, etc. and saves
                        the array as a '.pickle' file

          for "5d_array": Currently returns the 5d array associated with the last country, source 
                          combination as a numpy array

          for "npy": Creates a data array of dimensions bands x rows x columns x timestamps for every
                     grid for each country, source combination and saves as '.npy' files. Also creates
                     a fname.json file corresponding to the fname.npy file, with 'dates' (for s1 and s2)
                     and 'orbits' (only for s1) information. 'dates' gives the acquisition dates for all 
                     images stacked in the array, in sequential order. 'orbits' gives the type of orbit 
                     the image was taken in, either 'asc' for ascending or 'desc' for descending. 
      
      lbl_dir - (str) the directory name that the raster labels are stored in 
                      (i.e. 'raster', 'raster_64x64')

      filter_s1 - (boolean) If working with Sentinel-1 data, this will take only the last three bands,
                   which include `angle`, `vv_gamma`, and `vh_gamma` 

    """

    for country in countries:
        mask_fnames = [os.path.join(home, country, lbl_dir, f) for f in os.listdir(os.path.join(home, country, lbl_dir)) if f.endswith('.tif')]
        mask_ids = [f.split('_')[-1].replace('.tif', '') for f in mask_fnames]
    
        if verbose:
            print('Number of grids for country {}: {}'.format(country, len(mask_ids)))

        for source in sources:
            cur_path = os.path.join(home, country, source)
            files = [os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.endswith('.tif')]
            if country == 'ghana':
                grid_numbers = [f.split('_')[-2] for f in files]
            elif country in ['tanzania', 'southsudan']:
                grid_numbers = [f.split('_')[-3] for f in files]
            grid_numbers.sort()
 
            # read one image from list to get dimensions
            with rasterio.open(files[0]) as src:
                img = src.read()
        
            if out_format == 'pickle':
                data_array = np.zeros((len(set(grid_numbers)), img.shape[0], img.shape[1], img.shape[2], Counter(grid_numbers).most_common(1)[0][1]))
                g, b, r, c, t = data_array.shape       

            if verbose: 
                print('-----------------------------')
                print('Image dimensions: {}'.format(img.shape))
                print('Current data source: {}'.format(source))
                print('Number of grids in set: {}'.format(len(set(grid_numbers))))
                print('Set of grid numbers: {}'.format(sorted(set(grid_numbers))))
                print('Maximum timestamps from this data source: {}'.format(Counter(grid_numbers).most_common(1)[0][1]))
                if out_format == 'pickle':
                    print('Final array shape: {}'.format(data_array.shape))

            for grid_idx, grid in enumerate(sorted(set(grid_numbers))):
                if verbose:
                    print('Grid: {}'.format(grid))
                cur_grid_files = [f for f in files if '_' + grid + '_' in f]
                cur_grid_files.sort() # sorts in time
                
                if out_format == 'npy':
                    if 's1' in source and filter_s1:
                        data_array = np.zeros((3, img.shape[1], img.shape[2], len(cur_grid_files)))
                    else:
                        data_array = np.zeros((img.shape[0], img.shape[1], img.shape[2], len(cur_grid_files)))
                    
                    dates = []
                    if 's1' in source:
                        orbit = []

                for idx, fname in enumerate(cur_grid_files):
                    with rasterio.open(fname) as src:
                        if out_format == 'pickle':
                            data_array[grid_idx, :, :, :, idx] = src.read()
                        elif out_format == 'npy':
                            if 's1' in source and filter_s1:
                                s1_subset = src.read()[2:, :, :]
                                data_array[:, :, :, idx] = s1_subset
                            else:
                                data_array[:, :, :, idx] = src.read()
                            if country == 'ghana':
                                dates.append(fname.split('/')[-1][-14:-4])
                            elif country in ['tanzania', 'southsudan']:
                                tmp = fname.split('/')[-1].split('_')[:-1]+['.tif']
                                dates.append(tmp[-2])
                            if 's1' in source: 
                                orbit.append(fname.split('/')[-1].split('_')[2])

                if out_format == 'npy':
                    tmp = fname.split('/')
                    if country in ['tanzania', 'southsudan']: 
                        tmp[-1] = tmp[-1].split('_')[:-1]
                        tmp[-1] = "_".join(tmp[-1])+'.tif'
                    tmp[-1] = tmp[-1][:-15].replace('asc_', '').replace('desc_', '')
                    tmp[-2] = tmp[-2] + '_npy'
                    output_fname = "/".join(tmp)

                    if not os.path.exists('/'.join(output_fname.split('/')[:-1])):
                        os.makedirs('/'.join(output_fname.split('/')[:-1]))
                                   
                    # store and save metadata
                    meta = {}
                    meta['dates'] = dates
                    if 's1' in source:
                        meta['orbits'] = orbit
                    with open(output_fname + '.json', 'w') as fp:
                        json.dump(meta, fp)

                    # save image stack as .npy
                    np.save(output_fname, data_array)         

            if out_format == 'pickle':
                output_fname = "_".join([country, source, 'shape', 'g'+str(g), 'b'+str(b), 'r'+str(r), 'c'+str(c), 't'+str(t)+'.pickle'])
                with open(output_fname, "wb") as f:
                    pickle.dump((sorted(set(grid_numbers)), data_array), f)

            if out_format == '5d_array':            
                return data_array # only returns last one as of now       


if __name__ == '__main__':

    home = '/home/data'
    countries = ['southsudan']
    sources = ['s1']
    lbl_dir = 'raster'
    verbose = 1
    out_format = 'npy'
    filter_s1 = 1

    data_array = get_img_cube(home, countries, sources, verbose, out_format, lbl_dir, filter_s1)
    
    # To load pickle file ... 
    #with open(fname, "rb") as f:
    #    grid_nums,data_array = pickle.load(f) 

