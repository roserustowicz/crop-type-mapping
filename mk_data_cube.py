import os
import numpy as np
import pickle
import rasterio
from collections import Counter


def get_img_cube(home, countries, sources, verbose):
    """
    Creates a data array of dimensions grids x bands x rows x columns x timestamps
    for each country, source combination i.e. Ghana + s1, Ghana + s2, etc.
    """

    for country in countries:
        mask_fnames = [os.path.join(home, country, 'raster', f) for f in os.listdir(os.path.join(home, country, 'raster')) if f.endswith('.tif')]
        mask_ids = [f.split('_')[-1].replace('.tif', '') for f in mask_fnames]
    
        if verbose:
            print 'Number of grids for country {}: {}'.format(country, len(mask_ids))

        for source in sources:
            cur_path = os.path.join(home, country, source)
            files = [os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.endswith('.tif')]
            #grid_numbers = [f.split('_')[-2].zfill(5) for f in files]
            grid_numbers = [f.split('_')[-2] for f in files]
            grid_numbers.sort()
 
            # read one image from list to get dimensions
            with rasterio.open(files[0]) as src:
                img = src.read()
        
            data_array = np.zeros((len(set(grid_numbers)), img.shape[0], img.shape[1], img.shape[2], Counter(grid_numbers).most_common(1)[0][1]))
            g, b, r, c, t = data_array.shape       

            if verbose: 
                print '-----------------------------'
                print 'Image dimensions: {}'.format(img.shape)
                print 'Current data source: {}'.format(source)
                print 'Number of grids in set: {}'.format(len(set(grid_numbers)))
                print 'Set of grid numbers: {}'.format(sorted(set(grid_numbers)))
                print 'Maximum timestamps from this data source: {}'.format(Counter(grid_numbers).most_common(1)[0][1])
                print 'Final array shape: {}'.format(data_array.shape)

            for grid_idx, grid in enumerate(sorted(set(grid_numbers))):
                if verbose:
                    print 'Grid: {}'.format(grid)
                cur_grid_files = [f for f in files if '_' + grid + '_' in f]
                cur_grid_files.sort() # sorts in time
                for idx, fname in enumerate(cur_grid_files):
                    if verbose:
                        print fname
                    with rasterio.open(fname) as src:
                        data_array[grid_idx, :, :, :, idx] = src.read()

            output_fname = "_".join([country, source, 'shape', 'g'+str(g), 'b'+str(b), 'r'+str(r), 'c'+str(c), 't'+str(t)+'.pickle'])
            with open(output_fname, "wb") as f:
                pickle.dump((sorted(set(grid_numbers)), data_array), f)
            
    return data_array # only returns last one as of now       
 

if __name__ == '__main__':

    home = '/home/roserustowicz/data'
    countries = ['Ghana']
    sources = ['s2_64x64']
    verbose = 1

    data_array = get_img_cube(home, countries, sources, verbose)
    
    # To load pickle file ... 
    #with open(fname, "rb") as f:
    #    grid_nums,data_array = pickle.load(f) 

