import os
import numpy as np
import pickle
import rasterio
import pandas as pd

def croptype_mask(home, country, csv_source, grid_nums, satellite):
    """
    Creates a croptype array of dimensions grids x rows x columns for each country, 
    source combination i.e. Ghana + s1, Ghana + s2, etc. 
    """
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
   
    output_fname = "_".join([country, satellite, 'croptypemask', 'g'+str(len(grid_nums)),'r64', 'c64'+'.pickle'])

    with open(output_fname, "wb") as f:
        pickle.dump(crop_mask_array, f)


    return crop_mask_array

    
if __name__ == '__main__':
    
    home = '/home/lijing/data_from_rose/data'
    country = 'Ghana'
    csv_source = 'ghana_crop.csv'
    satellite = 's2'
    
    fname = os.path.join(home, country, 'Ghana_s2_64x64_shape_g227_b10_r64_c64_t51.pickle')
    
    with open(fname, 'rb') as f:
        grid_nums, _ = pickle.load(f, encoding='bytes')
    
    # Change bytes type to str
    grid_nums = [str(0)+grid_num.decode("utf-8") for grid_num in grid_nums]

    croptype_mask(home, country, csv_source, grid_nums, satellite)
    
    
