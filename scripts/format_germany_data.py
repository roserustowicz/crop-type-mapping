import os
import json
import rasterio
import numpy as np

import matplotlib.pyplot as plt
from skimage.transform import resize

def convert_label(label, crop_dict):
    label_new = np.zeros_like(label)
    for key in crop_dict:
        label_new[label == int(key)] = crop_dict[key]
    return label_new

# get filenames 
save_dir_s2 = '/home/roserustowicz/croptype_data_local/data/germany/s2_npy'
save_dir_lbl = '/home/roserustowicz/croptype_data_local/data/germany/raster_npy'
in_dir = '/home/roserustowicz/MTLCC-pytorch-fork/data/data'

crop_dict = {'1': 1, '2': 2, '3': 3, '5': 4, '8': 5, '9': 6, '12': 7, '13': 8, 
             '15': 9, '16': 10, '17': 11, '19': 12, '22': 13, '23': 14, '24': 15, '25': 16, '26': 17} 

fileids = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.isdigit()]

for id_dir in fileids:
    print(id_dir)
    fnames_10m = [os.path.join(id_dir, f) for f in os.listdir(id_dir) if f.endswith('10m.tif')]
    fnames_20m = [os.path.join(id_dir, f) for f in os.listdir(id_dir) if f.endswith('20m.tif')]
    fnames_60m = [os.path.join(id_dir, f) for f in os.listdir(id_dir) if f.endswith('60m.tif')]
    fname_label = os.path.join(id_dir, 'y.tif')
    
    fnames_10m.sort()
    fnames_20m.sort() 
    fnames_60m.sort()
    
    assert len(fnames_10m) > 1
    assert len(fnames_20m) > 1
    assert len(fnames_60m) > 1
    
    # Create timestack of 10m images
    imgs_10m = []
    for fname_10m in fnames_10m:
        with rasterio.open(fname_10m) as src:
            imgs_10m.append(np.expand_dims(src.read(), 3)) 
    imgs_10m = np.concatenate(imgs_10m, axis=3)
    
    # Create timestack of 20m images
    imgs_20m = []
    for fname_20m in fnames_20m:
        with rasterio.open(fname_20m) as src:
            imgs_20m.append(np.expand_dims(src.read(), 3))
    imgs_20m = np.concatenate(imgs_20m, axis=3)
    imgs_20m = resize(imgs_20m, (imgs_20m.shape[0], imgs_10m.shape[1], imgs_10m.shape[2], imgs_20m.shape[3]), mode='reflect')

    # Create timestack of 60m images
    imgs_60m = []
    for fname_60m in fnames_60m:
        with rasterio.open(fname_60m) as src:
            imgs_60m.append(np.expand_dims(src.read(), 3))
    imgs_60m = np.concatenate(imgs_60m, axis=3)
    imgs_60m = resize(imgs_60m, (imgs_20m.shape[0], imgs_10m.shape[1], imgs_10m.shape[2], imgs_20m.shape[3]), mode='reflect')
    
    # Reorder and concatenate all bands
    all_bands = [np.expand_dims(imgs_10m[0], 0), np.expand_dims(imgs_10m[1], 0), np.expand_dims(imgs_10m[2], 0),
                 np.expand_dims(imgs_20m[0], 0), np.expand_dims(imgs_20m[1], 0), np.expand_dims(imgs_20m[2], 0),
                 np.expand_dims(imgs_10m[3], 0), np.expand_dims(imgs_20m[3], 0), np.expand_dims(imgs_20m[4], 0),
                 np.expand_dims(imgs_20m[5], 0), np.expand_dims(imgs_60m[0], 0), np.expand_dims(imgs_60m[1], 0), 
                 np.expand_dims(imgs_60m[2], 0)]
    all_bands = np.concatenate(all_bands, axis=0)
   
    # Get and convert label using the crop_dict 
    with rasterio.open(fname_label) as src:
        label = np.squeeze(src.read())
    label = convert_label(label, crop_dict)

    # Get dates based on filenames
    dates = ["-".join([f.split("/")[-1].replace("_10m.tif", "")[:4], f.split("/")[-1].replace("_10m.tif", "")[4:6], f.split("/")[-1].replace("_10m.tif", "")[6:]]) for f in fnames_10m]
    meta = {}
    meta["dates"] = dates

    cur_fid = id_dir.split('/')[-1]
    # Save label
    output_fname_label = os.path.join(save_dir_lbl, 'germany_'+cur_fid+'_label.npy')
    np.save(output_fname_label, label)
    # Save raster stack
    output_fname_raster = os.path.join(save_dir_s2, 's2_germany_'+cur_fid+'.npy')
    np.save(output_fname_raster, all_bands)
    # Save json file with dates
    output_fname_json = os.path.join(save_dir_s2, 's2_germany_'+cur_fid+'.json')
    with open(output_fname_json, 'w') as fp:
        json.dump(meta, fp)  
        
    print('Saved for fid: ', cur_fid)
