import pandas as pd
import pickle
import numpy as np
import json
import os
import rasterio
from random import shuffle
import random
from collections import defaultdict
from shutil import copyfile
import argparse

def get_crop_from_field_id(csv, field_id):
    return csv[csv['id'] == field_id]['crop']

def cp_files(split, prefix, source, dest, suffix):
    for phase in ['train', 'val', 'test']:
        for grid in split[phase]:
            for ext in ['.npy', '.json']:
                filename = prefix + grid + ext
                full_path_file = os.path.join(source, filename)
                if os.path.isfile(full_path_file):
                    full_path_dest = os.path.join(dest, phase, suffix, filename)
                    copyfile(full_path_file, full_path_dest)
                    


def get_field_to_grid_mapping(raster_dir, npy_dir, country):
    field_to_grids = defaultdict(set)
    # iterate through the npy files
    for grid_fn in os.listdir(npy_dir):
        grid_no, file_type = grid_fn.split('_')[-1].split('.')
        if file_type != 'npy': continue
        mask_name = country + "_64x64_" + grid_no + ".tif" 
        with rasterio.open(os.path.join(raster_dir, mask_name)) as mask_data:
            mask = mask_data.read()
        # need to separate the mask from the data
        fields = np.unique(mask)
        for field in fields:
            if field == 0: continue # not a field, just a place holder
            field_to_grids[field].add(grid_no)
    
     # converts the keys of fields_to_grids into a list
    
    return field_to_grids

def load_csv_for_split(csvname, crop_labels, valid_fields):
    csv = pd.read_csv(csvname)
    csv = csv[csv['crop'] != 'Intercrop'] # remove intercrop
    csv['crop'] = csv['crop'].apply(lambda x: x.lower())
    csv.at[~csv['crop'].isin(crop_labels[:-1]), 'crop'] = 'other' 
    csv = csv[csv['id'].isin(valid_fields)]
    # shuffle to ensure no bias for earlier fields
    csv = csv.sample(frac=1, random_state=0)
    return csv

def split_evenly(seed, csv, crop_labels, min_area=1e5):
    
    np.random.seed(seed) 
    
    area_per_class = {'train': defaultdict(float),
                      'val': defaultdict(float),
                      'test': defaultdict(float)}
    
    field_splits = {'train': set(),
                    'val': set(),
                    'test': set()}
    
    grid_splits = {'train': [],
                   'val': [],
                   'test': []}
    
    # iterate through all the (valid) fields in the csv
    for index, row in csv.iterrows():
        crop_type = row['crop']
        phase = None
        # assign to train with probability .8, val .1, test .1 IF not full yet
        if np.random.random() < .8:
            if area_per_class['train'][crop_type] < min_area: phase = 'train'
            elif np.random.random() < .5 and area_per_class['val'][crop_type] < min_area: phase = 'val'
            elif area_per_class['test'][crop_type] < min_area: phase = 'test'
        elif np.random.random() < .9:
            if area_per_class['val'][crop_type] < min_area: phase = 'val'
            elif area_per_class['test'][crop_type] < min_area: phase = 'test'
        else:
            if area_per_class['test'][crop_type] < min_area: phase = 'test'

        if phase != None:
            field_splits[phase].add(row['id'])
            area_per_class[phase][crop_type] += row['area']        

    # prints the number of m^2 for each split of the data
    for phase in ['train', 'val', 'test']:
        print(f"AREAS FOR PHASE {phase}: {area_per_class[phase]}\n")

    # checks to make sure there are no overlap between the different fields in the dataset
    for crop in crop_labels:
        intersection = field_splits['train'] & field_splits['val'] & field_splits['test']
        assert intersection == set(), print(f"BAD!: INTERSECTION FOR CROP {crop}")
       
    # converts splits by field into splits by grid number
    for phase in ['train', 'val', 'test']:
        for field in field_splits[phase]:
            grids = field_to_grids[field]
            grid_splits[phase] += list(grids)
            
    return grid_splits

def split_matching_dist(seed, csv, crop_labels):
    
    np.random.seed(seed) 
    
    max_area = defaultdict(float)
    
    for crop_type in crop_labels:
        max_area[crop_type] = sum(csv[csv['crop'] == crop_type]['area'])
    
    area_per_class = {'train': defaultdict(float),
                      'val': defaultdict(float),
                      'test': defaultdict(float)}
    
    total_area_per_phase = defaultdict(float)
    
    field_splits = {'train': set(),
                    'val': set(),
                    'test': set()}
    
    grid_splits = {'train': [],
                   'val': [],
                   'test': []}
    
    # iterate through all the (valid) fields in the csv
    for crop_type in crop_labels:
        crop_csv = csv[csv['crop'] == crop_type]
        for index, row in crop_csv.iterrows():
            crop_type = row['crop']
            phase = None
            # assign to train with probability .8, val .1, test .1 IF not full yet
            # some very complicated control flow to ensure probablitities work out
            if np.random.random() < .8:
                if area_per_class['train'][crop_type] < max_area[crop_type] * .8: phase = 'train'
                elif np.random.random() < .5: 
                    if area_per_class['val'][crop_type] < max_area[crop_type] * .1: phase = 'val'
                    elif area_per_class['test'][crop_type] < max_area[crop_type] * .1: phase = 'test'
                else:
                    if area_per_class['test'][crop_type] < max_area[crop_type] * .1: phase = 'test'
                    elif area_per_class['val'][crop_type] < max_area[crop_type] * .1: phase = 'val'
            elif np.random.random() < .9:
                if area_per_class['val'][crop_type] < max_area[crop_type] * .1: phase = 'val'
                elif np.random.random() < 8 / 9:
                    if area_per_class['train'][crop_type] < max_area[crop_type] * .8: phase = 'train'
                    elif area_per_class['test'][crop_type] < max_area[crop_type] * .1: phase = 'test'
                elif area_per_class['test'][crop_type] < max_area[crop_type] * .1: phase = 'test'
                elif area_per_class['train'][crop_type] < max_area[crop_type] * .8: phase = 'train'
            elif area_per_class['test'][crop_type] < max_area[crop_type] * .1: phase = 'test'
            elif np.random.random() < 8 / 9:
                if area_per_class['train'][crop_type] < max_area[crop_type] * .8: phase = 'train'
                elif area_per_class['test'][crop_type] < max_area[crop_type] * .1: phase = 'test'
            elif area_per_class['val'][crop_type] < max_area[crop_type] * .1: phase = 'val'
            elif area_per_class['train'][crop_type] < max_area[crop_type] * .8: phase = 'train'
                
            field_splits[phase].add(row['id'])
            area_per_class[phase][crop_type] += row['area']        
            total_area_per_phase[phase] += row['area']
            
    # prints the number of m^2 for each split of the data
    for phase in ['train', 'val', 'test']:
        print(f"PHASE {phase}")
        for crop in area_per_class[phase]:
            print(f"% AREA FOR CROP {crop}: {area_per_class[phase][crop] / total_area_per_phase[phase]}\n")

    # checks to make sure there are no overlap between the different fields in the dataset
    for crop in crop_labels:
        intersection = field_splits['train'] & field_splits['val'] & field_splits['test']
        assert intersection == set(), print(f"BAD!: INTERSECTION FOR CROP {crop}")
    
    # converts splits by field into splits by grid number
    for phase in ['train', 'val', 'test']:
        for field in field_splits[phase]:
            grids = field_to_grids[field]
            grid_splits[phase] += list(grids)
            
    return grid_splits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster_dir', type=str, required=True,
                        help='Path to directory containing rasters.',
                        default='/home/data/Ghana/raster_64x64/')
    parser.add_argument('--npy_dir', type=str, required=True,
                        help='Path to directory containing numpy volumes of grids.',
                        default='/home/data/Ghana/s1_64x64_npy/')
    parser.add_argument('--country', type=str, required=True,
                        help='Country to use',
                        default='ghana')
    
    args = parser.parse_args()

    raster_dir = args.raster_dir
    npy_dir = args.npy_dir
    country = args.country
    
    
    field_to_grids = get_field_to_grid_mapping(raster_dir, npy_dir, country)
    valid_fields = field_to_grids.keys()
    crop_labels  = ['maize','groundnut', 'rice', 'soya bean', 'other'] # should be stored in constants.py eventually
    csvname = f'{country}_crop.csv'
    csv = load_csv_for_split(csvname, crop_labels, valid_fields)
    even_grid_splits = split_evenly(1, csv, crop_labels)

    for split in even_grid_splits:
        print(len(even_grid_splits[split]))
    
    # uncomment these lines to actually copy the files over
    # cp_files(even_grid_splits, prefix="s1_ghana_", source="/home/data/Ghana/s1_64x64_npy", dest="/home/data/small", suffix="s1")
    # cp_files(even_grid_splits, prefix="s2_ghana_", source="/home/data/Ghana/s2_64x64_npy", dest="/home/data/small", suffix="s2")
    
    # error checking for overlap between phases CURRENTLY FAILS
    for split in ['train', 'val', 'test']:
        print(even_grid_splits[split])
        print(set(even_grid_splits['train']) & set(even_grid_splits['val']) & set(even_grid_splits['test']))
    assert set(even_grid_splits['train']) & set(even_grid_splits['val']) & set(even_grid_splits['test']) == set(), print("RIP")
    
    
    dist_grid_splits = split_matching_dist(10, csv, crop_labels)
    # uncomment these lines to actually copy the files over
    # cp_files(dist_grid_splits, prefix="s1_ghana_", source="/home/data/Ghana/s1_64x64_npy", dest="/home/data/full", suffix="s1")
    # cp_files(dist_grid_splits, prefix="s2_ghana_", source="/home/data/Ghana/s2_64x64_npy", dest="/home/data/full", suffix="s2")
    
