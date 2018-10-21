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
    for split in ['train', 'val', 'test']:
        for grid in split[split]:
            for ext in ['.npy', '.json']:
                filename = prefix + grid + ext
                full_path_file = os.path.join(source, filename)
                if os.path.isfile(full_path_file):
                    full_path_dest = os.path.join(dest, split, suffix, filename)
                    copyfile(full_path_file, full_path_dest)

                    
def get_field_grid_mappings(raster_dir, npy_dir, country):
    field_to_grids = defaultdict(set)
    grid_to_fields = defaultdict(set)
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
            grid_to_fields[grid_no].add(field)
    
     # converts the keys of fields_to_grids into a list
    
    return field_to_grids, grid_to_fields

def create_clusters(csv, field_to_grids, grid_to_fields):
    # for each cluster, should ensure that
    # if a field appears in the cluster, then all of the grids that contain the field appear in the cluster
    # if a field appears in one cluster, then it only appears within that cluster
    clusters = []
    missing = []
    seen = set()
    for field in field_to_grids:
        if field not in seen:
            fields_to_add = Queue()
            fields_to_add.put(field)
            new_cluster = {'grids': set(),
                           'fields': set(),
                           'crop_counts': defaultdict(float)}
            
            while(not fields_to_add.empty()):
                cur_field = fields_to_add.get()
                new_cluster['fields'].add(cur_field)
                seen.add(cur_field)
                field_info = csv[csv['id'] == cur_field]
                if field_info.empty:
                    missing.append(cur_field)
                    continue
                # add the amount of crop this field contributes to the cluster total
                new_cluster['crop_counts'][field_info.crop.item()] += field_info.area.item()
                for grid in field_to_grids[cur_field]:
                    # add all grids of the current field
                    new_cluster['grids'].add(grid)
                    
                    # check whether the current grid contains any new fields
                    for potential_field in grid_to_fields[grid]:
                        # new field
                        if potential_field not in seen:
                            seen.add(potential_field)
                            fields_to_add.put(potential_field)
            
            clusters.append(new_cluster)
    
    return clusters, missing
        

def load_csv_for_split(csvname, crop_labels, valid_fields):
    # read in csv and rename crops not in `crop_labels`
    csv = pd.read_csv(csvname)
    csv = csv[csv['crop'] != 'Intercrop'] # remove intercrop
    csv['crop'] = csv['crop'].apply(lambda x: x.lower())
    csv.at[~csv['crop'].isin(crop_labels[:-1]), 'crop'] = 'other' 
    csv = csv[csv['id'].isin(valid_fields)]
    # shuffle to ensure no bias for earlier fields
    csv = csv.sample(frac=1, random_state=0)
    return csv

def split_evenly(seed, clusters, target_area=1e5):
        
    area_per_split = {'train': defaultdict(float),
                      'val': defaultdict(float),
                      'test': defaultdict(float)}
    
    cluster_splits = {'train': [],
                      'val': [],
                      'test': []}
    
    random.seed(seed)
    
    for cluster in clusters:
        available = ['train', 'val', 'test']
        for crop in cluster['crop_counts']:
            crop_count = cluster['crop_counts'][crop]
            if area_per_split['train'][crop] + crop_count > target_area * 1.05:
                if 'train' in available: available.remove('train')
            if area_per_split['val'][crop] + crop_count > target_area * 1.05:
                if 'val' in available: available.remove('val')
            if area_per_split['test'][crop] + crop_count > target_area * 1.05:
                if 'test' in available: available.remove('test')
        
        if available != []:
            split = random.choice(available)

            for crop in cluster['crop_counts']:
                crop_count = cluster['crop_counts'][crop]
                area_per_split[split][crop] += crop_count

            cluster_splits[split].append(cluster)
        
    print(area_per_split)
        
    return cluster_splits

def create_dist_split_targets(csv):
    targets = {'train': defaultdict(float),
               'val': defaultdict(float),
               'test': defaultdict(float)}
    
    total_area = sum(csv['area'])
    percent_per_crop = defaultdict(float)
    for crop in crop_labels:
        percent_per_crop[crop] = sum(csv[csv['crop'] == crop].area)/total_area
        
    for crop in crop_labels:
        targets['train'][crop] = percent_per_crop[crop] * .8 * total_area
        targets['val'][crop] = percent_per_crop[crop] * .1 * total_area
        targets['test'][crop] = percent_per_crop[crop] * .1 * total_area
    
    return targets

def assign_to_split(available):
    split = 'train'
    p = random.random()
    if 'train' in available:
        if 'val' in available:
            if 'test' in available:
                if p < .8: split = 'train'
                elif p < .9: split = 'val'
                else: split = 'test'
            elif p < 8/9.0: split = 'train'
            else: split = 'val'
        elif 'test' in available:
            if p < 8/9: split = 'train'
            else: split = 'test'
    elif 'val' in available:
        if 'test' in available:
            if p < .5: split = 'val'
            else: split = 'test'
        else: split = 'val'
    elif 'test' in available: split = 'test'
    
    return split

def dist_split(seed, clusters, targets):
    
    random.seed(seed)
    
    area_per_split = {'train': defaultdict(float),
                      'val': defaultdict(float),
                      'test': defaultdict(float)}

    cluster_splits = {'train': [],
                      'val': [],
                      'test': []}
    
    for cluster in clusters:    
        available = ['train', 'val', 'test']
        
        for crop in cluster['crop_counts']:
            crop_count = cluster['crop_counts'][crop]
            if area_per_split['train'][crop] + crop_count > targets['train'][crop]:
                if 'train' in available: available.remove('train')
            if area_per_split['val'][crop] + crop_count > targets['val'][crop]:
                if 'val' in available: available.remove('val')
            if area_per_split['test'][crop] + crop_count > targets['train'][crop]:
                if 'test' in available: available.remove('test')
        
        split = assign_to_split(available)
        
#         print(split)
        
        for crop in cluster['crop_counts']:
            crop_count = cluster['crop_counts'][crop]
            area_per_split[split][crop] += crop_count

        cluster_splits[split].append(cluster)
        
        
    for split in ['train', 'val', 'test']:
        print(f"Split: {split}")
        total_area = sum([area_per_split[split][crop] for crop in area_per_split[split]])
        print(f"TOTAL AREA: {total_area}")
        for crop in area_per_split[split]:
            print(f"% {crop} : {area_per_split[split][crop] / total_area}")
#             print(f"raw {crop} : {area_per_split[split][crop] / 1e5}")
#             print(f"target {crop}: {targets[split][crop] / 1e5}\n")
            
            
    return cluster_splits

def create_grid_splits(cluster_splits):
    grid_splits = {'train': set(),
                   'val': set(),
                   'test': set()}

    for split in cluster_splits:
        for cluster in cluster_splits[split]:
            grid_splits[split] = grid_splits[split] | cluster['grids']
    
    assert grid_splits['train'] & grid_splits['val'] & grid_splits['test'] == set(), "GRIDS OVERLAP BETWEEN DIFF SPLITS"
    
    return grid_splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster_dir', type=str,
                        help='Path to directory containing rasters.',
                        default='/home/data/Ghana/raster_64x64/')
    parser.add_argument('--npy_dir', type=str,
                        help='Path to directory containing numpy volumes of grids.',
                        default='/home/data/Ghana/s1_64x64_npy/')
    parser.add_argument('--country', type=str,
                        help='Country to use',
                        default='ghana')

    args = parser.parse_args()

    raster_dir = args.raster_dir
    npy_dir = args.npy_dir
    country = args.country


    field_to_grids, grid_to_fields = get_field_to_grid_mapping(raster_dir, npy_dir, country)
    valid_fields = field_to_grids.keys()
    crop_labels  = ['maize','groundnut', 'rice', 'soya bean', 'yam', 'other'] # should be stored in constants.py eventually
    csvname = f'/home/data/{country}_crop.csv'
    csv = load_csv_for_split(csvname, crop_labels, valid_fields)
    clusters, missing = create_clusters(csv, field_to_grids, grid_to_fields)
    even_cluster_splits = split_evenly(1, clusters)
    even_grid_splits = create_grid_splits(even_cluster_splits)
    
    # uncomment these lines to actually copy the files over
    # cp_files(even_grid_splits, prefix="s1_ghana_", source="/home/data/Ghana/s1_64x64_npy", dest="/home/data/small", suffix="s1")
    # cp_files(even_grid_splits, prefix="s2_ghana_", source="/home/data/Ghana/s2_64x64_npy", dest="/home/data/small", suffix="s2")

    dist_targets = create_dist_split_targets(csv)
    dist_cluster_splits = dist_split(4, clusters, dist_targets)
    
    dist_grid_splits = create_grid_splits(dist_cluster_splits)
    # cp_files(dist_grid_splits, prefix="s1_ghana_", source="/home/data/Ghana/s1_64x64_npy", dest="/home/data/full", suffix="s1")
    # cp_files(dist_grid_splits, prefix="s2_ghana_", source="/home/data/Ghana/s2_64x64_npy", dest="/home/data/full", suffix="s2")
