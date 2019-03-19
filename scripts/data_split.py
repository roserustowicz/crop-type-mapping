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
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from constants import *
from random import shuffle
from queue import Queue
from collections import defaultdict
from shutil import copyfile


def correctSouthSudanLabels(crop):
    if '/' in crop:
        return 'Intercrop'
    elif 'Sorghum' in crop:
        return 'Sorghum'
    else:
        return crop

def standardize_csv(csv, country):
    # for south sudan which uses LC instead of crop
    if country == 'southsudan':
        csv['crop'] = csv['LC']
        csv['geom_id'] = csv['ID']
        csv['crop'] = csv['crop'].apply(correctSouthSudanLabels)
    return csv
    

def get_crop_from_field_id(csv, field_id):
    field = csv[csv['geom_id'] == field_id]
    if field.empty:
        return None
    return field.iloc[0]['crop']
    
def get_crop_labels(country):
    if country in ['ghana', 'southsudan', 'tanzania']:
        return CROPS[country]
    raise ValueError(f"Country: {country} not supported!")
    
def load_csv_for_split(csvname, crop_labels, valid_fields, country):
    """ Preprocesses the csv for datasplitting purposes.

    Removes intercrop, relabels all crops not in crop_labels as other, and shuffles the csv.

    Args:
        csvname - (string) name of the csv file
        crop_labels - (list of strings) contains all crops under consideration
        valid_fields - (list of ints) contains all available field_nums (npy files exist)

    Returns:
        csv - (pandas df) preprocessed csv with intercrop removed, crops relabeled, and invalid / missing fields removed

    """
    # read in csv and rename crops not in `crop_labels`
    csv = pd.read_csv(csvname)
    csv = standardize_csv(csv, country)
    csv['crop'] = csv['crop'].apply(lambda x: x.lower())
    csv = csv[csv['crop'] != 'intercrop'] # remove intercrop
    csv['crop'] = csv['crop'].apply(lambda x: x if x in crop_labels else 'other')
    csv = csv[csv['geom_id'].isin(valid_fields)]
    # shuffle to ensure no bias for earlier fields
    csv = csv.sample(frac=1, random_state=0)
    return csv

def get_field_grid_mappings(raster_dir, npy_dir, country):
    """ Returns mappings from fields to grids and grids to fields.
    Args:
        raster_dir - (string) directory containing the rasters
        npy_dir - (string) directory containing the numpy representation of the grids
        country - (string) the country being analyzed

    Returns:
        field_to_grids - (map: int->list of ints) mapping from a field number to a list of grids that the field appears in
        grid_to_fields - (map: int->list of ints) mapping from a grid to a list of field nums that appear in the grid

    """
    field_to_grids = defaultdict(set)
    grid_to_fields = defaultdict(set)
    # iterate through the npy files
    for mask_name in os.listdir(raster_dir):
        grid_no, ext = mask_name.split('_')[-1].split('.')
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

def create_clusters_simple(mask_dir, unlabeled=0):
    """ Creates clusters containing one grid for simplification.

    Unlabeled (int) is the integer used to signal an unlabeled pixel.
    """
    clusters = []
    unique_crops = set()
    for grid_fn in os.listdir(mask_dir):
        grid_no = grid_fn.split('_')[2]
        cluster = {'grids': set(),
                   'crop_counts': defaultdict(float)} # technically a cluster of just one grid, but defined in this way to use existing code
        cluster['grids'].add(grid_no)
        grid = np.load(os.path.join(mask_dir, grid_fn))
        crops, counts = np.unique(grid, return_counts=True)
        for i, crop in enumerate(crops):
            if crop == unlabeled: continue
            cluster['crop_counts'][crop] += counts[i]
            unique_crops.add(crop)
        clusters.append(cluster)
    
    all_grids = set()
    for cluster in clusters:
        assert all_grids & cluster['grids'] == set(), f"GRID OVERLAP, {cluster['grids']}"
        all_grids = all_grids | cluster['grids']
    
    return clusters

def create_clusters(csv, field_to_grids, grid_to_fields, raster_dir, crop_mapping, verbose=False):
    """ Returns a division of fields and grids such that there is no overlap.

    For each cluster,
    if a field appears in the cluster, then all of the grids that contain the field appear in the cluster
    if a field appears in one cluster, then it only appears within that cluster

    Args:
        csv - (pandas df) a pre-processed csv containing all the valid fields
        field_to_grids - (map: int->list of ints) mapping from a field to a list of all grids the field appears in
        grid_to_fields - (map: int->list of ints) mapping from a grid num to a list of all fields within that grid
        raster_dir - (string) name of directory containing rasters masks
        crop_mapping - (map: crop -> numerical label)
        verbose - (bool) true / false for printing more information (mainly error checking)

    Returns:
        clusters - (list of dictionaries) a list of all unique clusters where each cluster contains information about the grids, fields, and crop counts within that cluster
        missing - (list of ints) a list of all fields that could not be found in the csv, currently checked to be only intercrop fields

    """
    clusters = []
    missing = []
    seen = set()
    for grid in grid_to_fields:
        if grid not in seen:
            grids_to_add = Queue()
            grids_to_add.put(grid)
            new_cluster = {'grids': set(),
                           'fields': set(),
                           'crop_counts': defaultdict(float)}

            while(not grids_to_add.empty()):
                cur_grid = grids_to_add.get()
                if cur_grid in seen: continue
                new_cluster['grids'].add(cur_grid)
                seen.add(cur_grid)
                mask_name = country + "_64x64_" + cur_grid + ".tif"
                with rasterio.open(os.path.join(raster_dir, mask_name)) as mask_data:
                    mask = mask_data.read()
                    
                # add the amount of crop this field contributes to the cluster total
                fields, counts = np.unique(mask, return_counts=True)
                for i, field in enumerate(fields):
                    if field == 0: continue
                    crop = get_crop_from_field_id(csv, field)
                    if crop is None:
                        missing.append(field)
                        continue
                    crop = 6 if crop == 'other' else crop_mapping[crop]
                    if crop >= 6: continue # ignore everything that's not a top 5 crop
                    new_cluster['crop_counts'][crop] +=  counts[i]
                    new_cluster['fields'].add(field)
                    for potential_grid in field_to_grids[field]:
                        grids_to_add.put(potential_grid)
                        
            if new_cluster['fields'] != set():
                clusters.append(new_cluster)

    if verbose:
        all_grids = set()
        all_fields = set()
        for cluster in clusters:
            assert all_grids & cluster['grids'] == set(), "GRID OVERLAP"
            assert all_fields & cluster['fields'] == set(), f"FIELD OVERLAP: {all_fields & cluster['fields']}, {all_fields}"
            all_grids = all_grids | cluster['grids']
            all_fields = all_fields | cluster['fields']
        print("CLUSTERS MAINTAIN INDEPENDENCE")
        print(f"NUM MISSING FIELDS: {len(missing)}")

    return clusters, missing



def split_evenly(seed, clusters, target_area=1e3, verbose=False):
    """ Returns a split of the data such that each class has equalish area.

    Args:
        seed - (int) the random seed to use
        clusters - (list of dicts) list of all valid clusters where each cluster contains information about fields, grids, and crop counts inside the cluster
        target_area - (int) approximate amount of area that should be present in each class (default 1e5)
        verbose - (bool) error printing (default false)

    Returns:
        cluster_splits - (dict of lists of clusters) mapping between a split and all the clusters that belong to the split

    """

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
            if crop not in range(1, 6): continue
            crop_count = cluster['crop_counts'][crop]
            # times 1.05 to give more leeway, not sure if that's the right idea
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

    if verbose:
        print("EVEN SPLITS: ")
        for split in ['train', 'val', 'test']:
            print(area_per_split[split])

    return cluster_splits

def create_dist_split_targets(clusters):
    """ Calculates and creates area targets for a distributed split.

    Assists in ensuring the naturally distributed split is naturally distributed.

    Args:
        clusters - 
    Returns:
        targets - (dict of dict of floats) maps each split to the amount of area that should exist for each crop in that split (based on an .8 / .1 / .1 train val test split)

    """
    targets = {'train': defaultdict(float),
               'val': defaultdict(float),
               'test': defaultdict(float)}

    total_area = defaultdict(float)
    for cluster in clusters:
        for crop in cluster['crop_counts']:
            total_area[crop] += cluster['crop_counts'][crop]

    for crop in total_area:
        targets['train'][crop] = total_area[crop] * .8
        targets['val'][crop] = total_area[crop] * .1
        targets['test'][crop] = total_area[crop] * .1

    return targets

def assign_to_split(available):
    """ Helper function to properly assign a cluster to a split.

    Args:
        available - (list of strings) the available splits

    Returns:
        split - (string) one of 'train', 'val', 'test' determined by a .8 / .1 / .1 split

    """
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

def dist_split(seed, clusters, targets, verbose=False):
    """ Splits the data according to its natural distribution.

    Args:
        seed - (int) random seed
        clusters - (list of dicts) list of all valid clusters where each cluster contains information about fields, grids, and crop counts inside the cluster
        targets - (dict of dict of ints) dict mapping approximate amount of area that should be present in each class for each split
        verbose - (bool) error printing (default: false)

    Returns:
        cluster_splits - (dict of lists of clusters) mapping between a split and all the clusters that belong to the split according to the data's natural distribution of classes

    """
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
            if area_per_split['test'][crop] + crop_count > targets['test'][crop]:
                if 'test' in available: available.remove('test')

        split = assign_to_split(available)

        for crop in cluster['crop_counts']:
            crop_count = cluster['crop_counts'][crop]
            area_per_split[split][crop] += crop_count


        cluster_splits[split].append(cluster)

    if verbose:
        for split in ['train', 'val', 'test']:
            print(f"Split: {split}")
            total_area = sum([area_per_split[split][crop] for crop in area_per_split[split]])
            print(f"TOTAL AREA: {total_area}")
            for crop in sorted(area_per_split[split]):
                print(f"% {crop} : {area_per_split[split][crop] / total_area * 100}")
                print(f"raw {crop} : {area_per_split[split][crop] / 1e3}")
                print(f"target {crop}: {targets[split][crop]}\n")

    return cluster_splits

def create_grid_splits(cluster_splits):
    """ Returns a dict mapping split to a set of grids in each split.

    Args:
        cluster_splits - (dict of lists of clusters) mapping between a split and all the clusters that belong to the split

    Returns:
        grid_splits - (dict of set of ints) mapping between a split and all grids that belong to the split

    """
    grid_splits = {'train': set(),
                   'val': set(),
                   'test': set()}

    for split in cluster_splits:
        for cluster in cluster_splits[split]:
            grid_splits[split] = grid_splits[split] | cluster['grids']

    assert grid_splits['train'] & grid_splits['val'] & grid_splits['test'] == set(), "GRIDS OVERLAP BETWEEN DIFF SPLITS"

    return grid_splits

def save_grid_splits(grid_splits, out_dir, prefix):
    """ Saves the grid_splits as a pickle object.

    Args:
        grid_splits - (dict of set of ints) mapping between a split and all grids in that split
        out_dir - (string) output directory
        prefix - (string) what to prepend the filename with (e.g "ghana_full_")
    """

    for split in grid_splits:
        with open(os.path.join(out_dir, prefix + split), "wb") as outfile:
            pickle.dump(grid_splits[split], outfile)

def check_pixel_counts(mask_dir, country, csv, grid_splits):
    """ For each class, prints the number of pixels in each split.

    Args:
        mask_dir - (string) directory containing the masks for each grid
        country - (string) country being analyzed
        csv - (csv) preprocessed csv containing information
        grid_splits - (dict of set of ints) mapping between a split and all grids that belong to the split

    """
    for split in ['train', 'val', 'test']:
        pixel_counts = defaultdict(float)
        for grid in grid_splits[split]:
            # look up mask for this grid number
            mask_name = country + "_64x64_" + grid + "_label.npy"
            mask = np.load(os.path.join(mask_dir, mask_name))
            # need to separate the mask from the data
            crops, counts = np.unique(mask, return_counts=True)
            for i, crop in enumerate(crops):
                if crop == 0: continue
                crop = min(crop, 6)
                pixel_counts[crop] += counts[i]
        print(f"FOR SPLIT {split}: ")
        for crop, pixels in sorted(pixel_counts.items(), key= lambda kv: kv[1]):
            print(f"\tCROP: {crop} has {pixels} pixels ")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster_dir', type=str,
                        help='Path to directory containing rasters.',
                        default=GHANA_RASTER_DIR)
    parser.add_argument('--mask_dir', type=str,
                        help='Path to directory containing npy mask.',
                        default=GHANA_RASTER_NPY_DIR)
    parser.add_argument('--npy_dir', type=str,
                        help='Path to directory containing numpy volumes of grids.',
                        default=GHANA_S1_DIR)
    parser.add_argument('--country', type=str,
                        help='Country to use',
                        default='ghana')
    parser.add_argument('--verbose', type=bool,
                        help='error-checking on?',
                        default=False)
    parser.add_argument('--save', type=bool,
                        help='actually save files?',
                        default=False)
    parser.add_argument('--out_dir', type=str,
                        help='Path to directory outputs should be stored in.',
                        default=LOCAL_DATA_DIR)
    parser.add_argument('--small_seed', type=int,
                        help='Random seed to use for small splitting',
                        default=0)
    parser.add_argument('--full_seed', type=int,
                        help='Random seed to use for full splitting',
                        default=0)
    parser.add_argument('--small_target', type=int,
                        help='Desired number of pixels of each class in small',
                        default=1e3)
    parser.add_argument('--unlabeled', type=int,
                        help='Integer that represents an unlabeled pixel',
                        default=0)

    args = parser.parse_args()

    mask_dir = args.mask_dir
    raster_dir = args.raster_dir
    npy_dir = args.npy_dir
    country = args.country
    out_dir = args.out_dir
    unlabeled = args.unlabeled

    if country in ['germany', 'southsudan']:
        clusters = create_clusters_simple(mask_dir, unlabeled)
        dist_targets = create_dist_split_targets(clusters)
        dist_cluster_splits = dist_split(args.full_seed, clusters, dist_targets, verbose=args.verbose)
        dist_grid_splits = create_grid_splits(dist_cluster_splits)
        if args.save:
            save_grid_splits(dist_grid_splits, out_dir=out_dir, prefix=f"{country}_full_v2_")
    else:
        crop_mapping = np.load(f'/home/data/{country}/{country}_crop_dict.npy').item()
        crop_mapping = {v.lower(): k for k, v in crop_mapping.items()}
        # create maps
        field_to_grids, grid_to_fields = get_field_grid_mappings(raster_dir, npy_dir, country)
        # gets valid fields
        valid_fields = field_to_grids.keys()
        crop_labels  = get_crop_labels(country) 
        csvname = f'/home/data/{country}/{country}_crop.csv'
        csv = load_csv_for_split(csvname, crop_labels, valid_fields, country)
        clusters, missing = create_clusters(csv, field_to_grids, grid_to_fields, raster_dir, crop_mapping, args.verbose)
        even_cluster_splits = split_evenly(args.small_seed, clusters, target_area=args.small_target, verbose=args.verbose)
        even_grid_splits = create_grid_splits(even_cluster_splits)
        if args.verbose:
            check_pixel_counts(mask_dir, country, csv, even_grid_splits)
        if args.save:
            save_grid_splits(even_grid_splits, out_dir=out_dir, prefix=f"{country}_small_")

        dist_targets = create_dist_split_targets(clusters)
        dist_cluster_splits = dist_split(args.full_seed, clusters, dist_targets, verbose=args.verbose)

        dist_grid_splits = create_grid_splits(dist_cluster_splits)
        if args.verbose:
            check_pixel_counts(mask_dir, country, csv, dist_grid_splits)
        if args.save:
            save_grid_splits(dist_grid_splits, out_dir=out_dir, prefix=f"{country}_full_")

