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
from queue import Queue


def get_crop_from_field_id(csv, field_id):
    """ Return the crop grown at `field_id`. """
    return csv[csv['id'] == field_id]['crop']


def cp_files(data_split, prefix, source, dest, suffix):
    """ Copies the files in a data split from a source to a dest.

    Args:
    data_split - (dictionary of lists of ints)  a dictionary mapping a split to a list of the grids in that split
    prefix - (string) any prefix necessary to append before the grid number, currently used to distinguish between s1 and s2
    source - (string) the home directory of the data
    dest - (string) the directory to copy the data to
    suffix - (string) a subdirectory to further divide each split, currently used to subdivide each split into s1 and s2

    """

    for split in ['train', 'val', 'test']:
        for grid in data_split[split]:
            for ext in ['.npy', '.json']:
                filename = prefix + grid + ext
                full_path_file = os.path.join(source, filename)
                if os.path.isfile(full_path_file):
                    full_path_dest = os.path.join(dest, split, suffix, filename)
                    copyfile(full_path_file, full_path_dest)


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
    """ Returns a division of fields and grids such that there is no overlap.

    For each cluster,
    if a field appears in the cluster, then all of the grids that contain the field appear in the cluster
    if a field appears in one cluster, then it only appears within that cluster

    Args:
    csv - (pandas df) a pre-processed csv containing all the valid fields
    field_to_grids - (map: int->list of ints) mapping from a field to a list of all grids the field appears in
    grid_to_fields - (map: int->list of ints) mapping from a grid num to a list of all fields within that grid

    Returns:
    clusters - (list of dictionaries) a list of all unique clusters where each cluster contains information about the grids, fields, and crop counts within that cluster
    missing - (list of ints) a list of all fields that could not be found in the csv, currently checked to be only intercrop fields

    """
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
    """ Preprocesses the csv for datasplitting purposes.

    Args:
    csvname - (string) name of the csv file
    crop_labels - (list of strings) contains all crops under consideration
    valid_fields - (list of ints) contains all available field_nums (npy files exist)

    Returns:
    csv - (pandas df) preprocessed csv with intercrop removed, crops relabeled, and invalid / missing fields removed

    """
    # read in csv and rename crops not in `crop_labels`
    csv = pd.read_csv(csvname)
    csv = csv[csv['crop'] != 'Intercrop'] # remove intercrop
    csv['crop'] = csv['crop'].apply(lambda x: x.lower())
    csv.at[~csv['crop'].isin(crop_labels[:-1]), 'crop'] = 'other'
    csv = csv[csv['id'].isin(valid_fields)]
    # shuffle to ensure no bias for earlier fields
    csv = csv.sample(frac=1, random_state=0)
    print(sum(csv['area']))
    return csv

def split_evenly(seed, clusters, target_area=1e5, verbose=False):
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
            if crop == "other": continue
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

    if verbose: print(area_per_split)

    return cluster_splits

def create_dist_split_targets(csv):
    """ Calculates and creates area targets for a distributed split.

    Args:
    csv - (pandas dataframe) pre-processed csv containing all the valid fields

    Returns:
    targets - (dict of dict of floats) maps each split to the amount of area that should exist for each crop in that split (based on an .8 / .1 / .1 train val test split)

    """
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
            if crop == "other": continue
            crop_count = cluster['crop_counts'][crop]
            if area_per_split['train'][crop] + crop_count > targets['train'][crop]:
                if 'train' in available: available.remove('train')
            if area_per_split['val'][crop] + crop_count > targets['val'][crop]:
                if 'val' in available: available.remove('val')
            if area_per_split['test'][crop] + crop_count > targets['train'][crop]:
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
            for crop in area_per_split[split]:
                print(f"% {crop} : {area_per_split[split][crop] / total_area}")
                print(f"raw {crop} : {area_per_split[split][crop] / 1e5}")
                print(f"target {crop}: {targets[split][crop] / 1e5}\n")


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


def check_pixel_counts(raster_dir, country, csv, grid_splits):
    """ For each class, prints the number of pixels in each split.

    Args:
        raster_dir - (string) directory containing the raster
        country - (string) country being analyzed
        csv - (csv) preprocessed csv containing information
        grid_splits - (dict of set of ints) mapping between a split and all grids that belong to the split

    """
    for split in ['train', 'val', 'test']:
        pixel_counts = defaultdict(float)
        missing = []
        for grid in grid_splits[split]:
            # look up mask for this grid number
            mask_name = country + "_64x64_" + grid + ".tif"
            with rasterio.open(os.path.join(raster_dir, mask_name)) as mask_data:
                mask = mask_data.read()
            # need to separate the mask from the data
            fields, counts = np.unique(mask, return_counts=True)
            for i, field in enumerate(fields):
                if field == 0: continue
                crop = get_crop_from_field_id(csv, field)
                if crop.empty:
                    missing.append(field)
                    pixel_counts['intercrop'] += counts[i]
                    continue
                pixel_counts[crop.item()] += counts[i]
        #print(missing)
        print(f"FOR SPLIT {split}: ")
        for crop in pixel_counts:
            if crop in ['Intercrop', 'other']
            print(f"\tCROP: {crop} has {pixel_counts[crop]} pixels ")


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
    parser.add_argument('--verbose', type=bool,
                        help='error-checking on?',
                        default=False)
    parser.add_argument('--copy', type=bool,
                        help='actually copy files?',
                        default=False)


    args = parser.parse_args()

    raster_dir = args.raster_dir
    npy_dir = args.npy_dir
    country = args.country


    field_to_grids, grid_to_fields = get_field_grid_mappings(raster_dir, npy_dir, country)
    valid_fields = field_to_grids.keys()
    crop_labels  = ['maize','groundnut', 'rice', 'soya bean', 'yam', 'other'] # should be stored in constants.py eventually
    csvname = f'/home/data/{country}_crop.csv'
    csv = load_csv_for_split(csvname, crop_labels, valid_fields)
    clusters, missing = create_clusters(csv, field_to_grids, grid_to_fields)
    even_cluster_splits = split_evenly(1, clusters, verbose=args.verbose)
    even_grid_splits = create_grid_splits(even_cluster_splits)

    if args.verbose:
        check_pixel_counts(raster_dir, country, csv, even_grid_splits)
    if args.copy:
        cp_files(even_grid_splits, prefix="s1_ghana_", source="/home/data/Ghana/s1_64x64_npy", dest="/home/data/small", suffix="s1")
        cp_files(even_grid_splits, prefix="s2_ghana_", source="/home/data/Ghana/s2_64x64_npy", dest="/home/data/small", suffix="s2")

    dist_targets = create_dist_split_targets(csv)
    dist_cluster_splits = dist_split(4, clusters, dist_targets, verbose=args.verbose)

    dist_grid_splits = create_grid_splits(dist_cluster_splits)
    if args.verbose:
        check_pixel_counts(raster_dir, country, csv, dist_grid_splits)
    if args.copy:
        cp_files(dist_grid_splits, prefix="s1_ghana_", source="/home/data/Ghana/s1_64x64_npy", dest="/home/data/full", suffix="s1")
        cp_files(dist_grid_splits, prefix="s2_ghana_", source="/home/data/Ghana/s2_64x64_npy", dest="/home/data/full", suffix="s2")

