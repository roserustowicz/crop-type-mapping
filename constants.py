import numpy as np
import torch.nn as nn
import torch
import os

"""
Constants for file paths
"""
BASE_DIR = os.getenv("HOME")

GCP_DATA_DIR = BASE_DIR + '/croptype_data/data'
#LOCAL_DATA_DIR = 'data'
LOCAL_DATA_DIR = BASE_DIR + '/croptype_data_local/data'

GHANA_RASTER_DIR = GCP_DATA_DIR + '/ghana/raster/'
GHANA_RASTER_NPY_DIR = GCP_DATA_DIR + '/ghana/raster_npy/'
GHANA_S1_DIR = GCP_DATA_DIR + '/ghana/s1_npy'
GHANA_S2_DIR = GCP_DATA_DIR + '/ghana/s2_npy'
GHANA_HDF5_PATH = LOCAL_DATA_DIR + '/ghana/data.hdf5'

# LOSS WEIGHTS
GHANA_LOSS_WEIGHT = 1 - np.array([.1795, .5988, .1701, .1220])
GHANA_LOSS_WEIGHT = torch.tensor(GHANA_LOSS_WEIGHT, dtype=torch.float32).cuda()

SSUDAN_LOSS_WEIGHT = 1 - np.array([.7265, .1199, .0836, .0710])
SSUDAN_LOSS_WEIGHT = torch.tensor(SSUDAN_LOSS_WEIGHT, dtype=torch.float32).cuda()

TANZ_LOSS_WEIGHT = 1- np.array([.53, .12, .1, .04, .04])
TANZ_LOSS_WEIGHT = torch.tensor(TANZ_LOSS_WEIGHT, dtype=torch.float32).cuda()

LOSS_WEIGHT = { 'ghana': GHANA_LOSS_WEIGHT, 'southsudan': SSUDAN_LOSS_WEIGHT, 'tanzania': TANZ_LOSS_WEIGHT }

#CROP_LABELS = ['maize','groundnut', 'rice', 'soya bean', 'sorghum', 'yam', 'sesame', 'beans', 'sunflower', 'chick peas', 'wheat', 'other']
SPLITS = ['train', 'val', 'test']
NON_DL_MODELS = ['logreg', 'random_forest']
DL_MODELS = ['bidir_clstm','fcn', 'unet', 'fcn_crnn']
S1_NUM_BANDS = 3
S2_NUM_BANDS = 10
GRID_SIZE = 64
MIN_TIMESTAMPS = 16

LABEL_DIR = "raster_npy"
S1_DIR = "s1_npy"
S2_DIR = "s2_npy"
NROW = 8

INT_POWER_EXP = ["hidden_dims"]
REAL_POWER_EXP = ["weight_decay", "lr"]
INT_HP = ['batch_size', 'crnn_num_layers', 'gamma', 'patience']
FLOAT_HP = ['momentum', 'weight_scale', 'percent_of_dataset']
STRING_HP = ['optimizer']
BOOL_HP = ['use_s1', 'use_s2', 'sample_w_clouds', 'include_clouds', 'include_doy', 'bidirectional', 'least_cloudy', 's2_num_bands']

HPS = [INT_POWER_EXP, REAL_POWER_EXP, INT_HP, FLOAT_HP, STRING_HP, BOOL_HP]

# GHANA
S1_BAND_MEANS = np.array([-11.4, -17.9, 1.16])
S1_BAND_STDS = np.array([3.7, 5.0, 12.2])
S2_BAND_MEANS = np.array([2626.4, 2520.2, 2615.8, 2720.9, 3204.0, 3536.5, 3331.2, 3757.6, 2819.2, 2032.5])
S2_BAND_STDS = np.array([2232.5, 2147.3, 2244.5, 2153.5, 2129.3, 2190.9, 2059.9, 2174.9, 1237.3, 937.4])

CM_LABELS = { 'ghana': [0, 1, 2, 3], 'southsudan': [0, 1, 2, 3], 'tanzania': [0, 1, 2, 3, 4] }

CROPS = { 'ghana': ['groundnut', 'maize', 'rice', 'soya bean'], 'southsudan': ['sorghum', 'maize', 'rice', 'groundnut'], 'tanzania': ['maize', 'beans', 'sunflower', 'chickpeas', 'wheat'] }

#S2_BAND_MEANS_wclouds = np.array([2626.4, 2520.2, 2615.8, 2720.9, 3204.0, 3536.5, 3331.2, 3757.6, 2819.2, 2032.5, 1.5])
#S2_BAND_STDS_wclouds = np.array([2232.5, 2147.3, 2244.5, 2153.5, 2129.3, 2190.9, 2059.9, 2174.9, 1237.3, 937.4, 1.5])

