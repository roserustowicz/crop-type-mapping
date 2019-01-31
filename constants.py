import numpy as np
import torch.nn as nn
import torch
import os

"""
Constants for file paths
"""
BASE_DIR = os.getenv("HOME")

GCP_DATA_DIR = BASE_DIR + '/croptype_data/data'
LOCAL_DATA_DIR = 'data'

#DATA_DIR = BASE_DIR + '/croptype_data/data'
GHANA_RASTER_DIR = GCP_DATA_DIR + '/ghana/raster/'
GHANA_RASTER_NPY_DIR = GCP_DATA_DIR + '/ghana/raster_npy/'
GHANA_S1_DIR = GCP_DATA_DIR + '/ghana/s1_npy'
GHANA_S2_DIR = GCP_DATA_DIR + '/ghana/s2_npy'
GHANA_HDF5_PATH = LOCAL_DATA_DIR + '/ghana/data.hdf5'

# FOR GHANA
# LOSS_WEIGHT = np.array([5.57, 1.67, 5.88, 8.20])
# LOSS_WEIGHT = torch.tensor(LOSS_WEIGHT)
# LOSS_WEIGHT = (1 - 1 / LOSS_WEIGHT.type(torch.FloatTensor).cuda())

# FOR SOUTH SUDAN
# LOSS_WEIGHT = 1 - np.array([.7265, .1199, .0836, .0710])
# LOSS_WEIGHT = torch.tensor(LOSS_WEIGHT, dtype=torch.float32).cuda()

# FOR TANZANIA
LOSS_WEIGHT = 1- np.array([.53, .12, .1, .04, .04])
LOSS_WEIGHT = torch.tensor(LOSS_WEIGHT, dtype=torch.float32).cuda()

CROP_LABELS = ['maize','groundnut', 'rice', 'soya bean', 'sorghum', 'yam', 'sesame', 'beans', 'sunflower', 'chick peas', 'wheat', 'other']
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

S1_BAND_MEANS = np.array([-11.4, -17.9, 1.16])

S1_BAND_STDS = np.array([3.7, 5.0, 12.2])

S2_BAND_MEANS = np.array([2626.4, 2520.2, 2615.8, 2720.9, 3204.0, 3536.5, 3331.2, 3757.6, 2819.2, 2032.5])

S2_BAND_STDS = np.array([2232.5, 2147.3, 2244.5, 2153.5, 2129.3, 2190.9, 2059.9, 2174.9, 1237.3, 937.4])

CM_LABELS = [0, 1, 2, 3, 4]
GHANA_CROPS = ['groundnut', 'maize', 'rice', 'soya bean']
SOUTHSUDAN_CROPS = ['sorghum', 'maize', 'rice', 'groundnut']
TANZANIA_CROPS = ['maize', 'beans', 'sunflower', 'chickpeas', 'wheat']

S2_BAND_MEANS_wclouds = np.array([2626.4, 2520.2, 2615.8, 2720.9, 3204.0, 3536.5, 3331.2, 3757.6, 2819.2, 2032.5, 1.5])

S2_BAND_STDS_wclouds = np.array([2232.5, 2147.3, 2244.5, 2153.5, 2129.3, 2190.9, 2059.9, 2174.9, 1237.3, 937.4, 1.5])

