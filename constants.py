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
#S1_BAND_MEANS = np.array([-11.4, -17.9, 1.16])
#S1_BAND_STDS = np.array([3.7, 5.0, 12.2])
#S2_BAND_MEANS = np.array([2626.4, 2520.2, 2615.8, 2720.9, 3204.0, 3536.5, 3331.2, 3757.6, 2819.2, 2032.5])
#S2_BAND_STDS = np.array([2232.5, 2147.3, 2244.5, 2153.5, 2129.3, 2190.9, 2059.9, 2174.9, 1237.3, 937.4])

S1_BAND_MEANS = { 'ghana': np.array([-10.50, -17.24, 1.17]), 
                  'southsudan': np.array([-9.02, -15.26, 1.15]), 
                  'tanzania': np.array([-9.80, -17.05, 1.30])} 

S1_BAND_STDS = { 'ghana': np.array([3.57, 4.86, 5.60]),
                 'southsudan': np.array([4.49, 6.68, 21.75]),
                 'tanzania': np.array([3.53, 4.78, 16.61])} 

S2_BAND_MEANS = { 'ghana': np.array([2620.00, 2519.89, 2630.31, 2739.81, 3225.22, 3562.64, 3356.57, 3788.05, 2915.40, 2102.65]),
                  'southsudan': np.array([2119.15, 2061.95, 2127.71, 2277.60, 2784.21, 3088.40, 2939.33, 3308.03, 2597.14, 1834.81]),
                  'tanzania': np.array([2551.54, 2471.35, 2675.69, 2799.99, 3191.33, 3453.16, 3335.64, 3660.05, 3182.23, 2383.79])}

S2_BAND_STDS = { 'ghana': np.array([2171.62, 2085.69, 2174.37, 2084.56, 2058.97, 2117.31, 1988.70, 2099.78, 1209.48, 918.19]),
                 'southsudan': np.array([2113.41, 2026.64, 2126.10, 2093.35, 2066.81, 2114.85, 2049.70, 2111.51, 1320.97, 1029.58]), 
                 'tanzania': np.array([2290.97, 2204.75, 2282.90, 2214.60, 2182.51, 2226.10, 2116.62, 2210.47, 1428.33, 1135.21])}

S2_BAND_MEANS_cldfltr = { 'ghana': np.array([1362.68, 1317.62, 1410.74, 1580.05, 2066.06, 2373.60, 2254.70, 2629.11, 2597.50, 1818.43]),
                  'southsudan': np.array([1137.58, 1127.62, 1173.28, 1341.70, 1877.70, 2180.27, 2072.11, 2427.68, 2308.98, 1544.26]),
                  'tanzania': np.array([1148.76, 1138.87, 1341.54, 1517.01, 1937.15, 2191.31, 2148.05, 2434.61, 2774.64, 2072.09])}

S2_BAND_STDS_cldfltr = { 'ghana': np.array([511.19, 495.87, 591.44, 590.27, 745.81, 882.05, 811.14, 959.09, 964.64, 809.53]),
                 'southsudan': np.array([548.64, 547.45, 660.28, 677.55, 896.28, 1066.91, 1006.01, 1173.19, 1167.74, 865.42]), 
                 'tanzania': np.array([462.40, 449.22, 565.88, 571.42, 686.04, 789.04, 758.31, 854.39, 1071.74, 912.79])}

CM_LABELS = { 'ghana': [0, 1, 2, 3], 'southsudan': [0, 1, 2, 3], 'tanzania': [0, 1, 2, 3, 4] }

CROPS = { 'ghana': ['groundnut', 'maize', 'rice', 'soya bean'], 'southsudan': ['sorghum', 'maize', 'rice', 'groundnut'], 'tanzania': ['maize', 'beans', 'sunflower', 'chickpeas', 'wheat'] }
