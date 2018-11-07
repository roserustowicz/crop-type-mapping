DATA_FILE_PATH = "/home/data"
CROP_LABELS = ['maize','groundnut', 'rice', 'soya bean', 'sorghum', 'yam', 'sesame', 'beans', 'sunflower', 'chick peas', 'wheat', 'other']
SPLITS = ['train', 'val', 'test']
NON_DL_MODELS = ['logreg', 'random_forest']
DL_MODELS = ['bidir_clstm','fcn','unet']
S1_NUM_BANDS = 2
S2_NUM_BANDS = 10
GRID_SIZE = 64
#MIN_TIMESTAMPS = 25
MIN_TIMESTAMPS = 16

LABEL_DIR = "raster_64x64_npy"
S1_DIR = "s1_npy"
S2_DIR = "s2_64x64_npy"

NROW = 10
