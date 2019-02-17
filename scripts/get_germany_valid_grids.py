"""
Gets overlap of valid_grids (those in the data folder) with grids in the train, test, val splits
and updates train, test, and val splits to only contain valid grids!
"""
import numpy as np
import pickle
import os

valid_grids = np.load('valid_grids.npy')

tile_dir = '/home/roserustowicz/croptype_data_local/data/germany/tileids'
train_grids = os.path.join(tile_dir, 'train_fold0.tileids')
test_grids = os.path.join(tile_dir, 'test_fold0.tileids')
eval_grids = os.path.join(tile_dir, 'eval.tileids')

# filter grids by those contained in the valid_grids list
with open(train_grids, "r") as f:
    train_grids = [line.strip() for line in f]
 
with open(test_grids, "r") as f:
    test_grids = [line.strip() for line in f]

with open(eval_grids, "r") as f:
    eval_grids = [line.strip() for line in f]

train_filt = [f for f in train_grids if f in valid_grids]
test_filt = [f for f in test_grids if f in valid_grids]
eval_filt = [f for f in eval_grids if f in valid_grids]

print('grids: ', len(train_grids) + len(test_grids) + len(eval_grids))
print('filtered: ', len(train_filt) + len(test_filt) + len(eval_filt))

# Write to pickle file 
with open(os.path.join(tile_dir, "germany_full_train"), "wb") as outfile:
   pickle.dump(train_filt, outfile)

with open(os.path.join(tile_dir, "germany_full_test"), "wb") as outfile:
   pickle.dump(eval_filt, outfile)

with open(os.path.join(tile_dir, "germany_full_val"), "wb") as outfile:
   pickle.dump(test_filt, outfile)
