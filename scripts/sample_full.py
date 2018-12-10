from collections import Counter
import numpy as np

def unison_shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

base_dir = '/home/data/ghana/pixel_arrays/full_balanced/raw/'
source_dir = 's1_sample/'
num_classes = 5

# S1
X_tr_fname = 'full_raw_s1_sample_bytime_Xtrain_g2321.npy'
X_val_fname = 'full_raw_s1_sample_bytime_Xval_g305.npy'
X_test_fname = 'full_raw_s1_sample_bytime_Xtest_g364.npy'

y_tr_fname = 'full_raw_s1_sample_bytime_ytrain_g2321.npy'
y_val_fname = 'full_raw_s1_sample_bytime_yval_g305.npy'
y_test_fname = 'full_raw_s1_sample_bytime_ytest_g364.npy'

# S2
#X_tr_fname = 'full_raw_s2_cloud_mask_reverseFalse_bytime_Xtrain_g2321.npy'
#X_val_fname = 'full_raw_s2_cloud_mask_reverseFalse_bytime_Xval_g305.npy'
#X_test_fname = 'full_raw_s2_cloud_mask_reverseFalse_bytime_Xtest_g364.npy'
#
#y_tr_fname = 'full_raw_s2_cloud_mask_reverseFalse_bytime_ytrain_g2321.npy'
#y_val_fname = 'full_raw_s2_cloud_mask_reverseFalse_bytime_yval_g305.npy'
#y_test_fname =  'full_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g364.npy'

# Load in
X_train = np.load(base_dir + source_dir + X_tr_fname)
y_train = np.load(base_dir + source_dir + y_tr_fname)
y_train, X_train = unison_shuffle(y_train, X_train)

X_val = np.load(base_dir + source_dir + X_val_fname)
y_val = np.load(base_dir + source_dir + y_val_fname)
y_val, X_val = unison_shuffle(y_val, X_val)

X_test = np.load(base_dir + source_dir + X_test_fname)
y_test = np.load(base_dir + source_dir + y_test_fname)
y_test, X_test = unison_shuffle(y_test, X_test)

# Train samples per class --> 16247
min_train = 16247
# Val samples per class --> 2183
min_val = 2183
# Test samples per class --> 3393
min_test = 3393

X_train_lst = []
y_train_lst = []
X_val_lst = []
y_val_lst = []
X_test_lst = []
y_test_lst = []

for cl in range(1, num_classes+1):
    tmp_train = X_train[y_train==cl]
    cur_train = tmp_train[:min_train, :]
    X_train_lst.append(cur_train)
    y_train_lst.append(np.ones((min_train,))*cl)

    tmp_val = X_val[y_val==cl]
    cur_val = tmp_val[:min_val, :]
    X_val_lst.append(cur_val)
    y_val_lst.append(np.ones((min_val,))*cl)

    tmp_test = X_test[y_test==cl]
    cur_test = tmp_test[:min_test, :]
    X_test_lst.append(cur_test)
    y_test_lst.append(np.ones((min_test,))*cl)

# Put back together
X_train = np.vstack((X_train_lst))
y_train = np.concatenate((y_train_lst))

X_val = np.vstack((X_val_lst))
y_val = np.concatenate((y_val_lst))

X_test = np.vstack((X_test_lst))
y_test = np.concatenate((y_test_lst))

# Shuffle output
y_train, X_train = unison_shuffle(y_train, X_train)
y_val, X_val = unison_shuffle(y_val, X_val)
y_test, X_test = unison_shuffle(y_test, X_test)

# Save
np.save(base_dir + source_dir + 'sampled/' + X_tr_fname, X_train)
np.save(base_dir + source_dir + 'sampled/' + y_tr_fname, y_train)
np.save(base_dir + source_dir + 'sampled/' + X_val_fname, X_val)
np.save(base_dir + source_dir + 'sampled/' + y_val_fname, y_val)
np.save(base_dir + source_dir + 'sampled/' + X_test_fname, X_test)
np.save(base_dir + source_dir + 'sampled/' + y_test_fname, y_test)
