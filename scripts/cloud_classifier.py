import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix, f1_score

# Set path so that functions can be imported from the utils script
sys.path.insert(0, '../')
from utils.utils import create_categorical_df_col, split_with_group, plot_confusion_matrix 
from models.sklearn import model_fit 

def main_cloud_classifier(data_dir, data_fname):
    """
    Main script to train and evaluate cloud classification methods. 

    Args: 
      data_dir - (str) the directory where the cloud data is
      data_fname - (str) the csv filename that contains the cloud data
                   to be used for training and evaluation of the cloud
                   classification model
    Returns: 
      model - (sklearn model) the model that has been trained for cloud
              classification. Can be used later using model.predict(X)

      this function also displays the train, val, and test metrics 
      (accuracy and f1-score) and plots all confusion matrices
    """

    data_path = os.path.join(data_dir, data_fname)

    # Read in all labeled pixels into dataframe
    df = pd.read_csv(data_path)
    df_sub = df[['poly_id', 'pixel_id', 'class', 'blue', 'green', 'red', 
                 'rded1', 'rded2', 'rded3', 'nir', 'rded4', 'swir1', 'swir2']]

    # Add in categorical representation of class type
    df_sub = create_categorical_df_col(df_sub, 'class', 'class_num')

    X_train, y_train, X_val, y_val, X_test, y_test = split_with_group(df_sub, 'poly_id', 
             0.8, 0.1, slice(3,-1), -1, random_seed=1234, shuffle=True)

    model = model_fit('random_forest', X_train, y_train)

    # Get the predictions from the model
    train_pred_lbls = model.predict(X_train)
    val_pred_lbls = model.predict(X_val)
    test_pred_lbls = model.predict(X_test)

    # Get the accuracy scores 
    tr_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)

    # Get the f1-scores
    tr_f1 = f1_score(y_train, train_pred_lbls, average='micro')
    val_f1 = f1_score(y_val, val_pred_lbls, average='micro')
    test_f1 = f1_score(y_test, test_pred_lbls, average='micro')

    print('Training metrics: accuracy {}, f1-score {}'.format(tr_score, tr_f1))
    print('Validation metrics: accuracy {}, f1-score {}'.format(val_score, val_f1))
    print('Test metrics: accuracy {}, f1-score {}'.format(test_score, test_f1))

    # Get the confusion matrices
    train_cm = confusion_matrix(y_train, train_pred_lbls)
    val_cm = confusion_matrix(y_val, val_pred_lbls)
    test_cm = confusion_matrix(y_test, test_pred_lbls)

    # Plot the confusion matrices
    class_names = ['clear', 'cloud', 'haze', 'shadow']

    for cm, split_type in zip([train_cm, val_cm, test_cm], ['Training', 'Validation', 'Test']):
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names,
                      title='{} confusion matrix, without normalization'.format(split_type))
        plt.show()
    return model

def save_cloud_masks(s2_data_dir, model, verbose):
    """
    Uses a trained model to predict and save cloud masks for sentinel-2 data
    stored in .npy format

    Args: 
      s2_data_dir - (str) the directory where .npy files associates with 
                    sentinel-2 imagery is stored. within this directory, a new
                    directory 'cloud_masks' will be created and will store the 
                    saved output cloud mask arrays for each .npy file
      model - (sklearn model) a trained model used for cloud prediction
      verbose - (boolean) whether to print outputs of the function

    Returns: 
      this function saves cloud masks as output in the folder
      's2_data_dir/cloud_masks' as .npy files. Given a npy file as 
      input that consists of n timestamps, the output .npy file consists
      of n corresponding cloud masks
    """
    data_fnames = [os.path.join(s2_data_dir, fname) for fname in os.listdir(s2_data_dir) if fname.endswith('.npy')]
    data_fnames.sort()    

    mask_dir = os.path.join(s2_data_dir, 'cloud_masks')
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    for fname in data_fnames:
        arr = np.load(fname)
        mask_arr = np.zeros((arr.shape[1], arr.shape[2], arr.shape[3]))

        for timestamp in range(arr.shape[3]):
            cur_img = arr[:,:,:,timestamp]
            # put channels last
            cur_img = np.transpose(cur_img, (1, 2, 0))
            # flatten into num smaples x num features (10 channels)
            cur_img = np.reshape(cur_img, [-1, 10]) 
            # use cloud model to predict the cloud mask
            cur_pred = model.predict(cur_img)
            # reshape prediction into image shape
            cur_pred = np.reshape(cur_pred, (64, 64))
            # save prediction for this timestamp into cube
            mask_arr[:, :, timestamp] = cur_pred

        out_fname = os.path.join(mask_dir, fname.split('/')[-1].split('.')[0] + '_mask.npy')
        if verbose:
            print("Mask for {} saved to {}".format(fname, out_fname))
        np.save(out_fname, mask_arr)

if __name__ == '__main__':
    data_dir = '/home/data/clouds/'
    data_fname = 'clean_samples.csv'
    s2_data_dir = '/home/data/full/train/s2'
    verbose = 1

    # Train cloud classificatio model
    model = main_cloud_classifier(data_dir, data_fname)
    # Use model to save out cloud masks
    save_cloud_masks(s2_data_dir, model, verbose)
