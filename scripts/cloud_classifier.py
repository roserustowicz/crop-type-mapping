import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
import rasterio
import pandas as pd
import matplotlib.cm as cm
import sys
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, '../')
from utils.utils import create_categorical_df_col, split_with_group, model_fit, plot_confusion_matrix 


def main_cloud_classifier(data_dir, data_fname):
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

if __name__ == '__main__':
    data_dir = '/home/data/clouds/'
    data_fname = 'clean_samples.csv'

    main_cloud_classifier(data_dir, data_fname)
