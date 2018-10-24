import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import rasterio
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
import json
import numpy.ma as ma
import argparse
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

"""
    Classification by Logistic Regression
    
    Args:
      home - (str) the base directory of data

      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'

      data_set - (str) balanced 'small' or unbalanced 'full' dataset

      satellite - (str) satellite to use 's1' 's2' 's1_s2'

      band_order - (str) band order: 'rrrgggbbb', 'rgbrgbrgb'

      pixeltype - (str) pixeltype: 'raw', 'pca'

"""

def fill_NA(Xtrain, Xtest):
    Xtrain_noNA = np.where(np.isnan(Xtrain), ma.array(Xtrain, mask=np.isnan(Xtrain)).mean(axis=0), Xtrain) 
    #Xval_noNA = np.where(np.isnan(Xval), ma.array(Xval, mask=np.isnan(Xval)).mean(axis=0), Xval) 
    Xtest_noNA = np.where(np.isnan(Xtest), ma.array(Xtest, mask=np.isnan(Xtest)).mean(axis=0), Xtest) 
    return(Xtrain_noNA, Xtest_noNA)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.style.context('seaborn-whitegrid')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 7,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def crop_ind(y, name_list = [1, 2, 3, 4, 5]):
    ##select data based on y: croptypes
    crop_index = [name in name_list for name in y]
    crop_index = np.where(crop_index)
    return crop_index

def classification(X_train, ytrain, X_test, ytest, classifier, satellite, crop_ind_train, crop_ind_test, data_set, pixeltype):
        
    classifier.fit(X_train[crop_ind_train,:][0,:,:], ytrain[crop_ind_train])
    
    #test set
    y_test_pred = classifier.predict(X_test[crop_ind_test,:][0,:,:])
    test_acc = accuracy_score(ytest[crop_ind_test], y_test_pred)
    test_f1 = f1_score(ytest[crop_ind_test], y_test_pred)

    cnf_matrix_test = confusion_matrix(ytest[crop_ind_test], y_test_pred)
    
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_test, classes=['Groundnut', 'Maize', 'Rice', 'Soya Bean', 'Yam'],
                          title = data_set+'_'+pixeltype+'_Confusion matrix:'+satellite+'_test: accuracy: '+str(np.round(test_acc,2))+', f1: '+str(np.round(test_f1,2)))
    plt.savefig(data_set+'_'+pixeltype+'_Confusion matrix:'+satellite+'_test'+'.png',dpi = 300)
    
    
    #training set
    y_train_pred = classifier.predict(X_train[crop_ind_train,:][0,:,:])
    train_acc = accuracy_score(ytrain[crop_ind_train], y_train_pred)
    train_f1 = f1_score(ytrain[crop_ind_train], y_train_pred)
    
    cnf_matrix_train = confusion_matrix(ytrain[crop_ind_train], y_train_pred)
    
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=['Groundnut', 'Maize', 'Rice', 'Soya Bean', 'Yam'],
                          title=data_set+'_'+pixeltype+'_Confusion matrix:'+satellite+'_train: accuracy: '+str(np.round(train_acc,2))+', f1: '+str(np.round(train_f1,2)))
    plt.show()
    plt.savefig(data_set+'_'+pixeltype+'_Confusion matrix:'+satellite+'_train'+'.png',dpi = 300)

    
    return (train_acc, test_acc, train_f1, test_f1, cnf_matrix_train, cnf_matrix_test)



if __name__ == '__main__':
    
    # Construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-hm", "--home", required=False, default='/home/data/')
    arg_parser.add_argument("-c", "--country", required=False, default='Ghana')
    arg_parser.add_argument("-ds", "--dataset", required=False, default='small')
    arg_parser.add_argument("-s", "--satellite", required=False, default='s1')
    arg_parser.add_argument("-type", "--pixeltype", required=False, default='pca')
    arg_parser.add_argument("-b", "--bandorder", required=False, default='rrrgggbbb')
    arg_parser.add_argument("-m", "--classifier", required = False, default = LogisticRegression())
    args = vars(arg_parser.parse_args())
    
    home = args['home']
    country = args['country']
    data_set = args['dataset']
    satellite = args['satellite']
    band_order = args['bandorder']
    pixeltype = args['pixeltype']
    classifier = args['classifier']
    
    if satellite == 's1_s2':
        satellite_list = ['s1','s2']
    else:
        satellite_list = [satellite]
        
    if pixeltype == 'raw':
        X = {}
        y = {}
        for dataset_temp in ['train','test']:
            X[dataset_temp] = {}
            y[dataset_temp] = {}
            for satellite_temp in satellite_list:
                data_dir = os.path.join(home, 'pixel_arrays', data_set, pixeltype, satellite_temp)                
                files = os.listdir(data_dir)
                for f in files: 
                    if (f.endswith('.npy')) and ('X'+dataset_temp in f and band_order in f):
                        X[dataset_temp][satellite_temp] = np.load(os.path.join(data_dir, f))
                    elif (f.endswith('.npy')) and ('y'+dataset_temp in f and band_order in f):
                        y[dataset_temp][satellite_temp] = np.load(os.path.join(data_dir, f))
                
        if satellite == 's1_s2':
            ytrain = y['train']['s1']
            ytest  = y['test']['s1']
            Xtrain = np.hstack((X['train']['s1'],X['train']['s2']))
            Xtest = np.hstack((X['test']['s1'],X['test']['s2']))          
        else:
            ytrain = y['train'][satellite]
            ytest  = y['test'][satellite]
            Xtrain = X['train'][satellite]
            Xtest = X['test'][satellite]
        [Xtrain, Xtest] = fill_NA(Xtrain, Xtest)
        
    elif pixeltype == 'pca':
        X = {}
        y = {}
        for dataset_temp in ['train','test']:
            data_dir = os.path.join(home, 'pixel_arrays', data_set, pixeltype, satellite)                
            files = os.listdir(data_dir)
            for f in files: 
                if (f.endswith('.npy')) and ('X'+dataset_temp in f and band_order in f):
                    X[dataset_temp] = np.load(os.path.join(data_dir, f))
            
            data_dir_y = os.path.join(home, 'pixel_arrays', data_set, 'raw', 's1')
            yfiles = os.listdir(data_dir_y)
            for f in yfiles: 
                if (f.endswith('.npy')) and ('y'+dataset_temp in f and band_order in f):
                    y[dataset_temp] = np.load(os.path.join(data_dir_y, f))
        
        ytrain = y['train']
        ytest  = y['test']
        Xtrain = X['train']
        Xtest =  X['test']  
    
    
    crop_train_id = crop_ind(ytrain)
    crop_test_id = crop_ind(ytest)
    classification(Xtrain, ytrain, Xtest, ytest, classifier, satellite, crop_train_id, crop_test_id, data_set, pixeltype)
 