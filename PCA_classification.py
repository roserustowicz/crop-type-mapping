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

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

def vectorize(home, data_set, data_type, satellite, ylabel_dir, cloud_mask = 0):
    satellite0 = np.copy(satellite)
    #X: # of pixels * # of features
    satellite = 's2'
    gridded_dir = os.path.join(home, data_set, data_type, satellite)
    gridded_fnames = [f for f in os.listdir(gridded_dir) if f.endswith('.npy')]
    gridded_IDs = [f.split('_')[-1].replace('.npy', '') for f in gridded_fnames]

    clouds_mask_dir = os.path.join(home, data_set, data_type, satellite, 'cloud_masks')
    clouds_mask_fnames = [f for f in os.listdir(clouds_mask_dir) if f.endswith('.npy')]
    clouds_mask_IDs = [f.split('_')[2].replace('.npy', '') for f in clouds_mask_fnames]
    clouds_mask_order = [np.where([gridded_ID==clouds_mask_ID for clouds_mask_ID in clouds_mask_IDs])[0][0] for gridded_ID in gridded_IDs]

    
    # Match time and gridded
    time_fnames = [f for f in os.listdir(gridded_dir) if f.endswith('.json')]
    time_json = [json.loads(open(os.path.join(gridded_dir,f),'r').read())['dates'] for f in time_fnames]
    time_IDs = [f.split('_')[-1].replace('.json', '') for f in time_fnames]
    time_order = [np.where([gridded_ID==time_ID for time_ID in time_IDs])[0][0] for gridded_ID in gridded_IDs]

    unique,pos = np.unique(time_json,return_inverse=True) #Finds all unique elements and their positions
    counts = np.bincount(pos)                     #Count the number of each unique element
    maxpos = counts.argmax() 

    if unique.dtype=='O':
        time_standard = time_json[np.where([unique[maxpos]==time for time in time_json])[0][0]]
    else:
        time_standard = time_json[0]
    num_timestamp = len(time_standard)
        
    Xtemp = np.load(os.path.join(gridded_dir,gridded_fnames[0]))
    num_band = Xtemp.shape[0]
    grid_size_a = Xtemp.shape[1]
    grid_size_b = Xtemp.shape[2]

    X = np.zeros((grid_size_a*grid_size_b*len(gridded_fnames),num_band*num_timestamp))
    X[:] = np.nan

    for i in range(len(gridded_fnames)):

        Xtemp = np.zeros((num_band, grid_size_a, grid_size_b, num_timestamp))
        Xtemp[:] = np.nan

        time_idx = np.zeros(len(time_json[time_order[i]]))

        cloud_name = clouds_mask_fnames[clouds_mask_order[i]]

        for j in range(len(time_json[time_order[i]])): 
            time1 = time_json[time_order[i]][j]
            try:
                time_idx[j] = np.where([time1==time2 for time2 in time_standard])[0][0]
            except:
                time_idx[j] = np.nan


        X_one = np.load(os.path.join(gridded_dir,gridded_fnames[i]))
        ## mask by cloudmask
        cloud_mask_one = np.load(os.path.join(clouds_mask_dir,cloud_name))
        cloud_mask_multi = np.zeros(X_one.shape)
        for ii in range(X_one.shape[0]):
            cloud_mask_multi[ii,:,:,:]=cloud_mask_one

        X_newOne = np.ma.masked_where(cloud_mask_multi > cloud_mask, X_one)
        X_newOne = X_newOne.filled(fill_value=np.nan)

        Xtemp[:,:,:,np.int64(time_idx)[~np.isnan(time_idx)]] = X_newOne[:,:,:,~np.isnan(time_idx)]
        Xtemp = Xtemp.reshape(Xtemp.shape[0],-1,Xtemp.shape[3])
        Xtemp = np.swapaxes(Xtemp, 0, 1).reshape(Xtemp.shape[1],-1)
        X[(i*Xtemp.shape[0]):((i+1)*Xtemp.shape[0]), :] = Xtemp

    #y: # of pixels
    mask_dir = [f for f in os.listdir(ylabel_dir) if f.startswith("_".join([data_set,data_type,'s2']))][0]
    y_mask = np.load(os.path.join(ylabel_dir,mask_dir))
    y = y_mask.reshape(-1)   
    
    if satellite != 's2':
        X_all = {}
        gridded_IDs_all = {}
        X_all['s2'] = X
        gridded_IDs_all['s2'] = gridded_IDs
        
        ## Calculate s1
        satellite = 's1'
        gridded_dir = os.path.join(home, data_set, data_type, satellite)
        gridded_fnames = [f for f in os.listdir(gridded_dir) if f.endswith('.npy')]
        gridded_IDs = [f.split('_')[-1].replace('.npy', '') for f in gridded_fnames]

        ## Match time and gridded
        time_fnames = [f for f in os.listdir(gridded_dir) if f.endswith('.json')]
        time_json = [json.loads(open(os.path.join(gridded_dir,f),'r').read())['dates'] for f in time_fnames]
        time_IDs = [f.split('_')[-1].replace('.json', '') for f in time_fnames]
        time_order = [np.where([gridded_ID==time_ID for time_ID in time_IDs])[0][0] for gridded_ID in gridded_IDs]

        unique,pos = np.unique(time_json,return_inverse=True) #Finds all unique elements and their positions
        counts = np.bincount(pos)                     #Count the number of each unique element
        maxpos = counts.argmax() 
            
        if unique.dtype=='O':
            time_standard = time_json[np.where([unique[maxpos]==time for time in time_json])[0][0]]
        else:
            time_standard = time_json[0]
        num_timestamp = len(time_standard)

        Xtemp = np.load(os.path.join(gridded_dir,gridded_fnames[0]))
        num_band = Xtemp.shape[0]
        grid_size_a = Xtemp.shape[1]
        grid_size_b = Xtemp.shape[2]

        X = np.zeros((grid_size_a*grid_size_b*len(gridded_fnames),num_band*num_timestamp))
        X[:] = np.nan

        for i in range(len(gridded_fnames)):
            Xtemp = np.zeros((num_band, grid_size_a, grid_size_b, num_timestamp))
            Xtemp[:] = np.nan
            time_idx = np.zeros(len(time_json[time_order[i]]))
            for j in range(len(time_json[time_order[i]])): 
                time1 = time_json[time_order[i]][j]
                try:
                    time_idx[j] = np.where([time1==time2 for time2 in time_standard])[0][0]
                except:
                    time_idx[j] = np.nan
                
            Xtemp[:,:,:,np.int64(time_idx)[~np.isnan(time_idx)]] = np.load(os.path.join(gridded_dir,gridded_fnames[i]))[:,:,:,~np.isnan(time_idx)]
            Xtemp = Xtemp.reshape(Xtemp.shape[0],-1,Xtemp.shape[3])
            Xtemp = np.swapaxes(Xtemp, 0, 1).reshape(Xtemp.shape[1],-1)
            X[(i*Xtemp.shape[0]):((i+1)*Xtemp.shape[0]), :] = Xtemp

        X_all[satellite] = X
        gridded_IDs_all[satellite] = gridded_IDs

        # Merge 2 datasets   
        s2_ind_in_s1 = [np.where([gridded_ID_s2 == gridded_ID_s1 for gridded_ID_s1 in gridded_IDs_all['s1']])[0][0] for gridded_ID_s2 in gridded_IDs_all['s2']]
        X1 = [X_all['s1'][idx*grid_size_a*grid_size_b:(idx+1)*grid_size_a*grid_size_b,:] for idx in s2_ind_in_s1]
        X1 = np.array(X1).reshape(-1,X1[0].shape[1])
        
        if satellite0 == 's1':
            X = np.copy(X1)
        else:
            X = {}
            X['s1'] = X1
            X['s2'] = X_all['s2']
        
        #y: # of pixels
        mask_dir = [f for f in os.listdir(ylabel_dir) if f.startswith("_".join([data_set,data_type,'s2']))][0]
        y_mask = np.load(os.path.join(ylabel_dir,mask_dir))
        y = y_mask.reshape(-1)   

    return(X,y)

def crop_ind(y, name_list = ['Maize', 'Groundnut', 'Rice', 'Soya Bean']):
    crop_index = [name in name_list for name in y]
    crop_index = np.where(crop_index)
    return crop_index


def fill_NA(Xtrain, Xtest):
    Xtrain_noNA = np.where(np.isnan(Xtrain), ma.array(Xtrain, mask=np.isnan(Xtrain)).mean(axis=0), Xtrain) 
    Xtest_noNA = np.where(np.isnan(Xtest), ma.array(Xtest, mask=np.isnan(Xtest)).mean(axis=0), Xtest) 
    return(Xtrain_noNA, Xtest_noNA)

def PCA_scaler(Xtrain, Xtest, scaler):
    pca = PCA()
    fit_scale = scaler.fit(Xtrain)
    
    Xtrain = fit_scale.transform(Xtrain)
    pca_X_train = pca.fit_transform(Xtrain)

    Xtest = fit_scale.transform(Xtest)
    pca_X_test = pca.transform(Xtest)
    
    return(pca_X_train, pca_X_test)


def classification(pca_X_train, ytrain, pca_X_test, ytest, classifier, satellite, num_comp):
    if satellite == 's1_s2':
        num_PC_s1 = num_comp[0]
        num_PC_s2 = num_comp[1]
        
        classifier.fit(np.hstack([pca_X_train[0][:,0:num_PC_s1],pca_X_train[1][:,0:num_PC_s2]]), ytrain)
        y_pred = classifier.predict(np.hstack([pca_X_test[0][:,0:num_PC_s1],pca_X_test[1][:,0:num_PC_s2]]))
        y_train_pred = classifier.predict(np.hstack([pca_X_train[0][:,0:num_PC_s1],pca_X_train[1][:,0:num_PC_s2]]))
        val_acc = accuracy_score(ytest, y_pred)
        val_f1 = f1_score(ytest, y_pred)
        cnf_matrix_val = confusion_matrix(ytest, y_pred)
        #print(cnf_matrix_val)
        train_acc = accuracy_score(ytrain, y_train_pred)
        train_f1 = f1_score(ytrain, y_train_pred)
        cnf_matrix_train = confusion_matrix(ytrain, y_train_pred)
        #print(cnf_matrix_train)

    else: 
        num_PC = num_comp
        param_grid = {
                 'min_samples_split': [2, 5, 10, 20]
        }
        classifier.fit(pca_X_train[:,0:num_PC], ytrain)
        y_pred = classifier.predict(pca_X_test[:,0:num_PC])
        y_train_pred = classifier.predict(pca_X_train[:,0:num_PC])
        val_acc = accuracy_score(ytest, y_pred)
        val_f1 = f1_score(ytest, y_pred)
        cnf_matrix_val = confusion_matrix(ytest, y_pred)
        #print(cnf_matrix_val)
        train_acc = accuracy_score(ytrain, y_train_pred)
        train_f1 = f1_score(ytrain, y_train_pred)
        cnf_matrix_train = confusion_matrix(ytrain, y_train_pred)
        #print(cnf_matrix_train)  
    return (train_acc, val_acc, train_f1, val_f1, cnf_matrix_train, cnf_matrix_val)
    
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
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 5,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
if __name__ == '__main__':
    
    # Construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-hm", "--home", required=False, default='/home/data/')
    arg_parser.add_argument("-ds", "--dataset", required=False, default='full')
    arg_parser.add_argument("-train", "--trainset", required=False, default='train')
    arg_parser.add_argument("-test", "--testset", required=False, default='test')
    arg_parser.add_argument("-s", "--satellite", required=False, default='s1')
    arg_parser.add_argument("-ylabel_dir", "--ylabel_dir", required=False, default='/home/lijing/ylabel/')
    
    
    scaler_map = {'standard': StandardScaler(),'robust': RobustScaler()}
    classifier_map = {'rf' : RandomForestClassifier(oob_score = True),'LogR' : LogisticRegression()}
    
    arg_parser.add_argument("-scale", "--scale_function", required = False, default = StandardScaler(), choices = scaler_map.keys())
    arg_parser.add_argument("-m", "--classifier", required = False, default = RandomForestClassifier(oob_score = True) , choices = classifier_map.keys())
    arg_parser.add_argument("-sv", "--save_dataset", required=False, default = False, action='store_true')
    args = vars(arg_parser.parse_args())
    
    home = args['home']
    data_set = args['dataset']
    train = args['trainset']
    test = args['testset']
    satellite = args['satellite']
    ylabel_dir = args['ylabel_dir']
    scale_func = args['scale_function']
    classifier = args['classifier']
    save_dataset = args['save_dataset']
    
    ## Return accuracy rate and confusion matrix
    [Xtrain, ytrain] = vectorize(home, data_set, train, satellite, ylabel_dir)
    [Xtest, ytest] = vectorize(home, data_set, test, satellite, ylabel_dir)
    
    
    if save_dataset:
        if satellite == 's1_s2':
            fname = data_set+'_s1_X'+train+'.npy'
            np.save(fname, Xtrain['s1'])
            fname = data_set+'_s1_X'+test+'.npy'
            np.save(fname, Xtest['s1'])
            fname = data_set+'_s2_X'+train+'.npy'
            np.save(fname, Xtrain['s2'])
            fname = data_set+'_s2_X'+test+'.npy'
            np.save(fname, Xtest['s2'])
            fname = data_set+'_'+satellite+'_'+'y'+train+'.npy'
            np.save(fname, ytrain)
            fname = data_set+'_'+satellite+'_'+'y'+test+'.npy'
            np.save(fname, ytest)
            
        else: 
            fname = data_set+'_'+satellite+'_'+'X'+train+'.npy'
            np.save(fname, Xtrain)
            fname = data_set+'_'+satellite+'_'+'X'+test+'.npy'
            np.save(fname, Xtest)
            fname = data_set+'_'+satellite+'_'+'y'+train+'.npy'
            np.save(fname, ytrain)
            fname = data_set+'_'+satellite+'_'+'y'+test+'.npy'
            np.save(fname, ytest)
        
        
        
    
    crop_train_ind = crop_ind(ytrain)
    crop_test_ind = crop_ind(ytest)
    
    if satellite == 's1_s2':
        [Xtrain_noNA_s1, Xtest_noNA_s1] = fill_NA(Xtrain['s1'], Xtest['s1'])
        [Xtrain_noNA_s2, Xtest_noNA_s2] = fill_NA(Xtrain['s2'], Xtest['s2'])
        [pca_Xtrain_s1, pca_Xtest_s1] = PCA_scaler(Xtrain_noNA_s1, Xtest_noNA_s1, scale_func)
        [pca_Xtrain_s2, pca_Xtest_s2] = PCA_scaler(Xtrain_noNA_s2, Xtest_noNA_s2, scale_func)  
        pca_Xtrain = [pca_Xtrain_s1[crop_train_ind,:][0,:,:], pca_Xtrain_s2[crop_train_ind,:][0,:,:]]
        pca_Xtest = [pca_Xtest_s1[crop_test_ind,:][0,:,:], pca_Xtest_s2[crop_test_ind,:][0,:,:]]
        [train_acc, val_acc, train_f1, val_f1, cnf_matrix_train, cnf_matrix_val]  = classification(pca_Xtrain, ytrain[crop_train_ind], pca_Xtest, ytest[crop_test_ind], classifier, satellite, num_comp = [60,60])
    else:
        [Xtrain_noNA, Xtest_noNA] = fill_NA(Xtrain, Xtest)
        [pca_Xtrain, pca_Xtest] = PCA_scaler(Xtrain_noNA, Xtest_noNA, scale_func)
        [train_acc, val_acc, train_f1, val_f1, cnf_matrix_train, cnf_matrix_val] = classification(pca_Xtrain[crop_train_ind,:][0,:,:], ytrain[crop_train_ind], pca_Xtest[crop_test_ind,:][0,:,:], ytest[crop_test_ind], classifier, satellite, num_comp = 60)
    
    
        
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=np.unique(ytrain[crop_train_ind]),
                          title=data_set+'_Confusion matrix:'+satellite+'_train: accuracy: '+str(np.round(train_acc,2))+', f1: '+str(np.round(train_f1,2)))
    plt.show()
    plt.savefig(data_set+'_Confusion matrix:'+satellite+'_train'+'.png',dpi = 300)
    
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix_val, classes=np.unique(ytest[crop_test_ind]),
                          title=data_set+'_Confusion matrix:'+satellite+'_test: accuracy: '+str(np.round(val_f1,2))+', f1: '+str(np.round(val_f1,2)))
    plt.show()
    plt.savefig(data_set+'_Confusion matrix:'+satellite+'_test'+'.png',dpi = 300)

    
    
