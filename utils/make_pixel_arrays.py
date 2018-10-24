import os
import numpy as np
import pandas as pd
import json
import argparse
import numpy.ma as ma

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
"""
    Save pixel arrays  # pixels * # features for raw/pca
    
    Args:
      home - (str) the base directory of data

      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'

      data_set - (str) balanced 'small' or unbalanced 'full' dataset

      satellite - (str) satellite to use 's1' 's2' 's1_s2'

      ylabel_dir - (str) dir to load ylabel

      band_order - (str) band order: 'rrrgggbbb', 'rgbrgbrgb'

    Output: 

    saved in HOME/pixel_arrays

"""



def crop_ind(y, name_list = [1, 2, 3, 4, 5]):
    crop_index = [name in name_list for name in y]
    crop_index = np.where(crop_index)
    return crop_index

def vectorize(home = '/home/data/', data_set = 'small', data_type = 'train', satellite = 's1', ylabel_dir = '/home/lijing/ylabel/', band_order = 'rrrgggbbb'):
    
    #X: # of pixels * # of features
    gridded_dir = os.path.join(home, data_set, data_type, satellite)
    gridded_fnames = [f for f in os.listdir(gridded_dir) if (f.endswith('.npy')) and ('mask' not in f)]
    gridded_IDs = [f.split('_')[-1].replace('.npy', '') for f in gridded_fnames]
    gridded_fnames = [gridded_fnames[ID] for ID in np.argsort(gridded_IDs)]
    gridded_IDs = np.array([gridded_IDs[ID] for ID in np.argsort(gridded_IDs)])


    # Match time and gridded
    time_fnames = [f for f in os.listdir(gridded_dir) if f.endswith('.json')]
    time_json = [json.loads(open(os.path.join(gridded_dir,f),'r').read())['dates'] for f in time_fnames]
    time_IDs = [f.split('_')[-1].replace('.json', '') for f in time_fnames]
    time_fnames = [time_fnames[ID] for ID in np.argsort(time_IDs)]
    time_json = [time_json[ID] for ID in np.argsort(time_IDs)]
    time_IDs = np.array([time_IDs[ID] for ID in np.argsort(time_IDs)])

    num_timestamp = 12
        
    Xtemp = np.load(os.path.join(gridded_dir,gridded_fnames[0]))
    
    if satellite == 's1':
        num_band = 2
    elif satellite == 's2':
        num_band = 10
    
    grid_size_a = Xtemp.shape[1]
    grid_size_b = Xtemp.shape[2]

    X = np.zeros((grid_size_a*grid_size_b*len(gridded_fnames),num_band*num_timestamp))
    X[:] = np.nan

    for i in range(len(gridded_fnames)):
        
        X_one = np.load(os.path.join(gridded_dir,gridded_fnames[i]))[0:num_band,:,:]

        Xtemp = np.zeros((num_band, grid_size_a, grid_size_b, num_timestamp))
        Xtemp[:] = np.nan

        time_idx = np.array([np.int64(time.split('-')[1]) for time in time_json[i]])

        for j in np.arange(12)+1:
            Xtemp[:,:,:,j-1] = np.nanmedian(X_one[:,:,:,np.where(time_idx==j)][:,:,:,0,:],axis = 3)
        
        Xtemp = Xtemp.reshape(Xtemp.shape[0],-1,Xtemp.shape[3])
        if band_order == 'rrrgggbbb':
            Xtemp = np.swapaxes(Xtemp, 0, 1).reshape(Xtemp.shape[1],-1)
        elif band_order == 'rgbrgbrgb':
            Xtemp = np.swapaxes(Xtemp, 0, 1)
        
        X[(i*Xtemp.shape[0]):((i+1)*Xtemp.shape[0]), :] = Xtemp

    #y: # of pixels
    mask_dir = [f for f in os.listdir(ylabel_dir) if f.startswith("_".join([data_set,data_type,'s2']))][0]
    y_mask = np.load(os.path.join(ylabel_dir,mask_dir))
    y = y_mask.reshape(-1)   
    crop_id = crop_ind(y)
    
    output_fname = "_".join([data_set, 'raw', satellite, band_order, 'X'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
    np.save(os.path.join(home, 'pixel_arrays', data_set, 'raw', satellite, output_fname), X[crop_id,:][0,:,:])
    
    output_fname = "_".join([data_set, 'raw', satellite, band_order, 'y'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
    np.save(os.path.join(home, 'pixel_arrays', data_set, 'raw', satellite, output_fname), y[crop_id])
    
    return(X[crop_id,:][0,:,:],y[crop_id])

def fill_NA(Xtrain, Xval, Xtest):
    Xtrain_noNA = np.where(np.isnan(Xtrain), ma.array(Xtrain, mask=np.isnan(Xtrain)).mean(axis=0), Xtrain) 
    Xval_noNA = np.where(np.isnan(Xval), ma.array(Xval, mask=np.isnan(Xval)).mean(axis=0), Xval) 
    Xtest_noNA = np.where(np.isnan(Xtest), ma.array(Xtest, mask=np.isnan(Xtest)).mean(axis=0), Xtest) 
    return(Xtrain_noNA, Xval_noNA, Xtest_noNA)

def PCA_scaler(Xtrain, Xval, Xtest, satellite, scaler = StandardScaler()):
    pca = PCA()
    num_samples = 100000
    if Xtrain.shape[0] < num_samples:
        num_samples = Xtrain.shape[0]
    fit_scale = scaler.fit(Xtrain)
    
    Xtrain = fit_scale.transform(Xtrain)
    pca.fit(Xtrain[:num_samples,:])
    num_pcs = np.where(np.cumsum(pca.explained_variance_ratio_)>0.99)[0][0]
    
    pca_X_train = pca.transform(Xtrain)
    Xval = fit_scale.transform(Xval)
    pca_X_val = pca.transform(Xval)

    Xtest = fit_scale.transform(Xtest)
    pca_X_test = pca.transform(Xtest)
    
    output_fname = "_".join([data_set, 'raw', satellite, band_order, 'Xtrain', 'preprocess.npy'])
    np.save(os.path.join(home, 'pixel_arrays', data_set, 'raw', satellite, output_fname), Xtrain)
    
    output_fname = "_".join([data_set, 'raw', satellite, band_order, 'Xval', 'preprocess.npy'])
    np.save(os.path.join(home, 'pixel_arrays', data_set, 'raw', satellite, output_fname), Xval)
    
    output_fname = "_".join([data_set, 'raw', satellite, band_order, 'Xtest', 'preprocess.npy'])
    np.save(os.path.join(home, 'pixel_arrays', data_set, 'raw', satellite, output_fname), Xtest)
    
    
    output_fname = "_".join([data_set, 'pca', satellite, band_order, 'Xtrain', 'num_PCs'+str(num_pcs)+'.npy'])
    np.save(os.path.join(home, 'pixel_arrays', data_set, 'pca', satellite, output_fname), pca_X_train[:,0:num_pcs])
    
    output_fname = "_".join([data_set, 'pca', satellite, band_order, 'Xval', 'num_PCs'+str(num_pcs)+'.npy'])
    np.save(os.path.join(home, 'pixel_arrays', data_set, 'pca', satellite, output_fname), pca_X_val[:,0:num_pcs])
    
    output_fname = "_".join([data_set, 'pca', satellite, band_order, 'Xtest', 'num_PCs'+str(num_pcs)+'.npy'])
    np.save(os.path.join(home, 'pixel_arrays', data_set, 'pca', satellite, output_fname), pca_X_test[:,0:num_pcs])
    
    return(pca_X_train,pca_X_val, pca_X_test)


if __name__ == '__main__':
    
    # Construct the argument parser and parse the arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-hm", "--home", required=False, default='/home/data/')
    arg_parser.add_argument("-c", "--country", required=False, default='Ghana')
    arg_parser.add_argument("-ds", "--dataset", required=False, default='small')
    arg_parser.add_argument("-s", "--satellite", required=False, default='s1')
    arg_parser.add_argument("-y", "--ylabel_dir", required=False, default='/home/data/ylabel/')
    arg_parser.add_argument("-b", "--bandorder", required=False, default='rrrgggbbb')
    args = vars(arg_parser.parse_args())
        
    home = args['home']
    country = args['country']
    data_set = args['dataset']
    satellite = args['satellite']
    ylabel_dir = args['ylabel_dir']
    band_order = args['bandorder']
    
    if satellite == 's1_s2':
        X = {}
        for dataset_temp in ['train','val','test']:
            X[dataset_temp] = {}
            for satellite_temp in ['s1','s2']:
                data_dir = os.path.join(home, 'pixel_arrays', data_set, 'raw', satellite_temp)
                files = os.listdir(data_dir)
                for f in files: 
                    if (f.endswith('.npy')) and ('X'+dataset_temp in f and band_order in f):
                        X[dataset_temp][satellite_temp] = np.load(os.path.join(data_dir, f))

        Xtrain = np.hstack((X['train']['s1'], X['train']['s2']))
        Xval = np.hstack((X['val']['s1'], X['val']['s2']))
        Xtest = np.hstack((X['test']['s1'], X['test']['s2']))

        [Xtrain_noNA, Xval_noNA, Xtest_noNA] = fill_NA(Xtrain, Xval, Xtest)
        [pca_X_train,pca_X_val, pca_X_test] = PCA_scaler(Xtrain_noNA, Xval_noNA, Xtest_noNA, satellite)
            
    else: 
        [Xtrain, ytrain] = vectorize(home, data_set, 'train', satellite, ylabel_dir, band_order)
        [Xval, yval] = vectorize(home, data_set, 'val', satellite, ylabel_dir, band_order)
        [Xtest, ytest] = vectorize(home, data_set, 'test', satellite, ylabel_dir, band_order)
        [Xtrain_noNA, Xval_noNA, Xtest_noNA] = fill_NA(Xtrain, Xval, Xtest)
        [pca_X_train,pca_X_val, pca_X_test] = PCA_scaler(Xtrain_noNA, Xval_noNA, Xtest_noNA, satellite)
