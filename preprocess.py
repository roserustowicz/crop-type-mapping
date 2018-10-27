"""

File that houses all functions used to format, preprocess, or manipulate the data.

Consider this essentially a util library specifically for data manipulation.

"""

import os
import numpy as np
import pandas as pd
import json
import argparse
import numpy.ma as ma

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def retrieve_mask(grid_name):
    """ Return the mask of the grid specified by grid_name.

    Args:
        grid_name - (string) string representation of the grid number

    Returns:
        mask - (npy arr) mask containing labels for each pixel
    """
    mask = None
    return mask

def retrieve_grid(grid_name):
    """ Retrieves a concatenation of the s1 and s2 values of the grid specified.

    Args:
        grid_name - (string) string representation of the grid number

    Returns:
        grid - (npy array) concatenation of the s1 and s2 values of the grid over time
    """
    grid = None
    return grid

def preprocess_grid(grid, model_name):
    """ Returns a preprocessed version of the grid based on the model.

    Args:
        grid - (npy array) concatenation of the s1 and s2 values of the grid
        model_name - (string) type of model (ex: "C-LSTM")

    """

    if model_name == "C-LSTM":
        return preprocessForCLSTM(grid)

def sample_timeseries(img_stack, num_samples, cloud_stack=None, remap_clouds=True,
                      reverse=False, seed=None, save=False):
    """
    Args: 
      img_stack - (numpy array) [bands x rows x cols x timestamps], temporal stack of images
      num_samples - (int) number of samples to sample from the img_stack (and cloud_stack)
                     and must be <= the number of timestamps
      cloud_stack - (numpy array) [rows x cols x timestamps], temporal stack of cloud masks
      reverse - (boolean) take 1 - probabilities, encourages cloudy images to be sampled
      seed - (int) a random seed for sampling 

    Returns: 
      sampled_img_stack - (numpy array) [bands x rows x cols x num_samples], temporal stack 
                          of sampled images
      sampled_cloud_stack - (numpy array) [rows x cols x num_samples], temporal stack of
                            sampled cloud masks
    """
    timestamps = img_stack.shape[3]
    np.random.seed(seed)

    # Given a stack of cloud masks, remap it and use to compute scores 
    if isinstance(cloud_stack,np.ndarray):
        # Remap cloud mask values so clearest pixels have highest values
        # Rank by clear, shadows, haze, clouds
        # clear = 0 --> 3, clouds = 1  --> 0, shadows = 2 --> 2, haze = 3 --> 1
        remap_cloud_stack = np.zeros_like((cloud_stack))
        remap_cloud_stack[cloud_stack == 0] = 3
        remap_cloud_stack[cloud_stack == 2] = 2
        remap_cloud_stack[cloud_stack == 3] = 1
        
        scores = np.mean(remap_cloud_stack, axis=(0, 1))

    else:
        print('NO INPUT CLOUD MASKS. USING RANDOM SAMPLING!')
        scores = np.ones((timestamps,))

    if reverse:
        scores = 3 - scores

    # Compute probabilities of scores with softmax
    probabilities = softmax(scores) 

    # Sample from timestamp indices according to probabilities
    samples = np.random.choice(timestamps, size=num_samples, replace=False, p=probabilities)
    # Sort samples to maintain sequential ordering
    samples.sort()

    # Use sampled indices to sample image and cloud stacks 
    sampled_img_stack = img_stack[:, :, :, samples]
    
    if isinstance(cloud_stack, np.ndarray):
        if remap_clouds:
            sampled_cloud_stack = remap_cloud_stack[:, :, samples]
        else:
            sampled_cloud_stack = cloud_stack[:, :, samples]
        return sampled_img_stack, sampled_cloud_stack
    else:
        return sampled_img_stack, None  




def vectorize(home, country, data_set, satellite, ylabel_dir, band_order= 'byband'):
    """
    Save pixel arrays  # pixels * # features for raw
    
    Args:
      home - (str) the base directory of data

      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'

      data_set - (str) balanced 'small' or unbalanced 'full' dataset

      satellite - (str) satellite to use 's1' 's2' 's1_s2'

      ylabel_dir - (str) dir to load ylabel

      band_order - (str) band order: 'byband', 'bytime'

    Output: 

    saved in HOME/pixel_arrays

    """

    satellite_original = np.copy(satellite)

    X_total3types = {}
    y_total3types = {}

    for data_type in ['train','val','test']:

        if satellite_original == 's1':
            num_band = 2
            satellite_list = ['s1']
        elif satellite_original == 's2':
            num_band = 10
            satellite_list = ['s1']
        elif satellite_original == 's1_s2':
            num_band = [2, 10]
            satellite_list = ['s1', 's2']

        X_total_s = {}

        for satellite in satellite_list:
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
            
            grid_size_a = Xtemp.shape[1]
            grid_size_b = Xtemp.shape[2]

            X = np.zeros((grid_size_a*grid_size_b*len(gridded_fnames),num_band*num_timestamp))
            X[:] = np.nan

            for i in range(len(gridded_fnames)):
                
                X_one = np.load(os.path.join(gridded_dir,gridded_fnames[i]))[0:num_band,:,:]

                Xtemp = np.zeros((num_band, grid_size_a, grid_size_b, num_timestamp))
                Xtemp[:] = np.nan

                time_idx = np.array([np.int64(time.split('-')[1]) for time in time_json[i]])

                # Take median in each bucket
                for j in np.arange(12)+1:
                    Xtemp[:,:,:,j-1] = np.nanmedian(X_one[:,:,:,np.where(time_idx==j)][:,:,:,0,:],axis = 3)
                
                Xtemp = Xtemp.reshape(Xtemp.shape[0],-1,Xtemp.shape[3])
                if band_order == 'byband':
                    Xtemp = np.swapaxes(Xtemp, 0, 1).reshape(Xtemp.shape[1],-1)
                elif band_order == 'bytime':
                    Xtemp = np.swapaxes(Xtemp, 0, 1)
                
                X[(i*Xtemp.shape[0]):((i+1)*Xtemp.shape[0]), :] = Xtemp

            #y: # of pixels
            y_mask = get_y_label(home, country, data_set, data_type, satellite, ylabel_dir)
            y = y_mask.reshape(-1)   
            crop_id = crop_ind(y)
            
            X_noNA = fill_NA(X[crop_id,:][0,:,:])
            y = y[crop_id]

            X_total[satellite] = X_noNA

        if len(satellite_list)<2:
            X_total3types[data_type] = np.copy(X_total[satellite_original])
        elif:
            X_total3types[data_type] = np.hstack((X_total['s1'], X_total['s2']))

        y_total3types[data_type] = np.copy(y)

        output_fname = "_".join([data_set, 'raw', satellite_original, band_order, 'X'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
        np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', satellite_original, output_fname), X_total3types[data_type])
                
        output_fname = "_".join([data_set, 'raw', satellite_original, band_order, 'y'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
        np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', satellite_original, output_fname), y_total3types[data_type])

   
    return [X_total3types, y_total3types]

def PCA_scaler(home, country, data_set, satellite, ylabel_dir, band_order, scaler = StandardScaler()):
    """
    Save pixel arrays  # pixels * # features for pca
    
    Args:
      home - (str) the base directory of data

      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'

      data_set - (str) balanced 'small' or unbalanced 'full' dataset

      satellite - (str) satellite to use 's1' 's2' 's1_s2'

      ylabel_dir - (str) dir to load ylabel

      band_order - (str) band order: 'byband', 'bytime'

    Output: 

    saved in HOME/country/pixel_arrays

    """

    [X_total3types, _] = vectorize(home, country, data_set, satellite, ylabel_dir, band_order)

    X_train = X_total3types['train']
    X_val = X_total3types['val']
    X_test = X_total3types['test']
    pca = PCA()

    num_samples = 100000
    if X_train.shape[0] < num_samples:
        num_samples = X_train.shape[0]
    
    fit_scale = scaler.fit(X_train)
    pca.fit(X_train[:num_samples,:])

    X_train = fit_scale.transform(X_train)
    num_pcs = np.where(np.cumsum(pca.explained_variance_ratio_)>0.99)[0][0]
    
    pca_X_train = pca.transform(X_train)

    X_val = fit_scale.transform(X_val)
    pca_X_val = pca.transform(X_val)

    X_test = fit_scale.transform(X_test)
    pca_X_test = pca.transform(X_test)    
    
    output_fname = "_".join([data_set, 'pca', satellite, band_order, 'Xtrain', 'num_PCs'+str(num_pcs)+'.npy'])
    np.save(os.path.join(home,  country, 'pixel_arrays', data_set, 'pca', satellite, output_fname), pca_X_train[:,0:num_pcs])
    
    output_fname = "_".join([data_set, 'pca', satellite, band_order, 'Xval', 'num_PCs'+str(num_pcs)+'.npy'])
    np.save(os.path.join(home,  country, 'pixel_arrays', data_set, 'pca', satellite, output_fname), pca_X_val[:,0:num_pcs])
    
    output_fname = "_".join([data_set, 'pca', satellite, band_order, 'Xtest', 'num_PCs'+str(num_pcs)+'.npy'])
    np.save(os.path.join(home,  country, 'pixel_arrays', data_set, 'pca', satellite, output_fname), pca_X_test[:,0:num_pcs])
    
    return(pca_X_train, pca_X_val, pca_X_test)

