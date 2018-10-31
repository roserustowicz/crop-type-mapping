"""

File that houses all functions used to format, preprocess, or manipulate the data.

Consider this essentially a util library specifically for data manipulation.

"""

def onehot_mask(mask, num_classes):
    """
    Return a one-hot version of the mask for a grid

    Args: 
      mask - (np array) mask for grid that contains crop labels according 
             to '/home/data/crop_dict.npy'
      num_classes - (int) number of classes to be encoded into the one-hot
                    mask, as all classes in crop_dict are in the original 
                    mask. This must include other as one of the classes. For 
                    example, if the 5 main crop types + other are used, 
                    then num_classes = 6.

    Returns: 
      Returns a mask of size [64 x 64 x num_classes]. If a pixel was unlabeled, 
      it has 0's in all channels of the one hot mask at that pixel location.
    """

    mask[mask >= num_classes] = num_classes
    return np.eye(num_classes+1)[mask][:, :, 1:] 
    
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

    if model_name == "bidir_clstm":
        return preprocessForCLSTM(grid)

def preprocessForCLSTM(grid):
    return grid

def sample_timeseries(img_stack, num_samples, cloud_stack=None, remap_clouds=True, reverse=False, seed=None, save=False):
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

    
    
def vectorize(home, country, data_set, satellite, ylabel_dir, band_order= 'bytime', random_sample = True, num_timestamp = 25, reverse = False, seed = 0):
    """
    Save pixel arrays  # pixels * # features for raw
    
    Args:
      home - (str) the base directory of data

      country - (str) string for the country 'ghana', 'tanzania', 'southsudan'

      data_set - (str) balanced 'small' or unbalanced 'full' dataset

      satellite - (str) satellite to use 's1' 's2' 's1_s2'

      ylabel_dir - (str) dir to load ylabel

      band_order - (str) band order: 'byband', 'bytime'

    Output: 

    saved in HOME/pixel_arrays

    """

    satellite_original = str(np.copy(satellite))

    X_total3types = {}
    y_total3types = {}
    
    bad_list = np.load(os.path.join(home, country, 'bad_timestamp_grids_list.npy')) # just for num_stamp 25
    
    ## Go through 'train' 'val' 'test'
    for data_type in ['train','val','test']:

        if satellite_original == 's1':
            num_band = 2
            satellite_list = ['s1']
        elif satellite_original == 's2':
            num_band = 10
            satellite_list = ['s2']
        elif satellite_original == 's1_s2':
            num_band = [2, 10]
            satellite_list = ['s1', 's2']

        X_total = {}

        for satellite in satellite_list:
            #X: # of pixels * # of features
            gridded_dir = os.path.join(home, country, satellite+'_64x64_npy')
            gridded_IDs = sorted(np.load(os.path.join(home, country, country+'_'+data_set+'_'+data_type)))
            gridded_fnames = [satellite+'_'+country+'_'+gridded_ID+'.npy' for gridded_ID in gridded_IDs]
            good_grid = np.where([gridded_ID not in bad_list for gridded_ID in gridded_IDs])[0]
            
            # Time json
            time_fnames = [satellite+'_'+country+'_'+gridded_ID+'.json' for gridded_ID in gridded_IDs]
            time_json = [json.loads(open(os.path.join(gridded_dir,f),'r').read())['dates'] for f in time_fnames]
            

            # keep num of timestamps >=25
            gridded_IDs = [gridded_IDs[idx] for idx in good_grid]
            gridded_fnames = [gridded_fnames[idx] for idx in good_grid]
            time_json = [time_json[idx] for idx in good_grid]
            time_fnames = [time_fnames[idx] for idx in good_grid]
            
            
            if random_sample == True and satellite == 's2':
                # cloud mask
                cloud_mask_fnames = [satellite+'_'+country+'_'+gridded_ID+'_mask.npy' for gridded_ID in gridded_IDs]
                num_band = num_band + 1

            Xtemp = np.load(os.path.join(gridded_dir,gridded_fnames[0]))

            grid_size_a = Xtemp.shape[1]
            grid_size_b = Xtemp.shape[2]

            X = np.zeros((grid_size_a*grid_size_b*len(gridded_fnames),num_band*num_timestamp))
            X[:] = np.nan

            for i in range(len(gridded_fnames)):

                X_one = np.load(os.path.join(gridded_dir,gridded_fnames[i]))[0:num_band,:,:]
                Xtemp = np.zeros((num_band, grid_size_a, grid_size_b, num_timestamp))
                Xtemp[:] = np.nan

                if random_sample == True and satellite == 's2':
                    cloud_stack = np.load(os.path.join(gridded_dir,cloud_mask_fnames[i]))
                    [sampled_img_stack, sampled_cloud_stack] = sample_timeseries(X_one, num_samples = num_timestamp, cloud_stack=cloud_stack, reverse = reverse, seed = seed)
                    Xtemp = np.copy(np.vstack((sampled_img_stack,np.expand_dims(sampled_cloud_stack, axis=0))))
                
                elif random_sample == True and satellite == 's1':
                    [sampled_img_stack, _] = sample_timeseries(X_one, num_samples = num_timestamp, cloud_stack=None, reverse = reverse, seed = seed)
                    Xtemp = np.copy(sampled_img_stack)
                    
                else:
                    time_idx = np.array([np.int64(time.split('-')[1]) for time in time_json[i]])

                    # Take median in each bucket
                    for j in np.arange(12)+1:
                        Xtemp[:,:,:,j-1] = np.nanmedian(X_one[:,:,:,np.where(time_idx==j)][:,:,:,0,:],axis = 3)

                Xtemp = Xtemp.reshape(Xtemp.shape[0],-1,Xtemp.shape[3])
                if band_order == 'byband':
                    Xtemp = np.swapaxes(Xtemp, 0, 1).reshape(Xtemp.shape[1],-1)
                elif band_order == 'bytime':
                    Xtemp = np.swapaxes(Xtemp, 0, 1)
                    Xtemp = np.swapaxes(Xtemp, 1, 2).reshape(Xtemp.shape[0],-1)

                X[(i*Xtemp.shape[0]):((i+1)*Xtemp.shape[0]), :] = Xtemp

            #y: # of pixels
            y_mask = get_y_label(home, country, data_set, data_type, ylabel_dir)
            y_mask = y_mask[good_grid,:,:]
            y = y_mask.reshape(-1)   
            crop_id = crop_ind(y)

            X_noNA = fill_NA(X[crop_id,:][0,:,:])
            y = y[crop_id]

            X_total[satellite] = X_noNA

        if len(satellite_list)<2:
            X_total3types[data_type] = np.copy(X_total[satellite_original])
        else:
            X_total3types[data_type] = np.hstack((X_total['s1'], X_total['s2']))

        y_total3types[data_type] = np.copy(y)

        
        
        if random_sample == True and satellite == 's2':
            output_fname = "_".join([data_set, 'raw', satellite_original, 'cloud_mask','reverse'+str(reverse), band_order, 'X'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', 'cloud_s2', 'reverse_'+str(reverse).lower(), output_fname), X_total3types[data_type])

            output_fname = "_".join([data_set, 'raw', satellite_original, 'cloud_mask','reverse'+str(reverse), band_order, 'y'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', 'cloud_s2', 'reverse_'+str(reverse).lower(), output_fname), y_total3types[data_type])
            
        elif random_sample == True and satellite == 's1':
            output_fname = "_".join([data_set, 'raw', satellite_original, 'sample', band_order, 'X'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', 'sample_s1', output_fname), X_total3types[data_type])

            output_fname = "_".join([data_set, 'raw', satellite_original, 'sample', band_order, 'y'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', 'sample_s1', output_fname), y_total3types[data_type])

        else: 
            output_fname = "_".join([data_set, 'raw', satellite_original, band_order, 'X'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', satellite_original, output_fname), X_total3types[data_type])

            output_fname = "_".join([data_set, 'raw', satellite_original, band_order, 'y'+data_type, 'g'+str(len(gridded_fnames))+'.npy'])
            np.save(os.path.join(home, country, 'pixel_arrays', data_set, 'raw', satellite_original, output_fname), y_total3types[data_type])
   
    return [X_total3types, y_total3types]