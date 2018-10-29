"""

File that houses all functions used to format, preprocess, or manipulate the data.

Consider this essentially a util library specifically for data manipulation.

"""

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
