"""

File that houses all functions used to format, preprocess, or manipulate the data.

Consider this essentially a util library specifically for data manipulation.

"""
import torch
import torch.nn.utils.rnn as rnn
import numpy as np
from constants import *
from util import *

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

    raise ValueError(f'Model: {model_name} unsupported')

def preprocess_label(label, model_name, num_classes=None):
    """ Returns a preprocess version of the label based on the model.

    Usually this just means converting to a one hot representation and 
    shifting the classes dimension of the mask to be the first dimension.

    Args:
        label - (npy arr) categorical labels for each pixel
        model_name - (str) name of the model
    Returns:
        (npy arr) [num_classes x 64 x 64]
    """
    if model_name == "bidir_clstm":
        assert not num_classes is None
        return preprocessLabelForCLSTM(label, num_classes)

    raise ValueError(f'Model: {model_name} unsupported')

def preprocessLabelForCLSTM(label, num_classes):
    """ Converts to onehot encoding and shifts channels to be first dim.

    Args:
        label - (npy arr) [64x64] categorical labels for each pixel
        num_classes - (npy arr) number of classes 
    """

    mask = onehot_mask(label, num_classes)
    return np.transpose(mask, [2, 0, 1])


def moveTimeToStart(arr):
    """ Moves time axis to the first dim.
    
    Args:
        arr - (npy arr) [bands x rows x cols x timestamps] """
    
    return np.transpose(arr, [3, 0, 1, 2])

def preprocessForCLSTM(grid):
    grid = moveTimeToStart(grid)
    return grid

def truncateToSmallestLength(batch):
    """ Truncates len of all sequences to MIN_TIMESTAMPS.

    Args:
        batch - (tuple of list of npy arrs) batch[0] is a list containing the torch versions of grids where each grid is [timestamps x bands x rows x cols]; batch[1] is a list containing the torch version of the labels

    """
    batch_X = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]

    for i in range(len(batch_X)):
        batch_X[i], _, _ = sample_timeseries(batch_X[i], MIN_TIMESTAMPS, timestamps_first=True)
        
    return [torch.stack(batch_X), torch.stack(batch_y)]

def padToVariableLength(batch):
    """ Pads all sequences to same length (variable per batch).

    Specifically, pads sequences to max length sequence with 0s.

    Args:
        batch - (tuple of list of npy arrs) batch[0] is a list containing the torch versions of grids where each grid is [timestamps x bands x rows x cols]; batch[1] is a list containing the torch version of the labels

    Returns:
        batch_X - (list of torch arrs) padded versions of each grid
    """
    batch_X = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]
    batch_X.sort(key=lambda x: x.shape[0], reverse=True)
    lengths = [x.shape[0] for x in batch_X]
    lengths = torch.tensor(lengths, dtype=torch.float32)
    batch_X = rnn.pad_sequence(batch_X, batch_first=True)
    return [batch_X, lengths, batch_y]


def concat_s1_s2(s1, s2):
    """ Returns a concatenation of s1 and s2 data.

    Specifically, returns s1 if s2 is None, s2 if s1 is None, and otherwise downsamples the larger series to size of the smaller one and returns the concatenation on the time axis.

    Args:
        s1 - (npy array) [bands x rows x cols x timestamps]
        s2 - (npy array) [bands x rows x cols x timestamps]

    Returns:
        (npy array) [bands x rows x cols x min(num s1 timestamps, num s2 timestamps) Concatenation of s1 and s2 data
    """
    if s1 is None:
        return s2
    if s2 is None:
        return s1
    if s1.shape[-1] > s2.shape[-1]:
        s1, _, _ = sample_timeseries(s1, s2.shape[-1])
    else:
        s2, _, _ = sample_timeseries(s2, s1.shape[-1])
    return np.concatenate((s1, s2), axis=0)


def sample_timeseries(img_stack, num_samples, dates=None, cloud_stack=None, remap_clouds=True, reverse=False, seed=None, verbose=False, timestamps_first=False):
    """
    Args:
      img_stack - (numpy array) [bands x rows x cols x timestamps], temporal stack of images
      num_samples - (int) number of samples to sample from the img_stack (and cloud_stack)
                     and must be <= the number of timestamps
      dates - (list) list of dates that correspond to the timestamps in the img_stack and
                     cloud_stack
      cloud_stack - (numpy array) [rows x cols x timestamps], temporal stack of cloud masks
      reverse - (boolean) take 1 - probabilities, encourages cloudy images to be sampled
      seed - (int) a random seed for sampling

    Returns:
      sampled_img_stack - (numpy array) [bands x rows x cols x num_samples], temporal stack
                          of sampled images
      sampled_dates - (list) [num_samples], list of dates associated with the samples
      sampled_cloud_stack - (numpy array) [rows x cols x num_samples], temporal stack of
                            sampled cloud masks, only returned if cloud_stack was an input

    To read in img_stack from npy file for input img_stack:

       img_stack = np.load('/home/data/ghana/s2_64x64_npy/s2_ghana_004622.npy')
    
    To read in cloud_stack from npy file for input cloud_stack:

       cloud_stack = np.load('/home/data/ghana/s2_64x64_npy/s2_ghana_004622_mask.npy')
    
    To read in dates from json file for input dates:
       
       with open('/home/data/ghana/s2_64x64_npy/s2_ghana_004622.json') as f:
           dates = json.load(f)['dates']

    """
    if timestamps_first:
        timestamps = img_stack.shape[0]
    else:
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
        if verbose:
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
    if timestamps_first:
        sampled_img_stack = img_stack[samples, :, :, :]
    else:
        sampled_img_stack = img_stack[:, :, :, samples]
    
    samples_list = list(samples)
    sampled_dates = None
    
    if not dates is None:
        sampled_dates = [dates[i] for i in samples_list]

    if isinstance(cloud_stack, np.ndarray):
        if remap_clouds:
            sampled_cloud_stack = remap_cloud_stack[:, :, samples]
        else:
            sampled_cloud_stack = cloud_stack[:, :, samples]
        return sampled_img_stack, sampled_dates, sampled_cloud_stack
    else:
        return sampled_img_stack, sampled_dates, None   
