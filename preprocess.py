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

    if model_name == "C-LSTM":
        return preprocessForCLSTM(grid)




