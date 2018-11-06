"""

File for visualizing model performance.

"""

import numpy as np
import preprocess

def visualize_rgb(argmax_array, num_classes, class_colors=None): 
    mask = []
    rgb_output = np.zeros((argmax_array.shape[0], 3, argmax_array.shape[2], argmax_array.shape[3]))

    if class_colors == None:
        rgbs = [ [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255] ]
    
    assert len(rgbs) == num_classes

    for cur_class in range(0, num_classes):
        tmp = np.asarray([argmax_array == cur_class+1])[0]

        mask_cat = np.concatenate((tmp, tmp, tmp), axis=1)

        class_vals = np.concatenate((np.ones_like(tmp)*rgbs[cur_class][0],
                                     np.ones_like(tmp)*rgbs[cur_class][1],
                                     np.ones_like(tmp)*rgbs[cur_class][2]), axis=1) 

        rgb_output += (mask_cat * class_vals)
        
    return rgb_output


def visualize_model_preds(model, grid_name, save=False):
    """ Outputs a visualization of model predictions for one grid.

    Args:
        model - (ML model) model to be evaluated
        grid_name - (string) name of the grid to evaluate
    """
    # assuming there is some way to store the model's name in the model itself
    # assuming these functions exists somewhere in preprocess
    
    # TODO: This function as a whole is a WIP -- was abandoned to
    #  get visdom working instead ... 

    label = preprocess.retrieve_label(grid_name, country) # get the mask given a grid's name (ex: "004232")
    best_grid = preprocess.retrieve_best_s2_grid(grid_name, country) # get the actual grid data given a grid's name
    
    grid = preprocess.preprocess_grid(grid, model.name) # preprocess the grid in a model specific way

    preds = model.predict(grid) # get model predictions

    # formats preds into a 64x64 grid and creates a visualization of the predicted values
    # masking everything that's not labeled
    visualize_preds(preds, mask)

    # save if flag set

