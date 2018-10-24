"""

File for visualizing model performance.

"""

from preprocess import *

def visualize_model_preds(model, grid_name, save=False):
    """ Outputs a visualization of model predictions for one grid.

    Args:
        model - (ML model) model to be evaluated
        grid_name - (string) name of the grid to evaluate
    """


    # assuming there is some way to store the model's name in the model itself

    # assuming these functions exists somewhere in preprocess

    mask = retrieve_mask(grid_name) # get the mask given a grid's name (ex: "004232")
    grid = retrieve_grid(grid_name) # get the actual grid data given a grid's name
    grid = preprocess_grid(grid, model.name) # preprocess the grid in a model specific way

    preds = model.predict(grid) # get model predictions

    # formats preds into a 64x64 grid and creates a visualization of the predicted values
    # masking everything that's not labeled
    visualize_preds(preds, mask)

    # save if flag set

