import numpy as np
import torch
import util

from constants import *
from sklearn.metrics import confusion_matrix, f1_score

def get_accuracy(y_pred, y_true, reduction='avg'):
    """
    Get accuracy from predictions and labels 

    Args: 
      y_pred -  
      y_true - 
      reduction - 

    Returns: 
      
    """
    batch, classes, rows, cols = y_true.shape

    # Reshape truth labels into [N, num_classes]
    y_true = util.bxclxrxc_to_brcxcl(y_true)

    # Get number of valid examples
    y_true_npy = y_true.cpu().numpy()
    num_pixels = int(np.sum(y_true_npy))

    # Reshape predictions into [N, num_classes]
    y_pred = util.bxclxrxc_to_brcxcl(y_pred)

    # Create mask for valid pixel locations
    loss_mask = torch.sum(y_true, dim=1).type(torch.LongTensor) 

    # Take argmax for labels and targets
    _, y_true = torch.max(y_true, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)

    # Multiply by mask
    y_true = y_true * loss_mask
    y_pred = y_pred.cpu() * loss_mask

    bool1 = [y_true.numpy() == y_pred.numpy()]
    bool2 = [loss_mask.numpy() == 1]
    
    total_correct = np.sum(np.logical_and(bool1, bool2))

    if reduction == 'avg':
        return total_correct / num_pixels
    else:
        return total_correct, num_pixels

def get_f1score(y_pred, y_true):
    # Reshape truth labels into [N, num_classes]
    y_true = util.bxclxrxc_to_brcxcl(y_true)

    # Reshape predictions into [N, num_classes]
    y_pred = util.bxclxrxc_to_brcxcl(y_pred)

    # Create mask for valid pixel locations
    loss_mask = torch.sum(y_true, dim=1).type(torch.LongTensor)

    # Take argmax for labels and targets
    _, y_true = torch.max(y_true, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)

    # Get only valid locations
    y_true = y_true[loss_mask == 1]
    y_pred = y_pred[loss_mask == 1]
    
    return f1_score(y_true, y_pred, labels=CM_LABELS, average='micro') 


def get_cm(y_pred, y_true):
    """
    Get confusion matrix from predictions and labels

    Args: 
      y_pred - 
      y_true -
    
    Returns: 
      cm - confusion matrix 
    """ 
    # Reshape truth labels into [N, num_classes]
    y_true = util.bxclxrxc_to_brcxcl(y_true)
    
    # Reshape predictions into [N, num_classes]
    y_pred = util.bxclxrxc_to_brcxcl(y_pred)
    
    # Create mask for valid pixel locations
    loss_mask = torch.sum(y_true, dim=1).type(torch.LongTensor) 
    
    # Take argmax for labels and targets
    _, y_true = torch.max(y_true, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)

    # Get only valid locations
    y_true = y_true[loss_mask == 1]
    y_pred = y_pred[loss_mask == 1]
    
    if y_true.shape[0] == 0:
        pass
    else: 
        return confusion_matrix(y_true, y_pred, labels=CM_LABELS)
