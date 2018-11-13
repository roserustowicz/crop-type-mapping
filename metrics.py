import numpy as np
import torch
import util

import preprocess
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
    # Reshape truth labels into [N, num_classes]
    y_true = preprocess.reshapeForLoss(y_true)

    # Reshape predictions into [N, num_classes]
    y_pred = preprocess.reshapeForLoss(y_pred)

    # Get rid of invalid pixels and take argmax
    y_pred, y_true = preprocess.maskForMetric(y_pred, y_true)

    # Get metrics for accuracy
    total_correct = np.sum([y_true.numpy() == y_pred.cpu().numpy()])
    num_pixels = y_pred.shape[0]
    print('acc num pixels: ') 
    if reduction == 'avg':
        if num_pixels == 0:
            return None
        else:
            return total_correct / num_pixels
    else:
        return total_correct, num_pixels


def get_f1score(y_pred, y_true):
    # Reshape truth labels into [N, num_classes]
    y_true = preprocess.reshapeForLoss(y_true)

    # Reshape predictions into [N, num_classes]
    y_pred = preprocess.reshapeForLoss(y_pred)

    # Get rid of invalid pixels and take argmax
    y_pred, y_true = preprocess.maskForMetric(y_pred, y_true)

    if y_true.shape[0] == 0:
        return None
    else: 
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
    y_true = preprocess.reshapeForLoss(y_true)
    
    # Reshape predictions into [N, num_classes]
    y_pred = preprocess.reshapeForLoss(y_pred)
   
    y_pred, y_true = preprocess.maskForMetric(y_pred, y_true)
 
    if y_true.shape[0] == 0:
        return None
    else: 
        return confusion_matrix(y_true, y_pred, labels=CM_LABELS)
