"""

File to house the loss functions we plan to use.

"""

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch

from constants import *

def get_loss_fn(model_name):
    return mask_ce_loss

def mask_ce_loss(y_true, y_pred):
    """
    Args:
        y_true - (npy arr) 

    nn.CrossEntropyLoss expects inputs: y_pred [N x classes] and y_true [N x 1]
    As input, y_pred and y_true have shapes [batch x classes x rows x cols] 

    To get them to the correct shape, we permute: 
      [batch x classes x rows x cols] --> [batch x rows x cols x classes]
      and then reshape to [N x classes], where N = batch*rows*cols

    Finally, to get y_true from [N x classes] to [N x 1], we take the argmax along
      the first dimension to get the largest class values from the one-hot encoding

    """
    batch, classes, rows, cols = y_true.shape
    
    # [batch x classes x rows x cols] --> [batch x rows x cols x classes]
    y_true = y_true.permute(0, 2, 3, 1)
    # [batch x rows x cols x classes] --> [batch*rows*cols x classes]
    y_true = y_true.contiguous().view(-1, y_true.shape[3])
    
    y_true_npy = y_true.cpu().numpy()
    num_examples = int(np.sum(y_true_npy))

    # [batch x classes x rows x cols] --> [batch x rows x cols x classes]
    y_pred = y_pred.permute(0, 2, 3, 1)
    # [batch x rows x cols x classes] --> [batch*rows*cols x classes]
    y_pred = y_pred.contiguous().view(-1, y_pred.shape[3])

    loss_mask = torch.sum(y_true, dim=1).type(torch.LongTensor)
    loss_mask_repeat = loss_mask.unsqueeze(1).repeat(1,y_pred.shape[1]).type(torch.FloatTensor).cuda()
   
    # take argmax to get true values from one-hot encoding 
    vals, y_true = torch.max(y_true, dim=1)
    
    y_true = y_true * loss_mask
    
    y_pred = y_pred * loss_mask_repeat

    loss_fn = nn.NLLLoss(reduction="sum")
    total_loss = loss_fn(y_pred, y_true.type(torch.LongTensor).cuda())
    return total_loss / (num_examples+1) #/ batch

# TODO: Incorporate lr decay
def get_optimizer(params, optimizer_name, lr, momentum, lrdecay):
    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer_name == "adam":
        return optim.Adam(params, lr=lr)

    raise ValueError(f"Optimizer: {optimizer_name} unsupported")
