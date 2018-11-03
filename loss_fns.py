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

    """
    y_true = y_true.permute(0, 2, 3, 1)
    y_true = y_true.contiguous().view(-1, y_true.shape[3])
    
    y_pred = y_pred.permute(0, 2, 3, 1)
    y_pred = y_pred.contiguous().view(-1, y_true.shape[-1])

    loss_mask = torch.sum(y_true, dim=1).type(torch.LongTensor)
    loss_mask_repeat = loss_mask.unsqueeze(1).repeat(1,y_pred.shape[1]).type(torch.DoubleTensor).cuda()
    
    vals, y_true = torch.max(y_true, dim=1)
    
    y_true = y_true * loss_mask
    y_pred = y_pred * loss_mask_repeat

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = loss_fn(y_pred, y_true.type(torch.LongTensor).cuda())
    
    num_examples = torch.sum(torch.clamp(torch.sum(y_true, dim=0), min=0, max=1)).type(torch.DoubleTensor).cuda()
    return total_loss / num_examples

# TODO: Incorporate lr decay
def get_optimizer(params, optimizer_name, lr, momentum, lrdecay):
    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer_name == "adam":
        return optim.Adam(params, lr=lr)

    raise ValueError(f"Optimizer: {optimizer_name} unsupported")
