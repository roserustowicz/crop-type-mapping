"""

File to house the loss functions we plan to use.

"""

import numpy as np
import torch.optim as optim
import torch.nn as nn

from constants import *

def get_loss_fn(model_name):
    return mask_ce_loss

def mask_ce_loss(y_true, y_pred):
    """
    Args:
        y_true - (npy arr) 

    """
    total_loss = nn.CrossEntropyLoss(y_true, y_pred, reduction="sum")
    num_examples = torch.sum(torch.clamp(torch.sum(y_true, dim=0), min=0, max=1))
    return total_loss / num_examples


def get_optimizer(params, optimizer_name, lr, momentum, lrdecay):
    if optimizer_name == "sgd":
        return optimizers.SGD(params, lr=lr, momentum=momentum, decay=lrdecay)
    elif optimizer_name == "adam":
        return optimizer.Adam(params, lr=lr, decay=lrdecay)

    raise ValueError(f"Optimizer: {optimizer_name} unsupported")
