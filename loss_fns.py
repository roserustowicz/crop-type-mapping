"""

File to house the loss functions we plan to use.

"""

import numpy as np
import tensorflow as tf
import keras
from keras import losses, optimizers

from constants import *

def get_loss_fn(model_name):
    return mask_ce_loss

def mask_ce_loss(y_true, y_pred):
    """

    Args:
        y_true - (npy arr) 

    """
    mask = tf.clip_by_value(tf.reduce_sum(y_true, axis=0), 0, 1)
    pixel_wise_loss = tf.reduce_sum(-1 * y_pred * y_true + tf.log(1 +  tf.exp(y_pred)), axis=[0, 1])
    return tf.reduce_sum(mask * pixel_wise_loss)

def get_optimizer(optimizer_name, lr, momentum, lrdecay):
    if optimizer_name == "sgd":
        return optimizers.SGD(lr=lr, momentum=momentum, decay=lrdecay)
    elif optimizer_name == "adam":
        return optimizer.Adam(lr=lr, decay=lrdecay)

    raise ValueError(f"Optimizer: {optimizer_name} unsupported")
