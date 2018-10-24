"""

Script for training and evaluating a model

"""

import argparse


def evaluate(model, inputs, labels, loss_fn):
    """ Evalautes the model on the inputs using the labels and loss fn.

    Args:
        model - (something with predict?) the model being tested
        inputs - (npy array / tf tensor) the inputs the model should use
        labels - (npy array / tf tensor) the labels for the inputs
        loss_fn - (function) function that takes preds and labels and outputs some metric

    Returns:
        loss - (float) the loss the model incurs
        TO BE EXPANDED
    """

    return -1

def train(args,):
    """ Trains the model on the inputs"""

    return -1

def get_loss_fn(args):
    return -1

if __name__ == "__main__":
    # parse args

    # load in data generator

    # load in loss function / optimizer

    # load in model

    # train model

    # evaluate model

    # save model

