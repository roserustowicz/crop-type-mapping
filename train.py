"""

Script for training and evaluating a model

"""
import os
import argparse
import h5py
from datasets import *
import loss_fns
import models
from constants import *
from util import *

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

def train(model, model_name, args=None, datagens=None, X=None, y=None):
    """ Trains the model on the inputs"""
    if model_name in NON_DL_MODELS:
        if X is None: raise ValueError("X not provided!")
        if  y is None: raise ValueError("y nor provided!")
        model.fit(X, y)
    elif model_name in DL_MODELS:
        if datagens is None: raise ValueError("DATA GENERATOR IS NONE")
        history = model.fit_generator(generator=datagens['train'], epochs=args.epochs, validation_data=datagens['val'], workers=8, use_multiprocessing=True, shuffle=args.shuffle)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help="model's name",
                        required=True)
    parser.add_argument('--hdf5_filepath', type=str,
                        help="full path to hdf5 data file",
                        default="/home/data/ghana/data.hdf5")
    parser.add_argument('--dataset', type=str,
                        help="Full or small?",
                        choices=('full', 'small'),
                        required=True)
    parser.add_argument('--country', type=str,
                        help="country to predict over",
                        default="ghana")
    parser.add_argument('--grid_dir', type=str,
                        help="full path to directory containing grid splits",
                        default="/home/data/ghana")
    parser.add_argument('--epochs', type=int,
                        help="# of times to train over the dataset")
    parser.add_argument('--batch_size', type=int,
                        help="batch size to use")
    parser.add_argument('--optimizer', type=str,
                        help="Optimizer to use for training",
                        default="sgd",
                        choices=('sgd', 'adam'))
    parser.add_argument('--lr', type=float,
                        help="Initial learning rate to use")
    parser.add_argument('--momentum', type=float,
                        help="Momentum to use when training",
                        default=.9)
    parser.add_argument('--lrdecay', type=float,
                        help="Learning rate decay per **batch**",
                        default=1)
    parser.add_argument('--shuffle', type=str2bool,
                        help="shuffle dataset between epochs?",
                        default=True)
    parser.add_argument('--use_s1', type=str2bool,
                        help="use s1 data?",
                        default=True)
    parser.add_argument('--use_s2', type=str2bool,
                        help="use s2 data?",
                        default=True)
    parser.add_argument('--num_classes', type=int,
                        help="Number of crops to predict over",
                        default=5)
    args = parser.parse_args()
    # load in data generator
    datagens = {}
    for split in SPLITS:
        grid_path = os.path.join(args.grid_dir, f"{args.country}_{args.dataset}_{split}")
        datagens[split] = CropTypeSequence(args.model_name, args.hdf5_filepath, grid_path, args.batch_size, args.use_s1, args.use_s2, args.num_classes)

    # load in model
    model = models.get_model(**vars(args))
    if args.model_name in DL_MODELS:
        # load in loss function / optimizer
        loss_fn = loss_fns.get_loss_fn(args.model_name)
        optimizer = loss_fns.get_optimizer(args.optimizer, args.lr, args.momentum, args.lrdecay)
        model.compile(optimizer=optimizer, loss=loss_fn)

    # train model
    train(model, args.model_name, args, datagens=datagens)
    # evaluate model

    # save model
 
