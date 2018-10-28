"""

Script for training and evaluating a model

"""
import os
import argparse
import h5py
from datasets import *
import models


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
    if model_name in ['random_forest', 'logistic_regression']:
        model.fit(X, y)
    else:
        assert datagens != None, "DATA GENERATOR IS NONE"
        history = model.fit_generator(generator=datagens['train'], epochs=args.epochs, validation_data=datagens['val'], workers=8, use_multiprocessing=True, shuffle=args.shuffle)

    return model

def get_loss_fn(args):
    return -1

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        help="model's name")
    parser.add_argument('--hdf5_filepath', type=str,
                        help="full path to hdf5 data file",
                        default="/home/data/ghana/data.hdf5")
    parser.add_argument('--dataset', type=str,
                        help="Full or small?",
                        choices=('full', 'small'))
    parser.add_argument('--country', type=str,
                        help="country to predict over",
                        default="ghana")
    parser.add_argument('--grid_dir', type=str,
                        help="full path to directory containing grid splits",
                        default="/home/data/ghana")
    parser.add_argument('--epochs', type=int,
                        help="# of times to train over the dataset",
                        required=True)
    parser.add_argument('--batch_size', type=int,
                        help="batch size to use",
                        required=True)
    parser.add_argument('--shuffle', type=bool,
                        help="shuffle dataset between epochs?",
                        default=True)
    parser.add_argument('--use_s1', type=bool,
                        help="use s1 data?",
                        default=True)
    parser.add_argument('--use_s2', type=bool,
                        help="use s2 data?",
                        default=True)
    args = parser.parse_args()

    # load in data generator
    datagens = {}
    for split in ['train', 'val', 'test']:
        grid_path = os.path.join(args.grid_dir, f"{args.country}_{args.dataset}_{split}")
        datagens[split] = CropTypeSequence(args.model_name, args.hdf5_filepath, grid_path, args.batch_size, args.use_s1, args.use_s2)

    # load in loss function / optimizer

    # TODO: make more general to allow for simpler baseline models

    # load in model
    model_gen_fn = models.__dict__[f'make_{args.model_name}_model']
    model = model_gen_fn(**vars(args))
    # train model
    train(model, model_name, args, datagens=datagens)
    # evaluate model

    # save model

