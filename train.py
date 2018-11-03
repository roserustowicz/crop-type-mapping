"""

Script for training and evaluating a model

"""
import os
import argparse
import h5py
import loss_fns
import models
import datetime
import torch

import datasets

from constants import *
from util import *
#from datasets import *
from tensorboardX import SummaryWriter

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

def train(model, model_name, args=None, dataloaders=None, X=None, y=None):
    """ Trains the model on the inputs
    
    Args:
        model - trainable model
        model_name - (str) name of the model
        args - (argparse object) args parsed in from main; used only for DL models
        dataloaders - (dict of dataloaders) used only for DL models
        X - (npy arr) data for non-dl models
        y - (npy arr) labels for non-dl models
    """
    if model_name in NON_DL_MODELS:
        if X is None: raise ValueError("X not provided!")
        if  y is None: raise ValueError("y nor provided!")
        model.fit(X, y)

    elif model_name in DL_MODELS:
        if dataloaders is None: raise ValueError("DATA GENERATOR IS NONE")
        if args is None: raise ValueError("Args is NONE")

        loss_fn = loss_fns.get_loss_fn(args.model_name)
        optimizer = loss_fns.get_optimizer(model.parameters(), args.optimizer, args.lr, args.momentum, args.lrdecay)
        writer = SummaryWriter()

            
        for split in ['train', 'val']:
            dl = dataloaders[split]
            batch_num = 0
            # TODO: Currently hardcoded to use padded inputs for an RNN model
            #       consider generalizing somehow so the training script can be
            #       more generic
            #for padded_inputs, lengths, targets in dl:
            for inputs, targets in dl:

                with torch.set_grad_enabled(True):
                    inputs.to(args.device)
                    targets.to(args.device)
                    preds = model.forward(inputs.double())
                    loss = loss_fn(targets.double(), preds.double())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                writer.add_scalar('/{split}/loss', loss, batch_num)
                batch_num += 1
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
    parser.add_argument('--num_workers', type=int,
                        help="Number of workers to use for pulling data",
                        default=8)
    # TODO: find correct string name
    parser.add_argument('--device', type=str,
                        help="Cuda or CPU",
                        default='cuda')
    parser.add_argument('--save_dir', type=str,
                        help="Directory to save the models in. If unspecified, saves the model to ./models.",
                        default='./models')
    parser.add_argument('--name', type=str,
                        help="Name of experiment. Used to uniquely save the model. Defaults to current time + model name if not set.")
    # Args for CLSTM model
    parser.add_argument('--hidden_dims', type=int, 
                        help="Number of channels in hidden state used in convolutional RNN",
                        default=4)
    parser.add_argument('--crnn_kernel_sizes', type=int,
                        help="Convolutional kernel size used within a recurrent cell",
                        default=3)
    parser.add_argument('--conv_kernel_size', type=int,
                        help="Convolutional kernel size used within a convolutional layer",
                        default=3)
    parser.add_argument('--crnn_num_layers', type=int,
                        help="Number of convolutional RNN cells to stack",
                        default=1)
    
    parser.add_argument('--time_slice', type=int,
                        help="which time slice for training FCN",
                        default=None)    

    args = parser.parse_args()
    # load in data generator
    dataloaders = {}
    for split in SPLITS:
        grid_path = os.path.join(args.grid_dir, f"{args.country}_{args.dataset}_{split}")
        dataloaders[split] = datasets.GridDataLoader(args, grid_path)
    
    # load in model
    model = models.get_model(**vars(args))
    if args.model_name in DL_MODELS and args.device == 'cuda' and torch.cuda.is_available():
        model.to(args.device)
    
    # train model
    train(model, args.model_name, args, dataloaders=dataloaders)
    # evaluate model

    # save model
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.model_name in DL_MODELS:
        if args.name is None:
            args.name = str(datetime.datetime.now()) + "_" + args.model_name
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.name))
        print("MODEL SAVED")
     
    
    