"""
Wrapper script for performing random search. 

Currently hard-coded to iterate over hp space, but with intelligeny key parsing should be possible to fully automate this process. 

run with:

python random_search.py --model_name bidir_clstm --dataset small --epochs 1 --batch_size_range="(4, 24)" --lr_range="(-5, -1)" --hidden_dims_range="(4, 7)" --weight_decay_range="(-5, 0)" --num_samples=3 --logfile=test.log

"""


import argparse
import os
import train 
import numpy as np
import util
import datasets
import os
import models
import torch
import sys
from ast import literal_eval

if __name__ ==  "__main__":
    # get all ranges of values

    search_parser = argparse.ArgumentParser()
    search_parser.add_argument('--model_name', type=str)
    search_parser.add_argument('--dataset', type=str)
    search_parser.add_argument('--lr_range', type=literal_eval,
                        help="Exp of the lr to sample from. If len > 2, samples from the tuple. Else, assumes entry 0 is the min exp and entry 1 is the max and randomly samples in the range with a uniform distribution.")
    search_parser.add_argument('--batch_size_range', type=literal_eval,
                        help="Batch sizes to consider. If > 2, samples from the tuple. Else, assumes entry 0 is the min and entry 1 is the mad and randomly samples an int from that range.")
    search_parser.add_argument('--hidden_dims_range', type=literal_eval,
                        help="Number of channels in hidden state of CLSTM unit to consider. If > 2, samples from the tuple. Else, assumes entry 0 is the min exp and entry 1 is the max exp. Base 2.") 
    search_parser.add_argument('--weight_decay_range', type=literal_eval,
                        help="Exp of the wd to sample from. Base 10.")
    search_parser.add_argument('--num_samples', type=int,
                        help="number of random searches to perform")
    search_parser.add_argument('--epochs', type=int,
                        help="number of epochs to train the model for")
    search_parser.add_argument('--numTrack', type=int,
                        help="number of models to track",
                        default=1)
    search_parser.add_argument('--logfile', type=str,
                        help="file to write logs to; if not specified, prints to terminal")

    search_range = search_parser.parse_args()
    #TODO: VERY HACKY, SWITCH TO USING PYTHON LOGGING MODULE OR ACTUALLY USING WRITE CALLS
    old_stdout = sys.stdout
    if search_range.logfile is not None:
        logfile = open(search_range.logfile, "w")
        sys.stdout = logfile

    lr_range = search_range.lr_range
    batch_size_range = search_range.batch_size_range
    hidden_dims_range = search_range.hidden_dims_range
    weight_decay_range = search_range.weight_decay_range

    experiments = {}

    # for some number of iterations
    for sample_no in range(search_range.num_samples):
        
        # generate new sets of hyper parameters in the ranges specified
        lr_exp = np.random.uniform(lr_range[0], lr_range[1])
        lr = 10 ** lr_exp

        batch_size = np.random.randint(batch_size_range[0], batch_size_range[1])
        hidden_dims_exp = np.random.randint(hidden_dims_range[0], hidden_dims_range[1])
        hidden_dims = 2 ** hidden_dims_exp
        weight_decay_exp = np.random.uniform(weight_decay_range[0], weight_decay_range[1])
        weight_decay = 10 ** weight_decay_exp
        # build argparse args by parsing args and then setting empty fields to specified ones above
        train_parser = util.get_train_parser()
        train_args = train_parser.parse_args(['--model_name', search_range.model_name, '--dataset', search_range.dataset])
        train_args.lr = lr
        train_args.batch_size = batch_size
        train_args.weight_decay = weight_decay
        train_args.hidden_dims = hidden_dims
        train_args.epochs = search_range.epochs

        dataloaders = datasets.get_dataloaders(train_args.grid_dir, train_args.country, train_args.dataset, train_args)
        
        model = models.get_model(**vars(train_args))
        model.to(train_args.device)
        experiment_name = f"lr{lr}_bs{batch_size}_wd{weight_decay}_hd{hidden_dims}_epochs{search_range.epochs}_model_{train_args.model_name}_dataset_{train_args.dataset}"

        train_args.name = experiment_name
        print("="*100)
        print(f"TRAINING: {experiment_name}")
        train.train(model, train_args.model_name, train_args, dataloaders=dataloaders) 
        print(f"FINISHED TRAINING") 
        for state_dict_name in os.listdir(train_args.save_dir):
            if experiment_name in state_dict_name:
                model.load_state_dict(torch.load(os.path.join(train_args.save_dir, state_dict_name)))
                loss, acc = train.evaluate_split(model, train_args.model_name, dataloaders['val'], train_args.device)
                print(f"Best Performance: \n\t loss: {loss} \n\t acc: {acc}\n")
                experiments[experiment_name] = [loss, acc]
                break

        torch.cuda.empty_cache()
   
    print("SUMMARY")
    for key, value in sorted(experiments.items(), key=lambda x: x[1][1], reverse=True):
        print(key, "\t", value[0], "\t", value[1])
        
    sys.stdout = old_stdout
    if search_range.logfile is not None:
        logfile.close()
