"""
Wrapper script for performing random search. 

run with:

python random_search.py --model_name bidir_clstm --dataset small --epochs 1 --batch_size_range="(4, 24)" --lr_range="(-5, -1)" --hidden_dims_range="(4, 7)" --weight_decay_range="(-5, 0)" --num_samples=3 --logfile=test.log

python random_search.py --model_name bidir_clstm --dataset small --epochs 1 --batch_size_range="(4, 24)" --lr_range="(10, -5, -1)" --hidden_dims_range="(2, 3, 7)" --weight_decay_range="(10, -5, 0)" --momentum_range="(.5, .999)" --optimizer_range="('adam', 'sgd')" --num_samples=3 --logfile=mytest.log

"""


import argparse
import os
import train 
import pickle
import numpy as np
import util
import datasets
import os
import models
import torch
import sys

from constants import *
from ast import literal_eval

def generate_int_power_HP(base, minVal, maxVal):
    exp = np.random.randint(minVal, maxVal)
    return base ** exp

def generate_real_power_HP(base, minVal, maxVal):
    exp = np.random.uniform(minVal, maxVal)
    return base ** exp

def generate_int_HP(minVal, maxVal):
    return np.random.randint(minVal, maxVal)

def generate_float_HP(minVal, maxVal):
    return np.random.uniform(minVal, maxVal)

def generate_string_HP(choices):
    return np.random.choice(choices)

def str2tuple(arg):
    return literal_eval(arg)

if __name__ ==  "__main__":
    # get all ranges of values

    search_parser = argparse.ArgumentParser()
    search_parser.add_argument('--model_name', type=str)
    search_parser.add_argument('--dataset', type=str)
    search_parser.add_argument('--lr_range', type=str2tuple,
                        help="Tuple containing (base, min exp, max exp). Picks learning rates from base ** min exp to base ** max exp on a logarithmic scale.")
    search_parser.add_argument('--batch_size_range', type=str2tuple,
                        help="Tuple containing (min bs, max bs). Picks bs uniformly between (min bs, max bs).") 
    search_parser.add_argument('--hidden_dims_range', type=str2tuple,
                        help="Tuple containing (base, min exp, max exp). Picks number of channels from base ** min exp to base ** max exp.")
    search_parser.add_argument('--weight_decay_range', type=str2tuple,
                        help="Tuple containing (base, min exp, max exp). Picks weight decay between base ** min exp to base ** max exp on a logarithmic scale.")
    search_parser.add_argument('--momentum_range', type=str2tuple)
    search_parser.add_argument('--optimizer_range', type=str2tuple)
    search_parser.add_argument('--num_samples', type=int,
                        help="number of random searches to perform")
    search_parser.add_argument('--epochs', type=int,
                        help="number of epochs to train the model for")
    search_parser.add_argument('--logfile', type=str,
                        help="file to write logs to; if not specified, prints to terminal")

    search_range = search_parser.parse_args()
    #TODO: VERY HACKY, SWITCH TO USING PYTHON LOGGING MODULE OR ACTUALLY USING WRITE CALLS
    old_stdout = sys.stdout
    if search_range.logfile is not None:
        logfile = open(search_range.logfile, "w")
        sys.stdout = logfile

    hps = {}
    for arg in vars(search_range):
        if "range" not in arg: continue
        hp = arg[:arg.find("range") - 1]
        hps[hp] = [] 

    experiments = {}

    # for some number of iterations
    for sample_no in range(search_range.num_samples):

        # build argparse args by parsing args and then setting empty fields to specified ones above
        train_parser = util.get_train_parser()
        train_args = train_parser.parse_args(['--model_name', search_range.model_name, '--dataset', search_range.dataset])

        # build argparse args by parsing args and then setting empty fields to specified ones above
        for arg in vars(search_range):
            if "range" not in arg: continue
            hp = arg[:arg.find("range") - 1]
            if hp in INT_POWER_EXP:
                hp_val = generate_int_power_HP(vars(search_range)[arg][0], vars(search_range)[arg][1], vars(search_range)[arg][2])
            elif hp in REAL_POWER_EXP:
                hp_val = generate_real_power_HP(vars(search_range)[arg][0], vars(search_range)[arg][1], vars(search_range)[arg][2])
            elif hp in INT_HP:
                hp_val = generate_int_HP(vars(search_range)[arg][0], vars(search_range)[arg][1])
            elif hp in FLOAT_HP:
                hp_val = generate_float_HP(vars(search_range)[arg][0], vars(search_range)[arg][1])
            elif hp in STRING_HP:
                hp_val = generate_string_HP(vars(search_range)[arg])
            else:
                raise ValueError(f"HP {hp} unsupported") 

            train_args.__dict__[hp] = hp_val

        train_args.epochs = search_range.epochs

        dataloaders = datasets.get_dataloaders(train_args.grid_dir, train_args.country, train_args.dataset, train_args)
        
        model = models.get_model(**vars(train_args))
        model.to(train_args.device)
        experiment_name = f"model:{train_args.model_name}_dataset:{train_args.dataset}_epochs:{search_range.epochs}"
        for hp in hps:
            experiment_name += f"_{hp}:{train_args.__dict__[hp]}"
        train_args.name = experiment_name
        print("="*100)
        print(f"TRAINING: {experiment_name}")
        train.train(model, train_args.model_name, train_args, dataloaders=dataloaders) 
        print(f"FINISHED TRAINING") 
        for state_dict_name in os.listdir(train_args.save_dir):
            if (experiment_name + "_best") in state_dict_name:
                model.load_state_dict(torch.load(os.path.join(train_args.save_dir, state_dict_name)))
                loss, acc = train.evaluate_split(model, train_args.model_name, dataloaders['val'], train_args.device)
                print(f"Best Performance: \n\t loss: {loss} \n\t acc: {acc}\n")
                experiments[experiment_name] = [loss, acc]
                for hp in hps:
                    hps[hp].append([train_args.__dict__[hp], loss, acc])
                break

        torch.cuda.empty_cache()
   
    print("SUMMARY")
    for key, value in sorted(experiments.items(), key=lambda x: x[1][1], reverse=True):
        print(key, "\t", value[0], "\t", value[1])
    
    with open("hps_results.pkl", "wb") as f:
        pickle.dump(hps, f)

    sys.stdout = old_stdout
    if search_range.logfile is not None:
        logfile.close()
