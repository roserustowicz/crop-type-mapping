"""

Script for training and evaluating a model

"""
import os
import loss_fns
import models
import datetime
import torch
import datasets
import metrics
import util
import numpy as np

from constants import *
import visualize

def evaluate_split(model, model_name, split_loader, device, loss_weight, weight_scale, gamma):
    total_correct = 0
    total_loss = 0
    total_pixels = 0
    loss_fn = loss_fns.get_loss_fn(model_name)
    for inputs, targets in split_loader:
        with torch.set_grad_enabled(False):
            inputs.to(device)
            targets.to(device)
            preds = model(inputs)   
            batch_loss, _, _, batch_correct, num_pixels = evaluate(preds, targets, loss_fn, reduction="sum", weight_scale=weight_scale, loss_weight=loss_weight, gamma=gamma)
            total_loss += batch_loss.item()
            total_correct += batch_correct
            total_pixels += num_pixels

    return total_loss / total_pixels, total_correct / total_pixels

def evaluate(preds, labels, loss_fn, reduction, loss_weight, weight_scale, gamma, f1_type):
    """ Evalautes loss and metrics for predictions vs labels.

    Args:
        preds - (tf tensor) model predictions
        labels - (npy array / tf tensor) ground truth labels
        loss_fn - (function) function that takes preds and labels and outputs some loss metric
        reduction - (str) "avg" or "sum", where "avg" calculates the average accuracy for each batch
                                          where "sum" tracks total correct and total pixels separately
        loss_weight - (bool) whether we use weighted loss function or not
        f1_type - (str) micro, macro, None -- see sklearn for more information

    Returns:
        loss - (float) the loss the model incurs
        cm - (nparray) confusion matrix given preds and labels
        f1 - (float) f1-score
        accuracy - (float) given "avg" reduction, returns accuracy 
        total_correct - (int) given "sum" reduction, gives total correct pixels
        num_pixels - (int) given "sum" reduction, gives total number of valid pixels
    """
    f1 = metrics.get_f1score(preds, labels, f1_type)
    cm = metrics.get_cm(preds, labels)

    if reduction == "avg":
        loss = loss_fn(labels, preds, reduction, loss_weight, weight_scale, gamma)
        accuracy = metrics.get_accuracy(preds, labels, reduction=reduction)
        return loss, cm, f1, accuracy
    elif reduction == "sum":
        loss, _ = loss_fn(labels, preds, reduction, loss_weight, weight_scale, gamma)
        total_correct, num_pixels = metrics.get_accuracy(preds, labels, reduction=reduction)
        return loss, cm, f1, total_correct, num_pixels

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

        # set up information lists for visdom    
        vis_data = {'train_loss': [], 'val_loss': [], 
                    'train_acc': [], 'val_acc': [], 
                    'train_f1': [], 'val_f1': [], 
                    'train_gradnorm': []}
        
        vis = visualize.setup_visdom(args.env_name, model_name)

        loss_fn = loss_fns.get_loss_fn(model_name)
        optimizer = loss_fns.get_optimizer(model.parameters(), args.optimizer, args.lr, args.momentum, args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.patience)
        best_val_acc = 0

        for i in range(args.epochs):
            
            all_metrics = {'train_loss': 0, 'train_acc': 0, 'train_pix': 0, 'train_f1': [], 
                       'train_cm': np.zeros((args.num_classes, args.num_classes)).astype(int),
                       'val_loss': 0, 'val_acc': 0, 'val_pix': 0, 'val_f1': [],
                       'val_cm': np.zeros((args.num_classes, args.num_classes)).astype(int)}

            for split in ['train', 'val']:
                dl = dataloaders[split]
                batch_num = 0
                # TODO: Currently hardcoded to use padded inputs for an RNN model
                #       consider generalizing somehow so the training script can be
                #       more generic
                for inputs, targets, cloudmasks in dl:
                    with torch.set_grad_enabled(True):
                        inputs.to(args.device)
                        targets.to(args.device)
                        preds = model(inputs)   
                        
                        if split == 'train':
                            loss, cm_cur, f1, total_correct, num_pixels = evaluate(preds, targets, loss_fn, reduction="sum", loss_weight = args.loss_weight, weight_scale=args.weight_scale, gamma=args.gamma, f1_type=args.f1_type)
                            if cm_cur is not None:        
                                # If there are valid pixels, update weights
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                                gradnorm = torch.norm(list(model.parameters())[0].grad)
                                vis_data['train_gradnorm'].append(gradnorm)
                        
                        elif split == 'val':
                            loss, cm_cur, f1, total_correct, num_pixels = evaluate(preds, targets, loss_fn, reduction="sum", loss_weight = args.loss_weight, weight_scale=args.weight_scale, gamma=args.gamma, f1_type=args.f1_type)
                        
                        if cm_cur is not None:
                            # If there are valid pixels, update metrics
                            all_metrics[f'{split}_cm'] += cm_cur
                            all_metrics[f'{split}_loss'] += loss.data
                            all_metrics[f'{split}_acc'] += total_correct
                            all_metrics[f'{split}_pix'] += num_pixels
                            all_metrics[f'{split}_f1'].append(f1)
        
                    visualize.record_batch(inputs, cloudmasks, targets, preds, args.num_classes, split, vis_data, vis, args.include_doy, args.use_s1, args.use_s2, model_name, args.time_slice)

                    batch_num += 1

                if split == 'val':
                    val_loss = all_metrics['val_loss'] / all_metrics['val_pix']
                    lr_scheduler.step(val_loss)
                    val_acc = all_metrics['val_acc'] / all_metrics['val_pix']
                    
                    if val_acc > best_val_acc:
                        torch.save(model.state_dict(), os.path.join(args.save_dir, args.name + "_best"))
                        best_val_acc = val_acc
                
                visualize.record_epoch(all_metrics, split, vis_data, vis, i)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

if __name__ == "__main__":
    # parse args
    parser = util.get_train_parser()

    args = parser.parse_args()

    # load in data generator
    dataloaders = datasets.get_dataloaders(args.grid_dir, args.country, args.dataset, args)
    
    # load in model
    model = models.get_model(**vars(args))
    if args.model_name in DL_MODELS and args.device == 'cuda' and torch.cuda.is_available():
        model.to(args.device)

    if args.name is None:
        args.name = str(datetime.datetime.now()) + "_" + args.model_name

    # train model
    train(model, args.model_name, args, dataloaders=dataloaders)
    
    # evaluate model

    # save model
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.model_name in DL_MODELS:
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.name))
        print("MODEL SAVED")
     
    
    
