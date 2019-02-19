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
import pickle 

from constants import *
from tqdm import tqdm
import visualize

def evaluate_split(model, model_name, split_loader, device, loss_weight, weight_scale, gamma, num_classes, country):
    total_loss = 0
    total_pixels = 0
    total_cm = np.zeros((num_classes, num_classes)).astype(int) 
    loss_fn = loss_fns.get_loss_fn(model_name)
    for inputs, targets, cloudmasks in split_loader:
        with torch.set_grad_enabled(False):
            inputs.to(device)
            targets.to(device)
            preds = model(inputs)   
            batch_loss, batch_cm, _, num_pixels, confidence = evaluate(model_name, preds, targets, country, loss_fn=loss_fn, reduction="sum", loss_weight=loss_weight, weight_scale=weight_scale, gamma=gamma)
            total_loss += batch_loss.item()
            total_pixels += num_pixels
            total_cm += batch_cm

    f1_avg = metrics.get_f1score(total_cm, avg=True)
    return total_loss / total_pixels, f1_avg 

def evaluate(model_name, preds, labels, country, loss_fn=None, reduction=None, loss_weight=None, weight_scale=None, gamma=None):
    """ Evalautes loss and metrics for predictions vs labels.

    Args:
        preds - (tensor) model predictions
        labels - (npy array / tensor) ground truth labels
        loss_fn - (function) function that takes preds and labels and outputs some loss metric
        reduction - (str) "avg" or "sum", where "avg" calculates the average accuracy for each batch
                                          where "sum" tracks total correct and total pixels separately
        loss_weight - (bool) whether we use weighted loss function or not

    Returns:
        loss - (float) the loss the model incurs
        cm - (nparray) confusion matrix given preds and labels
        accuracy - (float) given "avg" reduction, returns accuracy 
        total_correct - (int) given "sum" reduction, gives total correct pixels
        num_pixels - (int) given "sum" reduction, gives total number of valid pixels
    """
    cm = metrics.get_cm(preds, labels, country, model_name)
    
    if model_name in NON_DL_MODELS:
        accuracy = metrics.get_accuracy(model_name, preds, labels, reduction=reduction)
        return None, cm, accuracy, None
    elif model_name in DL_MODELS:
        if reduction == "avg":
            loss, confidence = loss_fn(labels, preds, reduction, country, loss_weight, weight_scale)
            accuracy = metrics.get_accuracy(model_name, labels, model_name, reduction=reduction)
            return loss, cm, accuracy, confidence
        elif reduction == "sum":
            loss, confidence, _ = loss_fn(labels, preds, reduction, country, loss_weight, weight_scale) 
            total_correct, num_pixels = metrics.get_accuracy(model_name, preds, labels, reduction=reduction)
            return loss, cm, total_correct, num_pixels, confidence

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
        if dataloaders is None: raise ValueError("DATA GENERATOR IS NONE")
        if args is None: raise ValueError("Args is NONE")
        
        results = {'train_acc': [], 'train_f1': [], 'val_acc': [], 'val_f1': [], 'test_acc': [], 'test_f1': []}
        for rep in range(args.num_repeat):
            for split in ['train', 'val'] if not args.eval_on_test else ['test']:
                dl = dataloaders[split]
                X, y = datasets.get_Xy(dl)            

                if split == 'train':
                    model.fit(X, y)
                    preds = model.predict(X)
                    _, cm, accuracy, _ = evaluate(model_name, preds, y, args.country, reduction='avg')
                    f1 = metrics.get_f1score(cm, avg=True) 
                
                    # save model
                    with open(os.path.join(args.save_dir, args.name + "_pkl"), "wb") as output_file:
                        pickle.dump(model, output_file)

                elif split in ['val', 'test']:
                    preds = model.predict(X)
                    _, cm, accuracy, _ = evaluate(model_name, preds, y, args.country, reduction='avg')
                    f1 = metrics.get_f1score(cm, avg=True) 

                print('{} accuracy: {}, {} f1-score: {}'.format(split, accuracy, split, f1))
                results[f'{split}_acc'].append(accuracy)
                results[f'{split}_f1'].append(f1)

        for split in ['train', 'val'] if not args.eval_on_test else ['test']: 
            print('\n------------------------\nOverall Results:\n')
            print('{} accuracy: {} +/- {}'.format(split, np.mean(results[f'{split}_acc']), np.std(results[f'{split}_acc'])))
            print('{} f1-score: {} +/- {}'.format(split, np.mean(results[f'{split}_f1']), np.std(results[f'{split}_f1'])))


    elif model_name in DL_MODELS:
        if dataloaders is None: raise ValueError("DATA GENERATOR IS NONE")
        if args is None: raise ValueError("Args is NONE")
        splits = ['train', 'val'] if not args.eval_on_test else ['test']

        # set up information lists for visdom    
        vis_data = {}
        for split in splits:
            vis_data[f'{split}_loss'] = []
            vis_data[f'{split}_acc'] = []
            vis_data[f'{split}_f1'] = []
            vis_data[f'{split}_classf1'] = None
        vis_data['train_gradnorm'] = []
        vis = visualize.setup_visdom(args.env_name, model_name)
        loss_fn = loss_fns.get_loss_fn(model_name)
        optimizer = loss_fns.get_optimizer(model.parameters(), args.optimizer, args.lr, args.momentum, args.weight_decay)
        
        if args.optimizer == 'sgd':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.patience)
        best_val_f1 = 0

        for i in range(args.epochs):
            print('Epoch: {}'.format(i))
            all_metrics = {}
            for split in splits:
                all_metrics[f'{split}_loss'] = 0
                all_metrics[f'{split}_correct'] = 0
                all_metrics[f'{split}_pix'] = 0
                all_metrics[f'{split}_cm'] = np.zeros((args.num_classes, args.num_classes)).astype(int)

            for split in ['train', 'val'] if not args.eval_on_test else ['test']:
                dl = dataloaders[split]
                batch_num = 0
                for inputs, targets, cloudmasks in tqdm(dl):
                    print('inputs: ', inputs.shape)
                    print('targets: ', targets.shape)
                    with torch.set_grad_enabled(True):
                        inputs.to(args.device)
                        targets.to(args.device)
                        preds = model(inputs)   
                        
                        if split == 'train':
                            loss, cm_cur, total_correct, num_pixels, confidence = evaluate(model_name, preds, targets, args.country, loss_fn=loss_fn, reduction="sum", loss_weight=args.loss_weight, weight_scale=args.weight_scale, gamma=args.gamma)
                            if cm_cur is not None:        
                                # If there are valid pixels, update weights
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                gradnorm = torch.norm(list(model.parameters())[0].grad).detach().cpu() / torch.prod(torch.tensor(list(model.parameters())[0].shape), dtype=torch.float32)
                                vis_data['train_gradnorm'].append(gradnorm)
    
                        
                        elif split in ['val', 'test']:
                            loss, cm_cur, total_correct, num_pixels, confidence = evaluate(model_name, preds, targets, args.country, loss_fn=loss_fn, reduction="sum", loss_weight=args.loss_weight, weight_scale=args.weight_scale, gamma=args.gamma)
                        
                        if cm_cur is not None:
                            # If there are valid pixels, update metrics
                            all_metrics[f'{split}_cm'] += cm_cur
                            all_metrics[f'{split}_loss'] += loss.data
                            all_metrics[f'{split}_correct'] += total_correct
                            all_metrics[f'{split}_pix'] += num_pixels
        
                    visualize.record_batch(inputs, cloudmasks, targets, preds, confidence, args.num_classes, split, vis_data, vis, args.include_doy, args.use_s1, args.use_s2, model_name, args.time_slice)

                    batch_num += 1

                visualize.record_epoch(all_metrics, split, vis_data, vis, i, args.country)
                
                if split == 'val':
                    val_loss = all_metrics['val_loss'] / all_metrics['val_pix']
                    if args.optimizer == 'sgd':
                        lr_scheduler.step(val_loss)
                    val_f1 = metrics.get_f1score(all_metrics['val_cm'], avg=True)                 
 
                    if val_f1 > best_val_f1:
                        torch.save(model.state_dict(), os.path.join(args.save_dir, args.name + "_best"))
                        best_val_f1 = val_f1
                        if args.save_best: 
                            # TODO: Ideally, this would save any batch except the last one so that the saved images
                            #  are not only the remainder from the last batch 
                            visualize.record_batch(inputs, cloudmasks, targets, preds, confidence, args.num_classes, 
                                                   split, vis_data, vis, args.include_doy, args.use_s1, 
                                                   args.use_s2, model_name, args.time_slice, save=True, 
                                                   save_dir=os.path.join(args.save_dir, args.name + "_best"))

                            visualize.record_epoch(all_metrics, split, vis_data, vis, i, args.country, save=True, 
                                                  save_dir=os.path.join(args.save_dir, args.name + "_best"))               
                            
                            visualize.record_epoch(all_metrics, 'train', vis_data, vis, i, args.country, save=True, 
                                                  save_dir=os.path.join(args.save_dir, args.name + "_best"))               
 

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
    
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

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
     
    
    
