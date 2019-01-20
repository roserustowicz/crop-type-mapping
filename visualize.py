"""
import random

File for visualizing model performance.

"""

import numpy as np
import os 
import matplotlib.pyplot as plt
import visdom
from torchvision.utils import save_image, make_grid

import metrics
import preprocess
import util
from constants import * 

def setup_visdom(env_name, model_name):
    # TODO: Add args to visdom envs default name
    env_name = model_name if not env_name else env_name
    return visdom.Visdom(port=8097, env=env_name)

def visdom_save_metric(metric_name, split, title, x_label, y_label, vis_data, save_dir):
    """
    Args: 
      metric_name - "loss", "acc", "f1"
    """
    Y=np.array(vis_data['{}_{}'.format(split, metric_name)])
    X=np.array(range(len(vis_data['{}_{}'.format(split, metric_name)])))

    plt.figure()
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(['{}_{}'.format(split, metric_name)])
    plt.savefig(os.path.join(save_dir, title + '.png'))
    plt.close()

def visdom_save_many_metrics(metric_name, split, title, x_label, y_label, legend_lbls, vis_data, save_dir):
    """
    Args: 
      metric_name - "loss", "acc", "f1"
    """
 
    Y = vis_data['{}_{}'.format(split, metric_name)]
    X = np.array([range(len(vis_data['{}_{}'.format(split, metric_name)]))] * Y.shape[1]).T 

    plt.figure()
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend_lbls)
    plt.savefig(os.path.join(save_dir, title + '.png'))
    plt.close()

def visdom_plot_metric(metric_name, split, title, x_label, y_label, vis_data, vis):
    """
    Args: 
      metric_name - "loss", "acc", "f1"
    """
    vis.line(Y=np.array(vis_data['{}_{}'.format(split, metric_name)]),
             X=np.array(range(len(vis_data['{}_{}'.format(split, metric_name)]))),
             win=title,
             opts={'legend': ['{}_{}'.format(split, metric_name)],
                   'markers': False, 
                   'title': title,
                   'xlabel': x_label,
                   'ylabel': y_label})

def visdom_plot_many_metrics(metric_name, split, title, x_label, y_label, legend_lbls, vis_data, vis):
    """
    Args: 
      metric_name - "loss", "acc", "f1"
    """
 
    Y = vis_data['{}_{}'.format(split, metric_name)]
    X = np.array([range(len(vis_data['{}_{}'.format(split, metric_name)]))] * Y.shape[1]).T 
    vis.line(Y=Y,
             X=X,
             win=title,
             opts={'legend': legend_lbls,
                   'markers': False, 
                   'title': title,
                   'xlabel': x_label,
                   'ylabel': y_label})
    
def visdom_plot_images(vis, imgs, win):
    """
    Plot image panel in visdom
    Args: 
      imgs - (array) array of images [batch x channels x rows x cols]
      win - (str) serves as both window name and title name
    """
    vis.images(imgs, nrow=NROW, win=win, padding=8, opts={'title': win})

def record_batch(inputs, clouds, targets, preds, confidence, num_classes, split, vis_data, vis, include_doy, use_s1, use_s2, model_name, time_slice, save=False, save_dir=None, show_visdom=True, show_matplot=False):
    """ Record values and images for batch in visdom
    """
    # Create and show mask for labeled areas
    label_mask = np.sum(targets.numpy(), axis=1)
    label_mask = np.expand_dims(label_mask, axis=1)
    if show_visdom:
        visdom_plot_images(vis, label_mask, 'Label Masks')
        #visdom_plot_images(vis, confidence, 'Confidence')

    # Show best inputs judging from cloud masks
    if torch.sum(clouds) != 0 and len(clouds.shape) > 1: 
        best = np.argmax(np.mean(np.mean(clouds.numpy()[:, 0, :, :, :], axis=1), axis=1), axis=1)
    else:
        best = np.random.randint(0, high=MIN_TIMESTAMPS, size=(inputs.shape[0],))
    best = np.zeros_like(best)

    # Get bands of interest (boi) to show best rgb version of s2 or vv, vh, vv version of s1
    boi = []
    add_doy = 1 if use_s2 and use_s1 and include_doy else 0
    # TODO: change these to be constants in constants.py eventually
    start_idx = 2 if use_s2 and use_s1 else 0
    end_idx = 5 if use_s2 and use_s1 else 3
    if model_name in ['fcn_crnn', 'bidir_clstm','unet3d']:
        for idx, b in enumerate(best):
            boi.append(inputs[idx, b, start_idx+add_doy:end_idx+add_doy, :, :].unsqueeze(0))
        boi = torch.cat(boi, dim=0)
    elif model_name in ['fcn', 'unet'] and time_slice is not None:
        boi = inputs[:, start_idx+add_doy:end_idx+add_doy, :, :]
    elif model_name in ['unet'] and time_slice is None:
        inputs = inputs.view(inputs.shape[0], MIN_TIMESTAMPS, -1, inputs.shape[2], inputs.shape[3])  
        for idx, b in enumerate(best):
            boi.append(inputs[idx, b, start_idx+add_doy:end_idx+add_doy, :, :].unsqueeze(0))
        boi = torch.cat(boi, dim=0)
            
    # Clip and show input bands of interest
    boi = clip_boi(boi)
    if show_visdom:
        visdom_plot_images(vis, boi, 'Input Images') 

    # Show targets (labels)
    disp_targets = np.concatenate((np.zeros_like(label_mask), targets.numpy()), axis=1)
    disp_targets = np.argmax(disp_targets, axis=1)
    disp_targets = np.expand_dims(disp_targets, axis=1)
    disp_targets = visualize_rgb(disp_targets, num_classes)
    if show_visdom:
        visdom_plot_images(vis, disp_targets, 'Target Images')

    # Show predictions, masked with label mask
    disp_preds = np.argmax(preds.detach().cpu().numpy(), axis=1) + 1
    disp_preds = np.expand_dims(disp_preds, axis=1)
    disp_preds = visualize_rgb(disp_preds, num_classes)
    disp_preds_w_mask = disp_preds * label_mask

    if show_visdom:
        visdom_plot_images(vis, disp_preds, 'Predicted Images')
        visdom_plot_images(vis, disp_preds_w_mask, 'Predicted Images with Label Mask')

    # Show gradnorm per batch
    if show_visdom:
        if split == 'train':
            visdom_plot_metric('gradnorm', split, 'Grad Norm', 'Batch', 'Norm', vis_data, vis)
    
    # TODO: put this into a separate helper function?
    if save:
        save_dir = save_dir.replace(" ", "")
        save_dir = save_dir.replace(":", "")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(torch.from_numpy(label_mask), os.path.join(save_dir, 'label_masks.png'), nrow=NROW, normalize=True) 
        save_image(boi, os.path.join(save_dir, 'inputs.png'), nrow=NROW, normalize=True)
        save_image(torch.from_numpy(disp_targets), os.path.join(save_dir, 'targets.png'), nrow=NROW, normalize=True) 
        save_image(torch.from_numpy(disp_preds), os.path.join(save_dir, 'preds.png'), nrow=NROW, normalize=True)
        save_image(torch.from_numpy(disp_preds_w_mask), os.path.join(save_dir, 'preds_w_masks.png'), nrow=NROW, normalize=True)
    
    if show_matplot:
        labels_grid = make_grid(torch.from_numpy(label_mask), nrow=NROW, normalize=True, padding=8, pad_value=255) 
        inputs_grid = make_grid(boi, nrow=NROW, normalize=True, padding=8, pad_value=255)
        targets_grid = make_grid(torch.from_numpy(disp_targets), nrow=NROW, normalize=True, padding=8, pad_value=255) 
        preds_grid = make_grid(torch.from_numpy(disp_preds), nrow=NROW, normalize=True, padding=8, pad_value=255)
        predsmask_grid = make_grid(torch.from_numpy(disp_preds_w_mask), nrow=NROW, normalize=True, padding=8, pad_value=255)
        return labels_grid, inputs_grid, targets_grid, preds_grid, predsmask_grid

def clip_boi(boi):
    """ Clip bands of interest outside of 2*std per image sample
    """
    for sample in range(boi.shape[0]):
        sample_mean = torch.mean(boi[sample, :, :, :])
        sample_std = torch.std(boi[sample, :, :, :])
        min_clip = sample_mean - 2*sample_std
        max_clip = sample_mean + 2*sample_std

        boi[sample, :, :, :][boi[sample, :, :, :] < min_clip] = min_clip
        boi[sample, :, :, :][boi[sample, :, :, :] > max_clip] = max_clip
   
        boi[sample, :, :, :] = (boi[sample, :, :, :] - min_clip)/(max_clip - min_clip)
    return boi

def record_epoch(all_metrics, split, vis_data, vis, epoch_num, country, save=False, save_dir=None):
    """ Record values for epoch in visdom
    """
    if country == 'ghana':
        class_names = GHANA_CROPS
    elif country == 'southsudan':
        class_names = SOUTHSUDAN_CROPS

    if all_metrics[f'{split}_loss'] is not None: loss_epoch = all_metrics[f'{split}_loss'] / all_metrics[f'{split}_pix']
    if all_metrics[f'{split}_correct'] is not None: acc_epoch = all_metrics[f'{split}_correct'] / all_metrics[f'{split}_pix']

    # Don't append if you are saving. Information has already been appended!
    if save == False:
        vis_data[f'{split}_loss'].append(loss_epoch)
        vis_data[f'{split}_acc'].append(acc_epoch)
        vis_data[f'{split}_f1'].append(metrics.get_f1score(all_metrics[f'{split}_cm'], avg=True))

        if vis_data[f'{split}_classf1'] is None:
            vis_data[f'{split}_classf1'] = metrics.get_f1score(all_metrics[f'{split}_cm'], avg=False)
            vis_data[f'{split}_classf1'] = np.vstack(vis_data[f'{split}_classf1']).T
        else:
            vis_data[f'{split}_classf1'] = np.vstack((vis_data[f'{split}_classf1'], metrics.get_f1score(all_metrics[f'{split}_cm'], avg=False)))

    for cur_metric in ['loss', 'acc', 'f1']:
        visdom_plot_metric(cur_metric, split, f'{split} {cur_metric}', 'Epoch', cur_metric, vis_data, vis)
        if save:
            save_dir = save_dir.replace(" ", "")
            save_dir = save_dir.replace(":", "")        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            visdom_save_metric(cur_metric, split, f'{split}{cur_metric}', 'Epoch', cur_metric, vis_data, save_dir)

    visdom_plot_many_metrics('classf1', split, f'{split}_per_class_f1-score', 'Epoch', 'per class f1-score', class_names, vis_data, vis)

    fig = util.plot_confusion_matrix(all_metrics[f'{split}_cm'], class_names,
                                     normalize=False,
                                     title='{} confusion matrix, epoch {}'.format(split, epoch_num),
                                     cmap=plt.cm.Blues)

    vis.matplot(fig, win=f'{split} CM')
    if save: 
        visdom_save_many_metrics('classf1', split, f'{split}_per_class_f1', 'Epoch', 'per class f1-score', class_names, vis_data, save_dir)               
        fig.savefig(os.path.join(save_dir, f'{split}_cm.png')) 

def visualize_rgb(argmax_array, num_classes, class_colors=None): 
    mask = []
    rgb_output = np.zeros((argmax_array.shape[0], 3, argmax_array.shape[2], argmax_array.shape[3]))

    if class_colors == None:
        rgbs = [ [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255] ]
        rgbs = rgbs[:num_classes]
 
    assert len(rgbs) == num_classes

    for cur_class in range(0, num_classes):
        tmp = np.asarray([argmax_array == cur_class+1])[0]

        mask_cat = np.concatenate((tmp, tmp, tmp), axis=1)

        class_vals = np.concatenate((np.ones_like(tmp)*rgbs[cur_class][0],
                                     np.ones_like(tmp)*rgbs[cur_class][1],
                                     np.ones_like(tmp)*rgbs[cur_class][2]), axis=1) 

        rgb_output += (mask_cat * class_vals)
        
    return rgb_output

