import numpy as np
import torch

def get_accuracy(model_preds, targets, reduction='avg'):
    batch, classes, rows, cols = targets.shape

    # Reshape truth labels into [N, num_classes]
    # [batch x classes x rows x cols] --> [batch x rows x cols x classes]
    y_true = targets.permute(0, 2, 3, 1)
    # [batch x rows x cols x classes] --> [batch*rows*cols x classes]
    y_true = y_true.contiguous().view(-1, y_true.shape[3])

    # Get number of valid examples
    y_true_npy = y_true.cpu().numpy()
    num_pixels = int(np.sum(y_true_npy))

    # Reshape predictions into [N, num_classes]
    # [batch x classes x rows x cols] --> [batch x rows x cols x classes]
    y_pred = model_preds.permute(0, 2, 3, 1)
    # [batch x rows x cols x classes] --> [batch*rows*cols x classes]
    y_pred = y_pred.contiguous().view(-1, y_pred.shape[3])

    # Create mask for valid pixel locations
    loss_mask = torch.sum(y_true, dim=1).type(torch.LongTensor) 

    # Take argmax for labels and targets
    _, y_true = torch.max(y_true, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)

    # Multiply by mask
    y_true = y_true * loss_mask
    y_pred = y_pred.cpu() * loss_mask

    bool1 = [y_true.numpy() == y_pred.numpy()]
    bool2 = [loss_mask.numpy() == 1]
    
    total_correct = np.sum(np.logical_and(bool1, bool2))
 
    if reduction == 'avg':
        return total_correct / (num_pixels + 1)
    else:
        return total_correct, num_pixels + 1

