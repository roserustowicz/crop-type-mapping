import torch 
import torch.nn as nn

from constants import *

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

                
def get_num_bands(kwargs):
    num_bands = -1
    added_doy = 0
    added_clouds = 0

    if kwargs.get('include_doy'):
        added_doy = 1
    if kwargs.get('include_clouds') and kwargs.get('use_s2'): 
        added_clouds = 1

    if kwargs.get('use_s1') and kwargs.get('use_s2'):
        num_bands = S1_NUM_BANDS + kwargs.get('s2_num_bands') + 2*added_doy + added_clouds
    elif kwargs.get('use_s1'):
        num_bands = S1_NUM_BANDS + added_doy + added_clouds
    elif kwargs.get('use_s2'):
        num_bands = kwargs.get('s2_num_bands') + added_doy + added_clouds
    else:
        raise ValueError("S1 / S2 usage not specified in args!")
    return num_bands

def get_num_s1_bands(kwargs):
    num_bands = -1
    added_doy = 0
    added_clouds = 0

    if kwargs.get('include_doy'):
        added_doy = 1
        
    num_bands = S1_NUM_BANDS + added_doy
    
    return num_bands


def get_num_s2_bands(kwargs):
    num_bands = -1
    added_doy = 0
    added_clouds = 0

    if kwargs.get('include_doy'):
        added_doy = 1
    if kwargs.get('include_clouds') and kwargs.get('use_s2'): 
        added_clouds = 1

    num_bands = kwargs.get('s2_num_bands') + added_doy + added_clouds
    
    return num_bands


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling for FCN"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()