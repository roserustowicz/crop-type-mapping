"""

File housing all models.

Each model can be created by invoking the appropriate function
given by:

    make_MODELNAME_model(MODEL_SETTINGS)

Changes to allow this are still in progess
"""


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import torchfcn
import fcn

from constants import *
from modelling.baselines import make_rf_model, make_logreg_model, make_1d_nn_model, make_1d_2layer_nn_model, make_1d_cnn_model
from modelling.recurrent_norm import RecurrentNorm2d
from modelling.clstm_cell import ConvLSTMCell
from modelling.clstm import CLSTM
from modelling.cgru_segmenter import CGRUSegmenter
from modelling.clstm_segmenter import CLSTMSegmenter
from modelling.util import initialize_weights, get_num_bands, get_upsampling_weight, set_parameter_requires_grad
from modelling.fcn8 import FCN8
from modelling.unet import UNet, UNet_Encode, UNet_Decode
from modelling.unet3d import UNet3D
from modelling.multi_input_clstm import MI_CLSTM


# TODO: figure out how to decompose this       
class FCN_CRNN(nn.Module):
    def __init__(self, fcn_input_size, fcn_model_name, crnn_input_size, crnn_model_name, 
                 hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, avg_hidden_states, 
                 num_classes, bidirectional, pretrained, early_feats, use_planet, resize_planet, num_bands_dict):
        super(FCN_CRNN, self).__init__()

        self.early_feats = early_feats

        if fcn_model_name == 'simpleCNN':
            self.fcn = simple_CNN(fcn_input_size, crnn_input_size[1])
        elif fcn_model_name == 'fcn8':
            self.fcn = make_fcn_model(crnn_input_size[1], fcn_input_size[1], freeze=False)
        elif fcn_model_name == 'unet':
            if not self.early_feats:
                self.fcn = make_UNet_model(crnn_input_size[1], num_bands_dict, late_feats_for_fcn=True, pretrained=pretrained, use_planet=use_planet, resize_planet=resize_planet)
            else:
                self.fcn_enc = make_UNetEncoder_model(num_bands_dict, use_planet=use_planet, resize_planet=resize_planet, pretrained=pretrained)
                self.fcn_dec = make_UNetDecoder_model(num_classes, late_feats_for_fcn=False,  use_planet=use_planet, resize_planet=resize_planet)
        
        if crnn_model_name == "gru":
            if self.early_feats:
                self.crnn = CGRUSegmenter(crnn_input_size, hidden_dims, lstm_kernel_sizes, 
                                      conv_kernel_size, lstm_num_layers, crnn_input_size[1], bidirectional, avg_hidden_states)
            else:
                self.crnn = CGRUSegmenter(crnn_input_size, hidden_dims, lstm_kernel_sizes, 
                                      conv_kernel_size, lstm_num_layers, num_classes, bidirectional, avg_hidden_states)
        elif crnn_model_name == "clstm":
            if self.early_feats:
                self.crnn = CLSTMSegmenter(crnn_input_size, hidden_dims, lstm_kernel_sizes, 
                                       conv_kernel_size, lstm_num_layers, crnn_input_size[1], bidirectional, avg_hidden_states, self.early_feats)
            else:
                self.crnn = CLSTMSegmenter(crnn_input_size, hidden_dims, lstm_kernel_sizes, 
                                       conv_kernel_size, lstm_num_layers, num_classes, bidirectional, avg_hidden_states, self.early_feats)

    def forward(self, input_tensor, hres_inputs=None):
        batch, timestamps, bands, rows, cols = input_tensor.size()
        fcn_input = input_tensor.view(batch * timestamps, bands, rows, cols)

        if len(hres_inputs.shape) > 1:
            _, _, hbands, hrows, hcols = hres_inputs.size()
            fcn_input_hres = hres_inputs.view(batch * timestamps, hbands, hrows, hcols)  
        else: fcn_input_hres = None 

        if self.early_feats:
            center1_feats, enc4_feats, enc3_feats, enc2_feats, enc1_feats = self.fcn_enc(fcn_input, fcn_input_hres)

            # Reshape tensors to separate batch and timestamps
            # TODO: Use attn weights here instead of averaging??
            crnn_input = center1_feats.view(batch, timestamps, -1, center1_feats.shape[-2], center1_feats.shape[-1])
            enc4_feats = enc4_feats.view(batch, timestamps, -1, enc4_feats.shape[-2], enc4_feats.shape[-1])
            enc4_feats = torch.mean(enc4_feats, dim=1, keepdim=False)
            enc3_feats = enc3_feats.view(batch, timestamps, -1, enc3_feats.shape[-2], enc3_feats.shape[-1])
            enc3_feats = torch.mean(enc3_feats, dim=1, keepdim=False)

            if enc2_feats is not None:
                enc2_feats = enc2_feats.view(batch, timestamps, -1, enc2_feats.shape[-2], enc2_feats.shape[-1])
                enc2_feats = torch.mean(enc2_feats, dim=1, keepdim=False)
                enc1_feats = enc1_feats.view(batch, timestamps, -1, enc1_feats.shape[-2], enc1_feats.shape[-1])
                enc1_feats = torch.mean(enc1_feats, dim=1, keepdim=False)

            pred_enc = self.crnn(crnn_input)
            preds = self.fcn_dec(pred_enc, enc4_feats, enc3_feats, enc2_feats, enc1_feats)
        else:
            fcn_output = self.fcn(fcn_input, fcn_input_hres)
            crnn_input = fcn_output.view(batch, timestamps, -1, fcn_output.shape[-2], fcn_output.shape[-1])
            preds = self.crnn(crnn_input)
        
        return preds

def make_MI_CLSTM_model(num_bands, 
                        unet_out_channels,
                        crnn_input_size,
                        hidden_dims, 
                        lstm_kernel_sizes, 
                        lstm_num_layers, 
                        conv_kernel_size, 
                        num_classes, 
                        avg_hidden_states, 
                        early_feats, 
                        bidirectional,
                        max_timesteps,
                        satellites,
                        resize_planet,
                        grid_size):
    
    model = MI_CLSTM(num_bands,
                     unet_out_channels,
                     crnn_input_size,
                     hidden_dims, 
                     lstm_kernel_sizes, 
                     conv_kernel_size, 
                     lstm_num_layers, 
                     avg_hidden_states, 
                     num_classes,
                     early_feats,
                     bidirectional,
                     max_timesteps,
                     satellites,
                     resize_planet,
                     grid_size)
    return model

def make_bidir_clstm_model(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional, avg_hidden_states, early_feats):
    """ Defines a (bidirectional) CLSTM model 
    Args:
        input_size - (tuple) size of input dimensions 
        hidden_dims - (int or list) num features for hidden layers 
        lstm_kernel_sizes - (int) kernel size for lstm cells
        conv_kernel_size - (int) ketnel size for convolutional layers
        lstm_num_layers - (int) number of lstm cells to stack
        num_classes - (int) number of classes to predict
        bidirectional - (bool) if True, include reverse inputs and concatenate output features from forward and reverse models
                               if False, use only forward inputs and features
    
    Returns:
      returns the model! 
    """
    clstm_segmenter = CLSTMSegmenter(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional, avg_hidden_states, early_feats)

    return clstm_segmenter


def make_fcn_model(n_class, n_channel, freeze=True):
    """ Defines a FCN8s model
    Args: 
      n_class - (int) number of classes to predict
      n_channel - (int) number of channels in input
      freeze - (bool) whether to use pre-trained weights
                TODO: unfreeze after x epochs of training

    Returns: 
      returns the model!
    """
    ## load pretrained model
    fcn8s_pretrained_model=torch.load(torchfcn.models.FCN8s.download())
    fcn8s = FCN8(n_class, n_channel)
    fcn8s.load_state_dict(fcn8s_pretrained_model,strict=False)
    
    if freeze:
        ## Freeze the parameter you do not want to tune
        for param in fcn8s.parameters():
            if torch.sum(param==0)==0:
                param.requires_grad = False
    
    return fcn8s
def make_UNet_model(n_class, num_bands_dict, late_feats_for_fcn=False, pretrained=True, use_planet=False, resize_planet=False):
    """ Defines a U-Net model
    Args:
      n_class - (int) number of classes to predict
      n_channel - (int) number of channels in input
      for_fcn - (bool) whether or not U-Net is to be used for FCN + CLSTM, 
                 or false if just used as a U-Net. When True, the last conv and 
                 softmax layer is removed and features are returned. When False, 
                 the softmax layer is kept and probabilities are returned. 
      pretrained - (bool) whether to use pre-trained weights

    Returns: 
      returns the model!
    """
    model = UNet(n_class, num_bands_dict, late_feats_for_fcn, use_planet, resize_planet)
    
    if pretrained:
        # TODO: Why are pretrained weights from vgg13? 
        pre_trained = models.vgg13(pretrained=True)
        pre_trained_features = list(pre_trained.features)

        model.unet_encode.enc3.encode[3] = pre_trained_features[2] #  64 in,  64 out
        model.unet_encode.enc4.encode[0] = pre_trained_features[5] #  64 in, 128 out
        model.unet_encode.enc4.encode[3] = pre_trained_features[7] # 128 in, 128 out
        model.unet_encode.center[0] = pre_trained_features[10]     # 128 in, 256 out
        
    model = model.cuda()
    return model

def make_UNetEncoder_model(num_bands_dict, use_planet=True, resize_planet=False, pretrained=True):
    model = UNet_Encode(num_bands_dict, use_planet, resize_planet)
    
    if pretrained:
       # TODO: Why are pretrained weights from vgg13? 
        pre_trained = models.vgg13(pretrained=True)
        pre_trained_features = list(pre_trained.features)

        model.enc3.encode[3] = pre_trained_features[2] #  64 in,  64 out
        model.enc4.encode[0] = pre_trained_features[5] #  64 in, 128 out
        model.enc4.encode[3] = pre_trained_features[7] # 128 in, 128 out
        model.center[0] = pre_trained_features[10]     # 128 in, 256 out

    model = model.cuda()
    return model

def make_UNetDecoder_model(n_class, late_feats_for_fcn, use_planet, resize_planet):
    model = UNet_Decode(n_class, late_feats_for_fcn, use_planet, resize_planet)
    model = model.cuda()
    return model

def make_fcn_clstm_model(country, fcn_input_size, fcn_model_name, crnn_input_size, crnn_model_name, 
                         hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, avg_hidden_states,
                         num_classes, bidirectional, pretrained, early_feats, use_planet, resize_planet,
                         num_bands_dict):
    """ Defines a fully-convolutional-network + CLSTM model
    Args:
      fcn_input_size - (tuple) input dimensions for FCN model
      fcn_model_name - (str) model name used as the FCN portion of the network
      crnn_input_size - (tuple) input dimensions for CRNN model
      crnn_model_name - (str) model name used as the convolutional RNN portion of the network 
      hidden_dims - (int or list) num features for hidden layers 
      lstm_kernel_sizes - (int) kernel size for lstm cells
      conv_kernel_size - (int) ketnel size for convolutional layers
      lstm_num_layers - (int) number of lstm cells to stack
      num_classes - (int) number of classes to predict
      bidirectional - (bool) if True, include reverse inputs and concatenate output features from forward and reverse models
                               if False, use only forward inputs and features
      pretrained - (bool) whether to use pre-trained weights

    Returns: 
      returns the model!
    """
    if early_feats:
        crnn_input_size += (GRID_SIZE[country] // 4, GRID_SIZE[country] // 4)
    else:
        crnn_input_size += (GRID_SIZE[country], GRID_SIZE[country]) 

    model = FCN_CRNN(fcn_input_size, fcn_model_name, crnn_input_size, crnn_model_name, hidden_dims, lstm_kernel_sizes, 
                     conv_kernel_size, lstm_num_layers, avg_hidden_states, num_classes, bidirectional, pretrained, 
                     early_feats, use_planet, resize_planet, num_bands_dict)
    model = model.cuda()

    return model

def make_UNet3D_model(n_class, n_channel, timesteps, dropout):
    """ Defined a 3d U-Net model
    Args: 
      n_class - (int) number of classes to predict
      n_channels - (int) number of input channgels

    Returns:
      returns the model!
    """

    model = UNet3D(n_channel, n_class, timesteps, dropout)
    model = model.cuda()
    return model

def get_model(model_name, **kwargs):
    """ Get appropriate model based on model_name and input arguments
    Args: 
      model_name - (str) which model to use 
      kwargs - input arguments corresponding to the model name

    Returns: 
      returns the model!
    """

    model = None

    if model_name == 'random_forest':
        # use class weights
        class_weight=None
        if kwargs.get('loss_weight'):
            class_weight = 'balanced'

        model = make_rf_model(random_state=kwargs.get('seed', None),
                              n_jobs=kwargs.get('n_jobs', None),
                              n_estimators=kwargs.get('n_estimators', 100),
                              class_weight=class_weight)

    elif model_name == 'bidir_clstm':
        num_bands = get_num_bands(kwargs)['all']
        num_timesteps = kwargs.get('num_timesteps')

        # TODO: change the timestamps passed in to be more flexible (i.e allow specify variable length / fixed / truncuate / pad)
        # TODO: don't hardcode values
        model = make_bidir_clstm_model(input_size=(num_timesteps, num_bands, GRID_SIZE[kwargs.get('country')], GRID_SIZE[kwargs.get('country')]), 
                                       hidden_dims=kwargs.get('hidden_dims'), 
                                       lstm_kernel_sizes=(kwargs.get('crnn_kernel_sizes'), kwargs.get('crnn_kernel_sizes')), 
                                       conv_kernel_size=kwargs.get('conv_kernel_size'), 
                                       lstm_num_layers=kwargs.get('crnn_num_layers'),
                                       num_classes=NUM_CLASSES[kwargs.get('country')],
                                       bidirectional=kwargs.get('bidirectional'),
                                       avg_hidden_states=kwargs.get('avg_hidden_states'),
                                       early_feats=kwargs.get('early_feats'))
    elif model_name == 'fcn':
        num_bands = get_num_bands(kwargs)['all']
        model = make_fcn_model(n_class=NUM_CLASSES[kwargs.get('country')], n_channel = num_bands, freeze=True)
    
    elif model_name == 'unet':
        num_bands = get_num_bands(kwargs)['all']
        num_timesteps = kwargs.get('num_timesteps')
        
        if kwargs.get('time_slice') is None:
            model = make_UNet_model(n_class=NUM_CLASSES[kwargs.get('country')], n_channel = num_bands*num_timesteps)
        else: 
            model = make_UNet_model(n_class=NUM_CLASSES[kwargs.get('country')], n_channel = num_bands)
    
    elif model_name == 'fcn_crnn':
        num_bands = get_num_bands(kwargs)
        num_timesteps = kwargs.get('num_timesteps')
        fix_feats = kwargs.get('fix_feats')
        pretrained_model_path = kwargs.get('pretrained_model_path')

        model = make_fcn_clstm_model(country=kwargs.get('country'),
                                     fcn_input_size=(num_timesteps, num_bands['all'], GRID_SIZE[kwargs.get('country')], GRID_SIZE[kwargs.get('country')]), 
                                     fcn_model_name=kwargs.get('fcn_model_name'),
                                     crnn_input_size=(num_timesteps, kwargs.get('fcn_out_feats')), 
                                     crnn_model_name=kwargs.get('crnn_model_name'),
                                     hidden_dims=kwargs.get('hidden_dims'), 
                                     lstm_kernel_sizes=(kwargs.get('crnn_kernel_sizes'), kwargs.get('crnn_kernel_sizes')), 
                                     conv_kernel_size=kwargs.get('conv_kernel_size'), 
                                     lstm_num_layers=kwargs.get('crnn_num_layers'), 
                                     avg_hidden_states=kwargs.get('avg_hidden_states'), 
                                     num_classes=NUM_CLASSES[kwargs.get('country')],
                                     bidirectional=kwargs.get('bidirectional'),                                       
                                     pretrained = kwargs.get('pretrained'),
                                     early_feats = kwargs.get('early_feats'),
                                     use_planet = kwargs.get('use_planet'),
                                     resize_planet = kwargs.get('resize_planet'), 
                                     num_bands_dict = num_bands)

        if (pretrained_model_path is not None) and (kwargs.get('pretrained') == True):
            pre_trained_model=torch.load(pretrained_model_path)
       
            # don't set pretrained weights for weights and bias before predictions 
            #  because number of classes do not agree (i.e. germany has 17 classes)
            dont_set = ['fcn_dec.final.6.weight', 'fcn_dec.final.6.bias']
            updated_keys = []
            for key, value in model.state_dict().items():
                if key in dont_set: continue
                elif key in pre_trained_model:
                    updated_keys.append(key) 
                    weights = pre_trained_model[key]   
                    model.state_dict()[key] = weights

            for name, param in model.named_parameters():
                if name in updated_keys:
                    param.requires_grad = not fix_feats

    elif model_name == 'unet3d':
        num_bands = get_num_bands(kwargs)['all']
        model = make_UNet3D_model(n_class=NUM_CLASSES[kwargs.get('country')], n_channel=num_bands, timesteps=kwargs.get('num_timesteps'), dropout=kwargs.get('dropout'))
    elif model_name == 'mi_clstm':
        satellites = {'s1': kwargs.get('use_s1'), 's2': kwargs.get('use_s2'), 'planet': kwargs.get('use_planet')}
        num_bands = {'s1': get_num_bands(kwargs)['s1'], 's2': get_num_bands(kwargs)['s2'], 'planet': get_num_bands(kwargs)['planet']}
        max_timesteps = kwargs.get('num_timesteps')
        country = kwargs.get('country')
        if kwargs.get('early_feats'):
            crnn_input_size = (max_timesteps, kwargs.get('fcn_out_feats'), GRID_SIZE[country] // 4, GRID_SIZE[country] // 4)
        else:
            crnn_input_size = (max_timesteps, NUM_CLASSES[kwargs.get('country')], GRID_SIZE[country], GRID_SIZE[country])
        
        model = make_MI_CLSTM_model(num_bands=num_bands,
                                    unet_out_channels=kwargs.get('fcn_out_feats'),
                                    crnn_input_size=crnn_input_size,
                                    hidden_dims=kwargs.get('hidden_dims'), 
                                    lstm_kernel_sizes=(kwargs.get('crnn_kernel_sizes'), kwargs.get('crnn_kernel_sizes')), 
                                    conv_kernel_size=kwargs.get('conv_kernel_size'), 
                                    lstm_num_layers=kwargs.get('crnn_num_layers'), 
                                    avg_hidden_states=kwargs.get('avg_hidden_states'), 
                                    num_classes=NUM_CLASSES[kwargs.get('country')],
                                    early_feats=kwargs.get('early_feats'),
                                    bidirectional=kwargs.get('bidirectional'),
                                    max_timesteps = kwargs.get('num_timesteps'),
                                    satellites=satellites,
                                    resize_planet=kwargs.get('resize_planet'),
                                    grid_size=GRID_SIZE[country])
    else:
        raise ValueError(f"Model {model_name} unsupported, check `model_name` arg") 
        

    return model

