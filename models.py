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
from modelling.attention import ApplyAtt, attn_or_avg

# TODO: figure out how to decompose this       
class FCN_CRNN(nn.Module):
    def __init__(self, fcn_input_size, crnn_input_size, crnn_model_name, 
                 hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, avg_hidden_states, 
                 num_classes, bidirectional, pretrained, early_feats, use_planet, resize_planet, 
                 num_bands_dict, main_crnn, main_attn_type, attn_dims, 
                 enc_crnn, enc_attn, enc_attn_type):
        super(FCN_CRNN, self).__init__()

        self.fcn_input_size = fcn_input_size
        self.crnn_input_size = crnn_input_size
        self.hidden_dims = hidden_dims
        self.lstm_kernel_sizes = lstm_kernel_sizes
        self.conv_kernel_size = conv_kernel_size
        self.lstm_num_layers = lstm_num_layers
        self.avg_hidden_states = avg_hidden_states
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.early_feats = early_feats
        self.use_planet = use_planet
        self.resize_planet = resize_planet
        self.num_bands_dict = num_bands_dict       
        self.main_attn_type = main_attn_type
        self.attn_dims = attn_dims
        self.main_crnn = main_crnn
        self.enc_crnn = enc_crnn
        self.enc_attn = enc_attn
        self.enc_attn_type = enc_attn_type
        self.processed_feats = {'main': None, 'enc4': None, 'enc3': None, 'enc2': None, 'enc1': None }

        # get appropriate encoder / decoder
        if not self.early_feats:
            self.fcn = make_UNet_model(n_class=crnn_input_size[1], num_bands_dict=num_bands_dict, late_feats_for_fcn=True, 
                                       pretrained=pretrained, use_planet=use_planet, resize_planet=resize_planet)
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
            self.attns = self.get_attns()
            self.crnns = self.get_crnns()  
            self.final_convs = self.get_final_convs()

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hres_inputs=None):
        batch, timestamps, bands, rows, cols = input_tensor.size()
        fcn_input = input_tensor.view(batch * timestamps, bands, rows, cols)

        if len(hres_inputs.shape) > 1:
            _, _, hbands, hrows, hcols = hres_inputs.size()
            fcn_input_hres = hres_inputs.view(batch * timestamps, hbands, hrows, hcols)  
        else: fcn_input_hres = None 

        if self.early_feats:
            # Encode features
            center1_feats, enc4_feats, enc3_feats, enc2_feats, enc1_feats = self.fcn_enc(fcn_input, fcn_input_hres)

            for cur_feats, cur_enc in zip([center1_feats, enc4_feats, enc3_feats, enc2_feats, enc1_feats], self.crnns):
                if cur_feats is not None:
                    # Apply CRNN
                    cur_feats = cur_feats.view(batch, timestamps, -1, cur_feats.shape[-2], cur_feats.shape[-1])
                    if self.crnns[cur_enc] is not None:
                        cur_feats_fwd, cur_feats_rev = self.crnns[cur_enc](cur_feats) 
                    else:
                        cur_feats_fwd = cur_feats
                        cur_feats_rev = None

                    # Apply attention
                    reweighted = attn_or_avg(self.attns[cur_enc], self.avg_hidden_states, cur_feats_fwd, cur_feats_rev, self.bidirectional)
                    # Apply final conv
                    final_feats = self.final_convs[cur_enc](reweighted) if self.final_convs[cur_enc] is not None else reweighted
                    self.processed_feats[cur_enc] = final_feats

            # Decode and predict
            preds = self.fcn_dec(self.processed_feats['main'], self.processed_feats['enc4'], self.processed_feats['enc3'], 
                                                               self.processed_feats['enc2'], self.processed_feats['enc1'])
        
        else:
            # Encode and decode features
            fcn_output = self.fcn(fcn_input, fcn_input_hres)
            
            # Apply CRNN
            crnn_input = fcn_output.view(batch, timestamps, -1, fcn_output.shape[-2], fcn_output.shape[-1])
            if self.crnn_main(crnn_input) is not None:
                crnn_output_fwd, crnn_output_rev = self.crnn_main(crnn_input)
            else:
                crnn_output_fwd = crnn_input
                crnn_output_rev = None
            
            # Apply attention
            reweighted = attn_or_avg(self.attns['main'], self.avg_hidden_states, crnn_output_fwd, crnn_output_rev, bidirectional)
                    
            # Apply final conv
            scores = self.final_convs['main'](reweighted)
            preds = self.logsoftmax(scores)            
        
        return preds

    def get_crnns(self):
        self.crnn_main = self.crnn_enc4 = self.crnn_enc3 = self.crnn_enc2 = self.crnn_enc1 = None
        if self.early_feats:
            if self.main_crnn:
                self.crnn_main = CLSTMSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes, 
                                   self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1], self.bidirectional) 
            if self.enc_crnn:
                crnn_input0, crnn_input1, crnn_input2, crnn_input3 = self.crnn_input_size
                self.crnn_enc4 = CLSTMSegmenter([crnn_input0, crnn_input1//2, crnn_input2*2, crnn_input3*2], self.hidden_dims, self.lstm_kernel_sizes, 
                                   self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//2, self.bidirectional)
                self.crnn_enc3 = CLSTMSegmenter([crnn_input0, crnn_input1//4, crnn_input2*4, crnn_input3*4], self.hidden_dims, self.lstm_kernel_sizes, 
                                   self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//4, self.bidirectional)
                if self.use_planet and not self.resize_planet:
                    self.crnn_enc2 = CLSTMSegmenter([crnn_input0, crnn_input1//8, crnn_input2*8, crnn_input3*8], self.hidden_dims, self.lstm_kernel_sizes, 
                                       self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//8, self.bidirectional)
                    self.crnn_enc1 = CLSTMSegmenter([crnn_input0, crnn_input1//16, crnn_input2*16, crnn_input3*16], self.hidden_dims, self.lstm_kernel_sizes, 
                                       self.conv_kernel_size, self.lstm_num_layers, self.crnn_input_size[1]//16, self.bidirectional)
        else:
            self.crnn_main = CLSTMSegmenter(self.crnn_input_size, self.hidden_dims, self.lstm_kernel_sizes, 
                                   self.conv_kernel_size, self.lstm_num_layers, self.num_classes, self.bidirectional)
        self.crnns = { 'main': self.crnn_main, 'enc4': self.crnn_enc4, 'enc3': self.crnn_enc3, 'enc2': self.crnn_enc2, 'enc1': self.crnn_enc1 }
        return self.crnns

    def get_attns(self):
        self.attn_enc4 = self.attn_enc3 = self.attn_enc2 = self.attn_enc1 = None
        if self.early_feats:
            if self.enc_attn:
                self.attn_enc4 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims) 
                self.attn_enc3 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims)
                if self.use_planet and not self.resize_planet:
                    self.attn_enc2 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims)
                    self.attn_enc1 = ApplyAtt(self.enc_attn_type, self.hidden_dims, self.attn_dims)
        self.attn_main = ApplyAtt(self.main_attn_type, self.hidden_dims, self.attn_dims)
        self.attns = { 'main': self.attn_main, 'enc4': self.attn_enc4, 'enc3': self.attn_enc3, 'enc2': self.attn_enc2, 'enc1': self.attn_enc1 } 
        return self.attns

    def get_final_convs(self):
        self.enc4_finalconv = self.enc3_finalconv = self.enc2_finalconv = self.enc1_finalconv = None
        if self.early_feats:
            self.main_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1], kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
            if self.enc_crnn:
                self.enc4_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1]//2, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
                self.enc3_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1]//4, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
                if self.use_planet and not self.resize_planet: 
                    self.enc2_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1]//8, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
                    self.enc1_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.crnn_input_size[1]//16, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
        else:
            self.main_finalconv = nn.Conv2d(in_channels=self.hidden_dims, out_channels=self.num_classes, kernel_size=self.conv_kernel_size, padding=int((self.conv_kernel_size-1)/2))
        self.final_convs = { 'main': self.main_finalconv, 'enc4': self.enc4_finalconv, 'enc3': self.enc3_finalconv, 'enc2': self.enc2_finalconv, 'enc1': self.enc1_finalconv}
        return self.final_convs

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
                        grid_size,
                        main_attn_type,
                        attn_dims): 

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
                     grid_size,
                     main_attn_type,
                     attn_dims)
    return model

def make_bidir_clstm_model(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, 
                           bidirectional, avg_hidden_states, main_attn_type, attn_dims):
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
    model = CLSTMSegmenter(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional, 
                           with_pred=True, avg_hidden_states=avg_hidden_states, attn_type=main_attn_type, attn_dims=attn_dims) 
    return model


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

def make_fcn_clstm_model(country, fcn_input_size, crnn_input_size, crnn_model_name, 
                         hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, avg_hidden_states,
                         num_classes, bidirectional, pretrained, early_feats, use_planet, resize_planet,
                         num_bands_dict, main_crnn, main_attn_type, attn_dims,
                         enc_crnn, enc_attn, enc_attn_type):
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

    model = FCN_CRNN(fcn_input_size, crnn_input_size, crnn_model_name, hidden_dims, lstm_kernel_sizes, 
                     conv_kernel_size, lstm_num_layers, avg_hidden_states, num_classes, bidirectional, pretrained, 
                     early_feats, use_planet, resize_planet, num_bands_dict, main_crnn, main_attn_type, attn_dims, 
                     enc_crnn, enc_attn, enc_attn_type)
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
                                       main_attn_type=kwargs.get('main_attn_type'),
                                       attn_dims = {'d': kwargs.get('d_attn_dim'), 'r': kwargs.get('r_attn_dim'),
                                                    'dk': kwargs.get('dk_attn_dim'), 'dv': kwargs.get('dv_attn_dim')})

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
                                     crnn_input_size=(num_timesteps, kwargs.get('fcn_out_feats')), 
                                     crnn_model_name=kwargs.get('crnn_model_name'),
                                     hidden_dims=kwargs.get('hidden_dims'), 
                                     lstm_kernel_sizes=(kwargs.get('crnn_kernel_sizes'), kwargs.get('crnn_kernel_sizes')), 
                                     conv_kernel_size=kwargs.get('conv_kernel_size'), 
                                     lstm_num_layers=kwargs.get('crnn_num_layers'), 
                                     avg_hidden_states=kwargs.get('avg_hidden_states'), 
                                     num_classes=NUM_CLASSES[kwargs.get('country')],
                                     bidirectional=kwargs.get('bidirectional'),                                       
                                     pretrained=kwargs.get('pretrained'),
                                     early_feats=kwargs.get('early_feats'),
                                     use_planet=kwargs.get('use_planet'),
                                     resize_planet=kwargs.get('resize_planet'), 
                                     num_bands_dict=num_bands,
                                     main_crnn=kwargs.get('main_crnn'),
                                     main_attn_type=kwargs.get('main_attn_type'),
                                     attn_dims = {'d': kwargs.get('d_attn_dim'), 'r': kwargs.get('r_attn_dim'),
                                                  'dk': kwargs.get('dk_attn_dim'), 'dv': kwargs.get('dv_attn_dim')},
                                     enc_crnn=kwargs.get('enc_crnn'),
                                     enc_attn=kwargs.get('enc_attn'),
                                     enc_attn_type=kwargs.get('enc_attn_type'))

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
   
        all_bands = get_num_bands(kwargs)['s1'] + get_num_bands(kwargs)['s2'] + get_num_bands(kwargs)['planet']
        num_bands = {'s1': get_num_bands(kwargs)['s1'], 's2': get_num_bands(kwargs)['s2'], 'planet': get_num_bands(kwargs)['planet'], 'all': all_bands }

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
                                    grid_size=GRID_SIZE[country], 
                                    main_attn_type=kwargs.get('main_attn_type'), 
                                    attn_dims={'d': kwargs.get('d_attn_dim'), 'r': kwargs.get('r_attn_dim'), 
                                               'dv': kwargs.get('dv_attn_dim'), 'dk':kwargs.get('dk_attn_dim')})
    else:
        raise ValueError(f"Model {model_name} unsupported, check `model_name` arg") 
        

    return model

