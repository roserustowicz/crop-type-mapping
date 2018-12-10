"""

File housing all models.

Each model is created by invoking the appropriate function
given by:

    make_MODELNAME_model(MODEL_SETTINGS)

"""
from keras.models import Sequential, Model
from keras.layers import InputLayer, Activation, BatchNormalization, Flatten, Dropout
from keras.layers import Dense, Conv2D, MaxPooling2D, ConvLSTM2D, Lambda
from keras.layers import Conv1D, MaxPooling1D
from keras import regularizers
from keras.layers import Bidirectional, TimeDistributed, concatenate
from keras.backend import reverse
from keras.engine.input_layer import Input

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import yaml
import torchfcn
import fcn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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


def make_rf_model(random_state, n_jobs, n_estimators):
    """
    Defines a sklearn random forest model. See sci-kit learn
    documentation of sklearn.ensemble.RandomForestClassifier
    for more information and other possible parameters

    Args:
      random_state - (int) random seed
      n_jobs - (int or None) the number of jobs to run in parallel
                for both fit and predict. None means 1, -1 means
                using all processors
      n_estimators - (int) number of trees in the forest

    Returns:
      model - a sklearn random forest model
    """
    model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, n_estimators=n_estimators)
    return model

def make_logreg_model(random_state=None, solver='lbfgs', multi_class='multinomial'):
    """
    Defines a skearn logistic regression model. See ski-kit learn
    documentation of sklearn.linear_model.LogisticRegression for
    more information or other possible parameters

    Args:
      random_state - (int) random seed used to shuffle data
      solver - (str) {'newton-cg', 'lbfgs', 'linlinear', 'sag', 'saga'}
               for multiclass problems, only 'newton-cg', 'sag', 'saga',
               and 'lbfgs' handle multinomial loss. See docs for more info
      multiclass - (str) {'ovr', 'multinomial', 'auto'} for 'ovr', a
                   binary problem is fit for each label. For 'multinomial',
                   the minimized loss is the multinomial loss fit across
                   the entire probability distribution, even when binary.
                   See sci-kit learn docs for more information.

    Returns:
      model - a sklearn logistic regression model
    """
    model = LogisticRegression(random_state, solver, multi_class)
    return model

def make_1d_nn_model(num_classes, num_input_feats, units, reg_strength, input_bands, dropout):
    """ Defines a keras Sequential 1D NN model

    Args:
      num_classes - (int) number of classes to predict
      num_input_feats - (int) number of input features (timestamps)
      units - (int) corresponds to hidden layer features
      reg_stength - (float) constant for regularization strength for model weights
      input_bands - (int) number of input channels
      dropout - (float) constant for percentage of connections to drop during training

    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(units=units, kernel_regularizer=reg, 
              bias_regularizer=reg, input_shape=(num_input_feats, input_bands)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model

def make_1d_2layer_nn_model(num_classes, num_input_feats, units, reg_strength, input_bands, dropout):
    """ Defines a keras Sequential 1D NN model

    Args:
      num_classes - (int) number of classes to predict
      num_input_feats - (int) number of input features (timestamps)
      units - (int) corresponds to hidden layer features
      reg_stength - (float) constant for regularization strength for model weights
      input_bands - (int) number of input channels
      dropout - (float) constant for percentage of connections to drop during training

    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(units=units, kernel_regularizer=reg, 
              bias_regularizer=reg, input_shape=(num_input_feats, input_bands)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(Dense(units=units, kernel_regularizer=reg,
              bias_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model

def make_1d_cnn_model(num_classes, num_input_feats, units, reg_strength, input_bands, dropout):
    """ Defines a keras Sequential 1D CNN model

    Args:
      num_classes - (int) number of classes to predict
      num_input_feats - (int) number of input features (timestamps)
      units - (int) corresponds to hidden layer features
      reg_stength - (float) constant for regularization strength for model weights
      input_bands - (int) number of input channels
      dropout - (float) constant for percentage of connections to drop during training

    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)
    
    model = Sequential()

    model.add(Conv1D(units, kernel_size=3,
              strides=1, padding='same',
              kernel_regularizer=reg,
              bias_regularizer=reg,
              input_shape=(num_input_feats, input_bands)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(units*2, kernel_size=3, padding='same',
              kernel_regularizer=reg, bias_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(units*4, activation='relu', 
              kernel_regularizer=reg, bias_regularizer=reg))
    model.add(Dropout(rate=dropout))
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model


def make_bidir_clstm_model(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional):
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
    clstm_segmenter = CLSTMSegmenter(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional)

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
    fcn8s = FCN8s_croptype(n_class, n_channel)
    fcn8s.load_state_dict(fcn8s_pretrained_model,strict=False)
    
    if freeze:
        ## Freeze the parameter you do not want to tune
        for param in fcn8s.parameters():
            if torch.sum(param==0)==0:
                param.requires_grad = False
    
    return fcn8s

def make_UNet_model(n_class, n_channel, for_fcn=False, pretrained=True):
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
    model = UNet(n_class, n_channel, for_fcn)
    
    if pretrained: 
        pre_trained = models.vgg13(pretrained=True)
        pre_trained_features = list(pre_trained.features)
        model.enc1.encode[3] = pre_trained_features[2]
        model.enc1.encode[6] = pre_trained_features[4]
        model.enc2.encode[0] = pre_trained_features[5]
        model.enc2.encode[3] = pre_trained_features[7]
        model.enc2.encode[6] = pre_trained_features[9]
        model.enc2.encode[6] = pre_trained_features[9]
        model.center.decode[0] = pre_trained_features[10]
        model.center.decode[3] = pre_trained_features[12]
        
    model = model.cuda()
        
    return model

def make_fcn_clstm_model(fcn_input_size, fcn_model_name, crnn_input_size, crnn_model_name, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional, pretrained):
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

    model = FCN_CRNN(fcn_input_size, fcn_model_name, crnn_input_size, crnn_model_name, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional, pretrained)
    model = model.cuda()

    return model

def make_UNet3D_model(n_class, n_channel):
    """ Defined a 3d U-Net model
    Args: 
      n_class - (int) number of classes to predict
      n_channels - (int) number of input channgels

    Returns:
      returns the model!
    """

    model = UNet3D(n_channel, n_class)
    model = model.cuda()
    return model

class _EncoderBlock(nn.Module):
    """ U-Net encoder block
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """ U-Net decoder block
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)
    
    
class UNet(nn.Module):
    """ U-Net architecture definition
    """
    def __init__(self, num_classes, num_channels, for_fcn):
        super(UNet, self).__init__()
        self.for_fcn = for_fcn
        self.enc1 = _EncoderBlock(num_channels, 64)
        self.enc2 = _EncoderBlock(64, 128)
        # self.enc3 = _EncoderBlock(128, 256, dropout=True)
        # self.enc4 = _EncoderBlock(256, 512, dropout=True)
        # self.center = _DecoderBlock(512, 1024, 512)
        self.center = _DecoderBlock(128, 256, 128)
        # self.dec4 = _DecoderBlock(1024, 512, 256)
        # self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, x):
        x = x.cuda()
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        # enc3 = self.enc3(enc2)
        # enc4 = self.enc4(enc3)
        # center = self.center(enc4)
        center = self.center(enc2)
        # dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([center, F.upsample(enc2, center.size()[2:], mode='bilinear')], 1))
        # dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        # dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        final = F.upsample(final, x.size()[2:], mode='bilinear')
        if self.for_fcn:
            return final
        else:
            final = self.softmax(final)
            final = torch.log(final)
            return final

        
def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block(in_dim, middle_dim, out_dim):
    model = nn.Sequential(
        nn.Conv3d(in_dim,middle_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_dim),
        nn.ReLU(inplace=True),
        nn.Conv3d(middle_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True),
    )
    return model


def up_conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU(inplace=True),
    )
    return model


class UNet3D(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(UNet3D, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.en1 = conv_block(in_channel, 32, 64)
        self.pool_1 = maxpool_3d()
        self.en2 = conv_block(64, 64, 128)
        self.pool_2 = maxpool_3d()
        self.en3 = conv_block(128, 128, 256)
        self.pool_3 = maxpool_3d()
        
        self.en4 = conv_block(256, 256, 512)
        
        self.trans4 = up_conv_block(512, 512)
        self.dc4 = conv_block(512+256, 256, 256)
        self.trans3 = up_conv_block(256, 256)
        self.dc3 = conv_block(256+128, 128, 128)
        self.trans2 = up_conv_block(128, 128)
        self.dc2 = conv_block(128+64, 64, 64)
        self.final = nn.Conv3d(64, n_classes, kernel_size=3, stride=1, padding=1)    
        self.fn = nn.Linear(MIN_TIMESTAMPS, 1)
        self.softmax = nn.Softmax2d()
        
    def forward(self, x):
        x = x.cuda()
        en1 = self.en1(x)
        pool_1 = self.pool_1(en1)
        en2 = self.en2(pool_1)
        pool_2 = self.pool_2(en2)
        en3 = self.en3(pool_2)
        pool_3 = self.pool_3(en3)
        
        en4 = self.en4(pool_3)
        trans4 = self.trans4(en4)
        concat4 = torch.cat([trans4,en3],dim=1)
        dc4     = self.dc4(concat4)
        trans3     = self.trans3(dc4)
        concat3 = torch.cat([trans3,en2],dim=1)
        dc3     = self.dc3(concat3)
        trans2     = self.trans2(dc3)
        concat2 = torch.cat([trans2,en1],dim=1)
        dc2     = self.dc2(concat2)
        final   = self.final(dc2)
        
        final = final.permute(0,1,3,4,2)
        
        shape_num = final.shape[0:4]
        final = final.reshape(-1,final.shape[4])
        final = self.fn(final)
        final = final.reshape(shape_num)
        final = self.softmax(final)
        final = torch.log(final)
        
        return final
        
        
class FCN_CRNN(nn.Module):
    def __init__(self, fcn_input_size, fcn_model_name, 
                       crnn_input_size, crnn_model_name, 
                       hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional, pretrained):
        super(FCN_CRNN, self).__init__()
        if fcn_model_name == 'simpleCNN':
            self.fcn = simple_CNN(fcn_input_size, crnn_input_size[1])
        elif fcn_model_name == 'fcn8':
            self.fcn = make_fcn_model(crnn_input_size[1], fcn_input_size[1], freeze=False)
        elif fcn_model_name == 'unet':
            self.fcn = make_UNet_model(crnn_input_size[1], fcn_input_size[1], for_fcn=True, pretrained = pretrained)
        if crnn_model_name == 'clstm': 
            self.crnn = CLSTMSegmenter(crnn_input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes, bidirectional)

    def forward(self, input_tensor):
        batch, timestamps, bands, rows, cols = input_tensor.size()
        fcn_input = input_tensor.view(batch * timestamps, bands, rows, cols)
        fcn_output = self.fcn(fcn_input)
 
        crnn_input = fcn_output.view(batch, timestamps, -1, rows, cols)
        preds = self.crnn(crnn_input)
        return preds

    
class simple_CNN(nn.Module):
    """ Simple CNN model used for testing
    """
    def __init__(self, input_size, fcn_out_feats):
        """ input_size is batch, time_steps, channels, height, width
        """
        super(simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_size[1], fcn_out_feats, 3, padding=1) 
    def forward(self, x):
        h = x.cuda()
        h = self.conv1(h)
        return h


class FCN8s_croptype(nn.Module):
    '''
    FCN implementation from https://github.com/wkentaro/pytorch-fcn/tree/63bc2c5bf02633f08d0847bb2dbd0b2f90034837
    '''
    def __init__(self, n_class=5, n_channel = 11):
        super(FCN8s_croptype, self).__init__()
        # conv1
        self.conv1_1_croptype = nn.Conv2d(n_channel, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr_croptype = nn.Conv2d(4096, n_class, 1)
        self.score_pool3_croptype = nn.Conv2d(256, n_class, 1)
        self.score_pool4_croptype = nn.Conv2d(512, n_class, 1)

        self.upscore2_croptype = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8_croptype = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4_croptype = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
                
    def forward(self, x):
        h = x.cuda()
        h = self.relu1_1(self.conv1_1_croptype(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr_croptype(h)
        h = self.upscore2_croptype(h)
        upscore2 = h  # 1/16

        h = self.score_pool4_croptype(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        
        
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4_croptype(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3_croptype(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8_croptype(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    
class ConvLSTMCell(nn.Module):
    """
        ConvLSTM Cell based on Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
        arXiv: https://arxiv.org/abs/1506.04214

        Implementation based on stefanopini's at https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
    """
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor.cuda(), h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())


class CLSTM(nn.Module):

    def __init__(self, input_size, hidden_dims, kernel_sizes, lstm_num_layers, batch_first=True, bias=True, return_all_layers=False):
        """
           Args:
                input_size - (tuple) should be (time_steps, channels, height, width)
                hidden_dims - (list of ints) number of filters to use per layer
                kernel_sizes - lstm kernel sizes
                lstm_num_layers - (int) number of stacks of ConvLSTM units per step
        """

        super(CLSTM, self).__init__()

        self.height = input_size[2]
        self.width = input_size[3]
        self.start_num_channels = input_size[1]
        self.lstm_num_layers = lstm_num_layers
        self.bias = bias
       
        if isinstance(kernel_sizes, list):
            if len(kernel_sizes) != lstm_num_layers and len(kernel_sizes) == 1:
                self.kernel_sizes = kernel_sizes * lstm_num_layers
            else:
                self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * lstm_num_layers      
        
        if isinstance(hidden_dims, list):
            if len(hidden_dims) != lstm_num_layers and len(hidden_dims) == 1:
                self.hidden_dims = hidden_dims * lstm_num_layers
            else:
                self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = [hidden_dims] * lstm_num_layers       

        cell_list = []
        for i in range(self.lstm_num_layers):
            cur_input_dim = self.start_num_channels if i == 0 else self.hidden_dims[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim = cur_input_dim,
                                          hidden_dim = self.hidden_dims[i],
                                          kernel_size = self.kernel_sizes[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        # figure out what this is doing
        hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.lstm_num_layers):

            h, c = hidden_state[layer_idx]
            output_inner_layers = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])

                output_inner_layers.append(h)

            layer_output = torch.stack(output_inner_layers, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # Just take last output for prediction
        layer_output_list = layer_output_list[-1:]
        last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.lstm_num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


class CLSTMSegmenter(nn.Module):
    """ CLSTM followed by conv for segmentation output
    """

    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, 
                 conv_kernel_size, lstm_num_layers, num_classes, bidirectional):

        super(CLSTMSegmenter, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.clstm = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)
        self.bidirectional = bidirectional
        in_channels = hidden_dims[-1] if not self.bidirectional else hidden_dims[-1] * 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        self.softmax = nn.Softmax2d()

    def forward(self, inputs):
        layer_output_list, last_state_list = self.clstm(inputs)
        final_state = last_state_list[0][0]
        if self.bidirectional:
            rev_inputs = torch.tensor(inputs.cpu().detach().numpy()[::-1].copy(), dtype=torch.float32).cuda()
            rev_layer_output_list, rev_last_state_list = self.clstm(rev_inputs)
            final_state = torch.cat([final_state, rev_last_state_list[0][0]], dim=1)
        scores = self.conv(final_state)
        preds = self.softmax(scores)
        preds = torch.log(preds)

        return preds


def get_num_bands(kwargs):
    num_bands = -1
    added_doy = 0
    added_clouds = 0

    if kwargs.get('include_doy'):
        added_doy = 1
    if kwargs.get('include_clouds') and kwargs.get('use_s2'): 
        added_clouds = 1

    if kwargs.get('use_s1') and kwargs.get('use_s2'):
        num_bands = S1_NUM_BANDS + S2_NUM_BANDS + 2*added_doy + added_clouds
    elif kwargs.get('use_s1'):
        num_bands = S1_NUM_BANDS + added_doy + added_clouds
    elif kwargs.get('use_s2'):
        num_bands = S2_NUM_BANDS + added_doy + added_clouds
    else:
        raise ValueError("S1 / S2 usage not specified in args!")
    return num_bands


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
        model = make_rf_model(random_state=kwargs.get('random_state', None),
                                        n_jobs=kwargs.get('n_jobs', -1),
                                        n_estimators=kwargs.get('n_estimators', 50))

    elif model_name == 'bidir_clstm':
        num_bands = get_num_bands(kwargs)

        # TODO: change the timestamps passed in to be more flexible (i.e allow specify variable length / fixed / truncuate / pad)
        # TODO: don't hardcode values
        model = make_bidir_clstm_model(input_size=(MIN_TIMESTAMPS, num_bands, GRID_SIZE, GRID_SIZE), 
                                       hidden_dims=kwargs.get('hidden_dims'), 
                                       lstm_kernel_sizes=(kwargs.get('crnn_kernel_sizes'), kwargs.get('crnn_kernel_sizes')), 
                                       conv_kernel_size=kwargs.get('conv_kernel_size'), 
                                       lstm_num_layers=kwargs.get('crnn_num_layers'),
                                       num_classes=kwargs.get('num_classes'),
                                       bidirectional=kwargs.get('bidirectional'))
    elif model_name == 'fcn':
        num_bands = get_num_bands(kwargs)
        model = make_fcn_model(n_class=kwargs.get('num_classes'), n_channel = num_bands, freeze=True)
    
    elif model_name == 'unet':
        num_bands = get_num_bands(kwargs)
        
        if kwargs.get('time_slice') is None:
            model = make_UNet_model(n_class=kwargs.get('num_classes'), n_channel = num_bands*MIN_TIMESTAMPS)
        else: 
            model = make_UNet_model(n_class=kwargs.get('num_classes'), n_channel = num_bands)
    
    elif model_name == 'fcn_crnn':
        num_bands = get_num_bands(kwargs)

        model = make_fcn_clstm_model(fcn_input_size=(MIN_TIMESTAMPS, num_bands, GRID_SIZE, GRID_SIZE), 
                                     fcn_model_name=kwargs.get('fcn_model_name'),
                                     crnn_input_size=(MIN_TIMESTAMPS, kwargs.get('fcn_out_feats'), GRID_SIZE, GRID_SIZE),
                                     crnn_model_name=kwargs.get('crnn_model_name'),
                                     hidden_dims=kwargs.get('hidden_dims'), 
                                     lstm_kernel_sizes=(kwargs.get('crnn_kernel_sizes'), kwargs.get('crnn_kernel_sizes')), 
                                     conv_kernel_size=kwargs.get('conv_kernel_size'), 
                                     lstm_num_layers=kwargs.get('crnn_num_layers'), 
                                     num_classes=kwargs.get('num_classes'),
                                     bidirectional=kwargs.get('bidirectional'),                                       
                                     pretrained = kwargs.get('pretrained'))
    elif model_name == 'unet3d':
        num_bands = get_num_bands(kwargs)
        model = make_UNet3D_model(n_class = kwargs.get('num_classes'), n_channel = num_bands)
  
    return model

