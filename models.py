"""

File housing all models.

Each model is created by invoking the appropriate function
given by:

    make_MODELNAME_model(MODEL_SETTINGS)

"""

import torch 
import torch.nn as nn


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from constants import *

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

def make_1d_nn_model(num_classes, num_input_feats, units, reg_strength):
    """ Defines a keras Sequential 1D NN model

    Args:
      num_classes - (int) number of classes to predict
    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(units=units, kernel_regularizer=reg, 
              bias_regularizer=reg, input_shape=(num_input_feats, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model

def make_1d_2layer_nn_model(num_classes, num_input_feats, units, reg_strength):
    """ Defines a keras Sequential 1D NN model

    Args:
      num_classes - (int) number of classes to predict
    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(units=units, kernel_regularizer=reg, 
              bias_regularizer=reg, input_shape=(num_input_feats, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=units, kernel_regularizer=reg,
              bias_regularizer=reg))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model

def make_1d_cnn_model(num_classes, num_input_feats, units, reg_strength):
    """ Defines a keras Sequential 1D CNN model

    Args:
      num_classes - (int) number of classes to predict
    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)
    
    model = Sequential()

    model.add(Conv1D(units, kernel_size=11,
              strides=11, padding='same',
              kernel_regularizer=reg,
              bias_regularizer=reg,
              input_shape=(num_input_feats, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(units*2, kernel_size=3, padding='same',
              kernel_regularizer=reg, bias_regularizer=reg))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(units*4, activation='relu', 
              kernel_regularizer=reg, bias_regularizer=reg))
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model


def make_bidir_clstm_model(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes):

    clstm_segmenter = CLSTMSegmenter(input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes)

    return clstm_segmenter


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
                kernel_size -
                lstm_num_layers - (int) number of stacks of ConvLSTM units per step
        """

        super(CLSTM, self).__init__()

        self.height = input_size[2]
        self.width = input_size[3]
        self.start_num_channels = input_size[1]
        self.lstm_num_layers = lstm_num_layers
        self.bias = bias
       
        if isinstance(kernel_sizes, list):
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes]        
        
        if isinstance(hidden_dims, list):
            self.hidden_dims = hidden_dims
        else:
            self.hidden_dims = [hidden_dims]        

        cell_list = []
        for i in range(self.lstm_num_layers):
            cur_input_dim = self.start_num_channels if i == 0 else hidden_dims[i-1]

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

    def __init__(self, input_size, hidden_dims, lstm_kernel_sizes, conv_kernel_size, lstm_num_layers, num_classes):

        super(CLSTMSegmenter, self).__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]        

        self.clstm = CLSTM(input_size, hidden_dims, lstm_kernel_sizes, lstm_num_layers)

        self.conv = nn.Conv2d(in_channels=hidden_dims[-1], out_channels=num_classes, kernel_size=conv_kernel_size, padding=int((conv_kernel_size - 1) / 2))
        self.softmax = nn.Softmax2d()

    def forward(self, inputs):
        layer_output_list, last_state_list = self.clstm(inputs)
        preds = self.softmax(self.conv(last_state_list[0][0]))
        return preds

def get_model(model_name, **kwargs):
    model = None
    if model_name == 'random_forest':
        model = make_rf_model(random_state=kwargs.get('random_state', None),
                                        n_jobs=kwargs.get('n_jobs', -1),
                                        n_estimators=kwargs.get('n_estimators', 50))

    if model_name == 'bidir_clstm':
        num_bands = -1
        if kwargs.get('use_s1') and kwargs.get('use_s2'):
            num_bands = S1_NUM_BANDS + S2_NUM_BANDS
        elif kwargs.get('use_s1'):
            num_bands = S1_NUM_BANDS
        elif kwargs.get('use_s2'):
            num_bands = S2_NUM_BANDS
        else:
            raise ValueError("S1 / S2 usage not specified in args!")
        
        # TODO: change the timestamps passed in to be more flexible (i.e allow specify variable length / fixed / truncuate / pad)
        # TODO: don't hardcode values
        model = make_bidir_clstm_model(input_size=(MIN_TIMESTAMPS, num_bands, GRID_SIZE, GRID_SIZE), 
                                       hidden_dims=kwargs.get('hidden_dims'), 
                                       lstm_kernel_sizes=(kwargs.get('crnn_kernel_sizes'), kwargs.get('crnn_kernel_sizes')), 
                                       conv_kernel_size=kwargs.get('conv_kernel_size'), 
                                       lstm_num_layers=kwargs.get('crnn_num_layers'),
                                       num_classes=kwargs.get('num_classes'))
    return model
