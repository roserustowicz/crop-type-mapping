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


def make_bidir_clstm_model(num_bands, num_crops=5):
    
    pass

class Bidirectional_CLSTM(nn.Module):

    def forward(self, padded_input, input_lengths):
        max_length = input_lengths[0] # necessary for data parallelism; see https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        packed_input = nn.utils.rnn.pack_padded_sequence(padded_input, input_lengths, batch_first=True)
        packed_output, _ = self.conv_lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_length)

def get_model(model_name, **kwargs):
    model = None
    if model_name == 'random_forest':
        model = make_rf_model(random_state=kwargs.get('random_state', None),
                                        n_jobs=kwargs.get('n_jobs', -1),
                                        n_estimators=kwargs.get('n_estimators', 50))

    # TODO: don't make hard coded shape
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

        model = make_bidir_clstm_model(data_shape=(None, num_bands, GRID_SIZE, GRID_SIZE))

    return model
