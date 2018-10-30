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
    #model.add(Dense(units=32, activation='relu', kernel_regularizer=reg,
    #          bias_regularizer=reg))
    #model.add(BatchNormalization())
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
              strides=1, padding='same',
              kernel_regularizer=reg,
              bias_regularizer=reg,
              input_shape=(num_input_feats, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv1D(32, kernel_size=3,
    #          strides=1, activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(Conv1D(32, kernel_size=3,
    #          strides=1, activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(units*2, kernel_size=3, padding='same',
              kernel_regularizer=reg, bias_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same',))
    #model.add(BatchNormalization())
    #model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same',))
    #model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    #model.add(Conv1D(128, kernel_size=3, padding='same',
    #          kernel_regularizer=reg, bias_regularizer=reg))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same',))
    #model.add(BatchNormalization())
    #model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same',))
    #model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(units*4, activation='relu', 
              kernel_regularizer=reg, bias_regularizer=reg))
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model

def make_bidir_clstm_model(data_shape, num_crops=5):
    """
    model = Sequential()
    model.add(ConvLSTM2D(filters=256,
                         kernel_size=3,
                         padding='same',
                         activation='relu',
                         data_format='channels_first',
                         input_shape=data_shape))
    model.add(Conv2D(filters=num_crops,
                     kernel_size=3,
                     padding='same',
                     activation='softmax',
                     data_format='channels_first'))
    return model
    """
    input_ = Input(shape=data_shape) # time, bands, rows, cols
    shared_CLSTM = ConvLSTM2D(filters=256,
                                            kernel_size=3,
                                            padding='same',
                                            activation='relu',
                                            data_format='channels_first')

    features = shared_CLSTM(input_)

    predictions = Conv2D(filters=num_crops,
                         kernel_size=3,
                         padding='same',
                         activation='softmax',
                         data_format='channels_first')(features)


    model = Model(inputs=input_, outputs=predictions)
    return model

def get_model(model_name, **kwargs):
    model = None
    if model_name == 'random_forest':
        model = make_rf_model(random_state=kwargs.get('random_state', None),
                                        n_jobs=kwargs.get('n_jobs', -1),
                                        n_estimators=kwargs.get('n_estimators', 50))

    # TODO: don't make hard coded shape
    if model_name == 'bidir_clstm':
        model = make_bidir_clstm_model(data_shape=(None, S1_NUM_BANDS, 64, 64))

    return model
