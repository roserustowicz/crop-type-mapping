"""

File housing all models.

Each model is created by invoking the appropriate function
given by:

    make_MODELNAME_model(MODEL_SETTINGS)

"""

from keras.models import Sequential
from keras.layers import InputLayer, Activation, BatchNormalization, Flatten, Dropout
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Bidirectional, TimeDistributed, concatenate
from keras.backend import reverse

def make_1d_nn_model(num_classes, num_input_feats):
    """ Defines a keras Sequential 1D NN model 
    
    Args: 
      num_classes - (int) number of classes to predict 
    Returns: 
      loads self.model as the defined model
    """
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu', input_shape=(num_input_feats, 1)))
    model.add(BatchNormalization())
    model.add(Dense(units=32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    return model

def make_1d_cnn_model(num_classes, num_input_feats):
    """ Defines a keras Sequential 1D CNN model 
    
    Args: 
      num_classes - (int) number of classes to predict 
    Returns: 
      loads self.model as the defined model
    """
    model = Sequential()

    model.add(Conv1D(32, kernel_size=3,
              strides=1, activation='relu', padding='same',
              input_shape=(num_input_feats, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(32, kernel_size=3,
              strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, kernel_size=3,
              strides=1, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same',))     
    model.add(BatchNormalization())
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same',))     
    model.add(BatchNormalization())
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same',))     
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same',))
    model.add(BatchNormalization())
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same',))
    model.add(BatchNormalization())
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same',))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def make_bidirectional_clstm_model():
    """
    """
    fwd_seq = Input(shape=(X.shape[1:])) # bands, rows, cols, time
    rev_seq = Input(shape=(X.shape[1:])) # bands, rows, cols, time
    
    shared_CLSTM = ConvLSTM2D(filters=256,
                              kernel_size=3,
                              padding='same',
                              activation='relu')

    fwd_features = shared_CLSTM(fwd_seq)
    rev_features = shared_CLSTM(rev_seq)

    concat_feats = concatenate([fwd_features, rev_features], axis=0) # change axis 

    predictions = Conv2D(filters=num_crops, 
                         kernel_size=3, 
                         padding='same', 
                         activation='softmax')

    model = Model(inputs=[fwd_seq, reverse(fwd_seq, axes=0)], # change axes 
                  outputs=predictions)

    return model

