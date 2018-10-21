import numpy as np
import os
import itertools

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation, BatchNormalization, Flatten, Dropout
from keras.layers import Dense, Conv1D, MaxPooling1D

from keras import regularizers
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

np.random.seed(10)

class DL_model:
    """A model class"""
    def __init__(self):
        self.model = None

    def load_data(self):
        """ Load .npy files for train, val, test splits
        
        X --> expand_dims along axis 2
        y --> should be one-hot encoded
        """
        # Loading in dummy data right now
        self.X_train = np.expand_dims(np.ones((10, 30)), axis=2)
        self.y_train = y_train = np.ones((10,5))

        self.X_val = np.expand_dims(np.ones((4, 30)), axis=2)
        self.y_val = np.ones((4,5))

        self.X_test = np.expand_dims(np.ones((2, 30)), axis=2)
        self.y_test = np.ones((2,5))

    def load_from_json(self, json_fname, h5_fname):
        json_file = open(json_fname, 'r')
        json_model = json_file.read()
        json_file.close()

        loaded_model = model_from_json(json_model)
        self.model = loaded_model.load_weights(h5_fname)

    def evaluate(self, data_split):
        self.model.compile(loss = 'categorical_crossentropy', 
                          optimizer='adam', 
                          metrics=['accuracy'])
        
        if data_split == 'train': 
            score = self.model.evaluate(self.X_train, self.y_train)
        elif data_split == 'val': 
            score = self.model.evaluate(self.X_val, self.y_val)
        elif data_split == 'test': 
            score = self.model.evaluate(self.X_val, self.y_val)
        print('%s: %.2f%%' % (self.model.metrics_names[1], score[1]*100))

    def create_cnn(self, num_classes):
        model = Sequential()
  
        model.add(Conv1D(32, kernel_size=5, 
                  strides=1, activation='relu', 
                  input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2, strides=2))

        model.add(Conv1D(64, 5, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2, strides=2))

        model.add(Conv1D(128, 5, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', 
                      metrics = ['accuracy'])

        self.model = model

    def create_nn(self, num_classes):
        model = Sequential()
  
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu', input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', 
                      metrics = ['accuracy'])

        self.model = model

    def fit(self):
        self.model.fit(self.X_train, 
                       self.y_train, 
                       batch_size=2,
                       epochs=1,
                       validation_data=(self.X_val, self.y_val),
                       verbose=1)


def main():
    # Define NN model
    keras_1d_NN = DL_model()
    # Load data into model
    keras_1d_NN.load_data()
    # Define model 
    keras_1d_NN.create_nn(num_classes=5)
    # Fit model
    keras_1d_NN.fit()
    # Evaluate
    keras_1d_NN.evaluate('train')
    keras_1d_NN.evaluate('val')
    

    # Define CNN model
    keras_1d_CNN = DL_model()
    # Load data into model
    keras_1d_CNN.load_data()
    # Define model 
    keras_1d_CNN.create_nn(num_classes=5)
    # Fit model
    keras_1d_CNN.fit()
    # Evaluate
    keras_1d_CNN.evaluate('train')
    keras_1d_CNN.evaluate('val')

if __name__ == '__main__':
   main()
