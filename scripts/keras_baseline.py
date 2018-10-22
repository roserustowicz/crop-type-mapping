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
    """Class for a keras deep learning model"""
    def __init__(self):
        self.model = None

    def load_data(self, dataset_type, use_pca, source):
        """ Load .npy files for train, val, test splits
        
        X --> expand_dims along axis 2
        y --> should be one-hot encoded

        Args: 
          dataset_type - (str) 'full' or 'small' indicates dataset to use
          use_pca - (boolean) whether or not to use PCA features. if not, 
                    raw features (times x bands) are used
          source - (str) 's1' or 's2' indicates the data source to use
        """

        base_dir = '/home/data/pixel_arrays'
        if dataset_type == 'small':
            if source == 's1':
                if use_pca:
                    self.X_train = np.load(base_dir + '/small/pca/s1/small_pca_s1_rrrgggbbb_Xtrain_num_PCs23.npy')
                    self.X_val = np.load(base_dir + '/small/pca/s1/small_pca_s1_rrrgggbbb_Xval_num_PCs23.npy')
                    self.X_test = np.load(base_dir + '/small/pca/s1/small_pca_s1_rrrgggbbb_Xtest_num_PCs23.npy')
                else:
                    self.X_train = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xtrain_g72.npy')        
                    self.X_val = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xval_g66.npy')        
                    self.X_test = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xtest_g85.npy') 
       
                self.y_train = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_ytrain_g72.npy')
                self.y_val = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_yval_g66.npy')
                self.y_test = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_ytest_g85.npy')
            elif source == 's2':
                if use_pca:
                    self.X_train = np.load(base_dir + '/small/pca/s2/small_pca_s2_rrrgggbbb_Xtrain_num_PCs27.npy')
                    self.X_val = np.load(base_dir + '/small/pca/s2/small_pca_s2_rrrgggbbb_Xval_num_PCs27.npy')
                    self.X_test = np.load(base_dir + '/small/pca/s2/small_pca_s2_rrrgggbbb_Xtest_num_PCs27.npy')
                else:
                    self.X_train = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xtrain_g72.npy')
                    self.X_val = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xval_g66.npy')
                    self.X_test = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xtest_g85.npy')

                self.y_train = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_ytrain_g72.npy')
                self.y_val = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_yval_g66.npy')
                self.y_test = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_ytest_g85.npy')
      
        elif dataset_type == 'full':
            if source == 's1': 
                if use_pca:
                    pass
                else:
                    pass
            elif source == 's2':
                if use_pca:
                    pass
                else:
                    pass
        elif dataset_type == 'dummy:':

            self.X_train = np.expand_dims(np.ones((10, 30)), axis=2)
            self.y_train = y_train = np.ones((10,5))

            self.X_val = np.expand_dims(np.ones((4, 30)), axis=2)
            self.y_val = np.ones((4,5))

            self.X_test = np.expand_dims(np.ones((2, 30)), axis=2)
            self.y_test = np.ones((2,5))

    def load_from_json(self, json_fname, h5_fname):
        """
        Loads a pre-trained model from json, h5 files

        Args: 
          json_fname - (str) the path and filename of the '.json'
                        file associated with a pre-trained model 
          h5_fname - (str) the path and filename of the '.h5'
                      file associated with pre-trained model weights
        Returns: 
          loads self.model as the model loaded in from the input files
        """
        json_file = open(json_fname, 'r')
        json_model = json_file.read()
        json_file.close()

        loaded_model = model_from_json(json_model)
        self.model = loaded_model.load_weights(h5_fname)

    def evaluate(self, data_split):
        """ Evaluate the model accuracy
  
        Args: 
          data_split - (str) the data split to use for evaluation 
                        of options 'train', 'val', or 'test'
        Returns: 
          prints the accuracy score
        """
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
        """ Defines a keras Sequential 1D CNN model 
    
        Args: 
          num_classes - (int) number of classes to predict 
        Returns: 
          loads self.model as the defined model
        """
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
        """ Defines a keras Sequential 1D NN model 
    
        Args: 
          num_classes - (int) number of classes to predict 
        Returns: 
          loads self.model as the defined model
        """
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
        """ Trains the model
        """
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
