import numpy as np
import os
import itertools
import sys

import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras import regularizers
from keras.utils import np_utils, to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Set path so that functions can be imported from the utils script
sys.path.insert(0, '../')
from models.keras_models import make_1d_nn_model, make_1d_cnn_model

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
                    self.X_train = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xtrain_g49.npy')        
                    self.X_val = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xval_g57.npy')        
                    self.X_test = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xtest_g62.npy') 
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)
            elif source == 's2':
                if use_pca:
                    self.X_train = np.load(base_dir + '/small/pca/s2/small_pca_s2_rrrgggbbb_Xtrain_num_PCs19.npy')
                    self.X_val = np.load(base_dir + '/small/pca/s2/small_pca_s2_rrrgggbbb_Xval_num_PCs19.npy')
                    self.X_test = np.load(base_dir + '/small/pca/s2/small_pca_s2_rrrgggbbb_Xtest_num_PCs19.npy')
                else:
                    self.X_train = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xtrain_g49.npy')
                    self.X_val = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xval_g57.npy')
                    self.X_test = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xtest_g62.npy')
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)
            elif source == 's1_s2':
                if use_pca: 
                    self.X_train = np.load(base_dir + '/small/pca/s1_s2/small_pca_s1_s2_rrrgggbbb_Xtrain_num_PCs41.npy') 
                    self.X_val = np.load(base_dir + '/small/pca/s1_s2/small_pca_s1_s2_rrrgggbbb_Xval_num_PCs41.npy')
                    self.X_test = np.load(base_dir + '/small/pca/s1_s2/small_pca_s1_s2_rrrgggbbb_Xtest_num_PCs41.npy')
                else: 
                    # use np.hstack() to combine       
                    s1_train = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xtrain_g49.npy')
                    s2_train = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xtrain_g49.npy')
                    self.X_train = np.hstack((s1_train, s2_train))

                    s1_val = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xval_g57.npy')
                    s2_val = self.X_val = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xval_g57.npy')
                    self.X_val = np.hstack((s1_val, s2_val))

                    s1_test = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_Xtest_g62.npy')
                    s2_test = np.load(base_dir + '/small/raw/s2/small_raw_s2_rrrgggbbb_Xtest_g62.npy')
                    self.X_test = np.hstack((s1_test, s2_test))
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)
    
            self.y_train = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_ytrain_g49.npy')
            self.y_val = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_yval_g57.npy')
            self.y_test = np.load(base_dir + '/small/raw/s1/small_raw_s1_rrrgggbbb_ytest_g62.npy')

        elif dataset_type == 'full':
            if source == 's1': 
                if use_pca:
                    self.X_train = np.load(base_dir + '/full/pca/s1/full_pca_s1_rrrgggbbb_Xtrain_num_PCs23.npy')
                    self.X_val = np.load(base_dir + '/full/pca/s1/full_pca_s1_rrrgggbbb_Xval_num_PCs23.npy')
                    self.X_test = np.load(base_dir + '/full/pca/s1/full_pca_s1_rrrgggbbb_Xtest_num_PCs23.npy')
                else:
                    self.X_train = np.load(base_dir + '/full/raw/s1/full_raw_s1_rrrgggbbb_Xtrain_g2338.npy')
                    self.X_val = np.load(base_dir + '/full/raw/s1/full_raw_s1_rrrgggbbb_Xval_g305.npy')
                    self.X_test = np.load(base_dir + '/full/raw/s1/full_raw_s1_rrrgggbbb_Xtest_g366.npy')
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)
            elif source == 's2':
                if use_pca:
                    self.X_train = np.load(base_dir + '/full/pca/s2/full_pca_s2_rrrgggbbb_Xtrain_num_PCs23.npy')
                    self.X_val = np.load(base_dir + '/full/pca/s2/full_pca_s2_rrrgggbbb_Xval_num_PCs23.npy')
                    self.X_test = np.load(base_dir + '/full/pca/s2/full_pca_s2_rrrgggbbb_Xtest_num_PCs23.npy')
                else:
                    self.X_train = np.load(base_dir + '/full/raw/s2/full_raw_s2_rrrgggbbb_Xtrain_g2338.npy')
                    self.X_val = np.load(base_dir + '/full/raw/s2/full_raw_s2_rrrgggbbb_Xval_g305.npy')
                    self.X_test = np.load(base_dir + '/full/raw/s2/full_raw_s2_rrrgggbbb_Xtest_g366.npy')
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)
            elif source == 's1_s2':
                if use_pca: 
                    self.X_train = np.load(base_dir + '/full/pca/s1_s2/full_pca_s1_s2_rrrgggbbb_Xtrain_num_PCs45.npy')
                    self.X_val = np.load(base_dir + '/full/pca/s1_s2/full_pca_s1_s2_rrrgggbbb_Xval_num_PCs45.npy')
                    self.X_test = np.load(base_dir + '/full/pca/s1_s2/full_pca_s1_s2_rrrgggbbb_Xtest_num_PCs45.npy')
                else:
                    # use np.hstack() to combine raw s1 and s2
                    s1_train = np.load(base_dir + '/full/raw/s1/full_raw_s1_rrrgggbbb_Xtrain_g2338.npy')
                    s2_train = np.load(base_dir + '/full/raw/s2/full_raw_s2_rrrgggbbb_Xtrain_g2338.npy') 
                    self.X_train = np.hstack((s1_train, s2_train))

                    s1_val = np.load(base_dir + '/full/raw/s1/full_raw_s1_rrrgggbbb_Xval_g305.npy')
                    s2_val = np.load(base_dir + '/full/raw/s2/full_raw_s2_rrrgggbbb_Xval_g305.npy')
                    self.X_val = np.hstack((s1_val, s2_val))

                    s1_test = np.load(base_dir + '/full/pca/s1/full_pca_s1_rrrgggbbb_Xtest_num_PCs23.npy')
                    s2_test = np.load(base_dir + '/full/raw/s2/full_raw_s2_rrrgggbbb_Xtest_g366.npy')
                    self.X_test = np.hstack((s1_test, s2_test))
        
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)
 
            self.y_train = np.load(base_dir + '/full/raw/s1/full_raw_s1_rrrgggbbb_ytrain_g2338.npy')
            self.y_val = np.load(base_dir + '/full/raw/s1/full_raw_s1_rrrgggbbb_yval_g305.npy')
            self.y_test = np.load(base_dir + '/full/raw/s1/full_raw_s1_rrrgggbbb_ytest_g366.npy')

        elif dataset_type == 'dummy:':

            self.X_train = np.ones((10, 30))
            self.y_train = np.ones((10,5))

            self.X_val = np.ones((4, 30))
            self.y_val = np.ones((4,5))

            self.X_test = np.ones((2, 30))
            self.y_test = np.ones((2,5))
    
  
        self.X_train = np.expand_dims(self.X_train, axis=2)
        self.X_val = np.expand_dims(self.X_val, axis=2)
        self.X_test = np.expand_dims(self.X_test, axis=2)

        self.y_train = to_categorical(self.y_train.astype(int)-1,num_classes=5)
        self.y_val = to_categorical(self.y_val.astype(int)-1,num_classes=5)
        self.y_test = to_categorical(self.y_test.astype(int)-1,num_classes=5)

        #print('X train: ', self.X_train.shape, self.X_train)
        #print('X val: ', self.X_val.shape)
        #print('X test: ', self.X_test.shape)
        #print('y train: ', self.y_train.shape)
        #print('y val: ', self.y_val.shape)
        #print('y test: ', self.y_test.shape)

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
            pred = np.argmax(self.model.predict(self.X_train), axis=1)
            cm = confusion_matrix(np.argmax(self.y_train, axis=1), pred)
        elif data_split == 'val': 
            score = self.model.evaluate(self.X_val, self.y_val)
            pred = np.argmax(self.model.predict(self.X_val), axis=1)
            cm = confusion_matrix(np.argmax(self.y_val, axis=1), pred)
        elif data_split == 'test': 
            score = self.model.evaluate(self.X_val, self.y_val)
            pred = np.argmax(self.model.predict(self.X_test), axis=1)
            cm = confusion_matrix(np.argmax(self.y_test, axis=1), pred)
        print('%s: %.2f%%' % (self.model.metrics_names[1], score[1]*100))
        print('Confusion Matrix: ', cm) 


    def fit(self):
        """ Trains the model
        """
        # Compile model
        self.model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
        # Fit model
        self.model.fit(self.X_train, 
                       self.y_train, 
                       batch_size=500,
                       epochs=10,
                       validation_data=(self.X_val, self.y_val),
                       verbose=1)


def main():

    for dataset_type in ['small']:
        for use_pca in [1]:
            for source in ['s1_s2']:

                print('---------------------------------------')
                print('CNN model, {} dataset, {} pca, {} source'.format(dataset_type, use_pca, source))

                # Define NN model
                keras_1d_NN = DL_model()
                # Load data into model
                keras_1d_NN.load_data(dataset_type, use_pca, source)
                # Define model 
                keras_1d_NN.model = make_1d_nn_model(num_classes=5, 
                         num_input_feats=keras_1d_NN.X_train.shape[1])
                # Fit model
                keras_1d_NN.fit()
                # Evaluate
                print('evaluate train: ')
                keras_1d_NN.evaluate('train')
                print('evaluate val: ')
                keras_1d_NN.evaluate('val')
                print('evaluate test: ')
                keras_1d_NN.evaluate('test')
    

                # Define CNN model
                #keras_1d_CNN = DL_model()
                # Load data into model
                #keras_1d_CNN.load_data(dataset_type, use_pca, source)
                # Define model 
                #keras_1d_CNN.model = make_1d_cnn_model(num_classes=5,
                #           num_input_feats=keras_1d_CNN.X_train.shape[1])
                # Fit model
                #keras_1d_CNN.fit()
                # Evaluate
                #print('evaluate train: ')
                #keras_1d_CNN.evaluate('train')
                #print('evaluate val: ')
                #keras_1d_CNN.evaluate('val')
                #print('evaluate test: ')
                #keras_1d_CNN.evaluate('test')

if __name__ == '__main__':
   main()
