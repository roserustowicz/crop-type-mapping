import numpy as np
import os
import itertools
import sys

import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras import regularizers
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Set path so that functions can be imported from the utils script
sys.path.insert(0, '../')
from models import make_1d_nn_model, make_1d_cnn_model

np.random.seed(10)

class DL_model:
    """Class for a keras deep learning model"""
    def __init__(self):
        self.model = None

    def load_data(self, dataset_type, use_pca, source, ordering, reverse_clouds, verbose, full_sampled, reshape_bands):
        """ Load .npy files for train, val, test splits
        
        X --> expand_dims along axis 2
        y --> should be one-hot encoded

        Args: 
          dataset_type - (str) 'full' or 'small' indicates dataset to use
          use_pca - (boolean) whether or not to use PCA features. if not, 
                    raw features (times x bands) are used
          source - (str) 's1' or 's2' indicates the data source to use
        """

        base_dir = '/home/data/ghana/pixel_arrays'
        if dataset_type == 'small':
            if source == 's1':
                if use_pca:
                    pass
                else:
                    if ordering == 'byband':
                        self.X_train = np.load(base_dir + '/small/raw/s1/small_raw_s1_byband_Xtrain_g49.npy')        
                        self.X_val = np.load(base_dir + '/small/raw/s1/small_raw_s1_byband_Xval_g57.npy')        
                        self.X_test = np.load(base_dir + '/small/raw/s1/small_raw_s1_byband_Xtest_g62.npy') 
                    elif ordering == 'bytime':
                        self.X_train = np.load(base_dir + '/small/raw/s1/small_raw_s1_bytime_Xtrain_g49.npy')        
                        self.X_val = np.load(base_dir + '/small/raw/s1/small_raw_s1_bytime_Xval_g57.npy')        
                        self.X_test = np.load(base_dir + '/small/raw/s1/small_raw_s1_bytime_Xtest_g62.npy') 
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)

            elif source == 's2':
                if use_pca:
                    pass
                else:
                    if reverse_clouds:
                        if ordering == 'bytime':
                            self.X_train = np.load(base_dir + 
                             '/small/raw/cloud_s2/reverse_true/small_raw_s2_cloud_mask_reverseTrue_bytime_Xtrain_g66.npy')
                            self.X_val = np.load(base_dir + 
                             '/small/raw/cloud_s2/reverse_true/small_raw_s2_cloud_mask_reverseTrue_bytime_Xval_g82.npy')
                            self.X_test = np.load(base_dir + 
                             '/small/raw/cloud_s2/reverse_true/small_raw_s2_cloud_mask_reverseTrue_bytime_Xtest_g80.npy')
                    else:
                        if ordering == 'bytime':
                            self.X_train = np.load(base_dir + 
                             '/small/raw/cloud_s2/reverse_false/small_raw_s2_cloud_mask_reverseFalse_bytime_Xtrain_g66.npy')
                            self.X_val = np.load(base_dir + 
                             '/small/raw/cloud_s2/reverse_false/small_raw_s2_cloud_mask_reverseFalse_bytime_Xval_g82.npy')
                            self.X_test = np.load(base_dir + 
                             '/small/raw/cloud_s2/reverse_false/small_raw_s2_cloud_mask_reverseFalse_bytime_Xtest_g80.npy')
        
                    self.y_train = np.load(base_dir + 
                         '/small/raw/cloud_s2/reverse_false/small_raw_s2_cloud_mask_reverseFalse_bytime_ytrain_g66.npy')
                    self.y_val = np.load(base_dir + 
                         '/small/raw/cloud_s2/reverse_false/small_raw_s2_cloud_mask_reverseFalse_bytime_yval_g82.npy')
                    self.y_test = np.load(base_dir + 
                         '/small/raw/cloud_s2/reverse_false/small_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g80.npy')
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)

            elif source == 's1_s2':
                if use_pca:
                    pass 
                else: 
                    if ordering == 'bytime':
                        # use np.hstack() to combine       
                        s1_train = np.load(base_dir + '/small/raw/s1/small_raw_s1_bytime_Xtrain_g49.npy')
                        s2_train = np.load(base_dir + '/small/raw/s2/small_raw_s2_bytime_Xtrain_g49.npy')
                        self.X_train = np.hstack((s1_train, s2_train))

                        s1_val = np.load(base_dir + '/small/raw/s1/small_raw_s1_bytime_Xval_g57.npy')
                        s2_val = self.X_val = np.load(base_dir + '/small/raw/s2/small_raw_s2_bytime_Xval_g57.npy')
                        self.X_val = np.hstack((s1_val, s2_val))

                        s1_test = np.load(base_dir + '/small/raw/s1/small_raw_s1_bytime_Xtest_g62.npy')
                        s2_test = np.load(base_dir + '/small/raw/s2/small_raw_s2_bytime_Xtest_g62.npy')
                        self.X_test = np.hstack((s1_test, s2_test))
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)

        elif dataset_type == 'full':
            if source == 's1': 
                if use_pca:
                    pass
                elif full_sampled:
                    self.X_train = np.load(base_dir + 
                        '/full_balanced/raw/s1_sample/sampled/full_raw_s1_sample_bytime_Xtrain_g2321.npy')
                    #self.X_val = np.load(base_dir +
                    #    '/full_balanced/raw/s1_sample/sampled/full_raw_s1_sample_bytime_Xval_g305.npy')
                    #self.X_test = np.load(base_dir + 
                    #    '/full_balanced/raw/s1_sample/sampled/full_raw_s1_sample_bytime_Xtest_g364.npy')
                    self.y_train = np.load(base_dir + 
                        '/full_balanced/raw/s1_sample/sampled/full_raw_s1_sample_bytime_ytrain_g2321.npy')
                    #self.y_val = np.load(base_dir +
                    #    '/full_balanced/raw/s1_sample/sampled/full_raw_s1_sample_bytime_yval_g305.npy')
                    #self.y_test = np.load(base_dir + 
                    #    '/full_balanced/raw/s1_sample/sampled/full_raw_s1_sample_bytime_ytest_g364.npy')
                    self.X_val = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_Xval_g305.npy')
                    self.X_test = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_Xtest_g364.npy')
                    self.y_val = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_yval_g305.npy')
                    self.y_test = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_ytest_g364.npy')
                else:
                    if ordering == 'bytime':
                        self.X_train = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_Xtrain_g2321.npy')
                        self.X_val = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_Xval_g305.npy')
                        self.X_test = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_Xtest_g364.npy')
                            
                    self.y_train = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_ytrain_g2321.npy')
                    self.y_val = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_yval_g305.npy')
                    self.y_test = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_ytest_g364.npy')
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)

            elif source == 's2':
                if use_pca:
                    pass
                elif full_sampled:
                    self.X_train = np.load(base_dir + 
                        '/full_balanced/raw/s2_cloudsample/sampled/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtrain_g2321.npy')
                    self.X_val = np.load(base_dir +
                        '/full_balanced/raw/s2_cloudsample/sampled/full_raw_s2_cloud_mask_reverseFalse_bytime_Xval_g305.npy')
                    self.X_test = np.load(base_dir + 
                        '/full_balanced/raw/s2_cloudsample/sampled/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtest_g364.npy')
                    self.y_train = np.load(base_dir + 
                        '/full_balanced/raw/s2_cloudsample/sampled/full_raw_s2_cloud_mask_reverseFalse_bytime_ytrain_g2321.npy')
                    self.y_val = np.load(base_dir +
                        '/full_balanced/raw/s2_cloudsample/sampled/full_raw_s2_cloud_mask_reverseFalse_bytime_yval_g305.npy')
                    self.y_test = np.load(base_dir + 
                        '/full_balanced/raw/s2_cloudsample/sampled/full_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g364.npy')
                    #self.X_val = np.load(base_dir + 
                    #  '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_Xval_g305.npy')
                    #self.X_test = np.load(base_dir + 
                    #  '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtest_g364.npy')
                    #self.y_val = np.load(base_dir + 
                    #  '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_yval_g305.npy')
                    #self.y_test = np.load(base_dir + 
                    #  '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g364.npy')
                else:
                    if reverse_clouds:
                        if ordering == 'bytime':
                            self.X_train = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_true/full_raw_s2_cloud_mask_reverseTrue_bytime_Xtrain_g2321.npy')
                            self.X_val = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_true/full_raw_s2_cloud_mask_reverseTrue_bytime_Xval_g305.npy')
                            self.X_test = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_true/full_raw_s2_cloud_mask_reverseTrue_bytime_Xtest_g364.npy')
                            self.y_train = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_ytrain_g2321.npy')
                            self.y_val = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_yval_g305.npy')
                            self.y_test = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g364.npy')
                    else:
                        if ordering == 'bytime':
                            self.X_train = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtrain_g2321.npy')
                            self.X_val = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_Xval_g305.npy')
                            self.X_test = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtest_g364.npy')
                            self.y_train = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_ytrain_g2321.npy')
                            self.y_val = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_yval_g305.npy')
                            self.y_test = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g364.npy')

                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)

            elif source == 's1_s2':
                if use_pca:
                    pass 
                else:
                    # use np.hstack() to combine raw s1 and s2
                    if reverse_clouds:  
                        if ordering == 'bytime':                   
                            s2_train = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_true/full_raw_s2_cloud_mask_reverseTrue_bytime_Xtrain_g2321.npy')
                            s2_val = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_true/full_raw_s2_cloud_mask_reverseTrue_bytime_Xval_g305.npy')
                            s2_test = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_true/full_raw_s2_cloud_mask_reverseTrue_bytime_Xtest_g364.npy')
                    else:
                        if ordering == 'bytime':
                            s2_train = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtrain_g2321.npy')
                            s2_val = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_Xval_g305.npy')
                            s2_test = np.load(base_dir + 
                              '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtest_g364.npy')

                    if ordering == 'bytime':
                        s1_train = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_Xtrain_g2321.npy')
                        s1_val = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_Xval_g305.npy')
                        s1_test = np.load(base_dir + '/full/raw/sample_s1/full_raw_s1_sample_bytime_Xtest_g364.npy')
                   
                    self.X_train = np.hstack((s1_train, s2_train))
                    self.X_val = np.hstack((s1_val, s2_val))
                    self.X_test = np.hstack((s1_test, s2_test))
        
                    self.y_train = np.load(base_dir + 
                      '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_ytrain_g2321.npy')
                    self.y_val = np.load(base_dir + 
                      '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_yval_g305.npy')
                    self.y_test = np.load(base_dir + 
                      '/full/raw/cloud_s2/reverse_false/full_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g364.npy')
                    
                    # Normalize by standard scalar
                    scaler = StandardScaler()
                    scaler.fit(self.X_train)
                    self.X_train = scaler.transform(self.X_train)
                    self.X_val = scaler.transform(self.X_val)
                    self.X_test = scaler.transform(self.X_test)
 

        elif dataset_type == 'dummy:':

            self.X_train = np.ones((10, 30))
            self.y_train = np.ones((10,5))

            self.X_val = np.ones((4, 30))
            self.y_val = np.ones((4,5))

            self.X_test = np.ones((2, 30))
            self.y_test = np.ones((2,5))
    
        if reshape_bands:
            if 's1' in source: num_bands = 2
            elif 's2' in source: num_bands = 11
            self.X_train = reshape_channels(self.X_train, num_bands, ordering)
            self.X_val = reshape_channels(self.X_val, num_bands, ordering)
            self.X_test = reshape_channels(self.X_test, num_bands, ordering)
        else:
            self.X_train = np.expand_dims(self.X_train, axis=2)
            self.X_val = np.expand_dims(self.X_val, axis=2)
            self.X_test = np.expand_dims(self.X_test, axis=2)

        self.y_train = to_categorical(self.y_train.astype(int)-1,num_classes=5)
        self.y_val = to_categorical(self.y_val.astype(int)-1,num_classes=5)
        self.y_test = to_categorical(self.y_test.astype(int)-1,num_classes=5)

        if verbose:
            print('X train: ', self.X_train.shape) #, self.X_train)
            print('X val: ', self.X_val.shape)
            print('X test: ', self.X_test.shape)
            print('y train: ', self.y_train.shape)
            print('y val: ', self.y_val.shape)
            print('y test: ', self.y_test.shape)

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

    def evaluate(self, data_split, f, verbose):
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
            score = self.model.evaluate(self.X_test, self.y_test)
            pred = np.argmax(self.model.predict(self.X_test), axis=1)
            cm = confusion_matrix(np.argmax(self.y_test, axis=1), pred)

        if f:
            f.write('%s: %.2f%% \n' % (self.model.metrics_names[1], score[1]*100))
            f.write('Confusion Matrix:\n {}\n'.format(cm)) 
        if verbose:
            print('%s: %.2f%%' % (self.model.metrics_names[1], score[1]*100))
            print('Confusion Matrix: ', cm) 

    def fit(self):
        """ Trains the model
        """
        # Compile model
        self.model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
        
        #callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto', 
        #             baseline=None, restore_best_weights=True), ModelCheckpoint(filepath='best_model.h5',
        #             monitor='val_loss', save_best_only=True)]
       
        # Fit model
        history = self.model.fit(self.X_train, 
                       self.y_train, 
                       batch_size=500,
                       epochs=200,
                       validation_data=(self.X_val, self.y_val),
                       verbose=1) #,
                       #callbacks=callbacks)
        return history

def reshape_channels(array, num_bands, ordering):
    bs = []
    if ordering == 'bytime':
        for b in range(num_bands):
            bs.append(array[:, b::num_bands])
    elif ordering == 'byband':
        assert array.shape[1] % num_bands == 0
        crop_every = int(array.shape[1]/num_bands)
        for b in range(num_bands):
            bs.append(vec[:, b*crop_every:(b+1)*crop_every])

    return np.dstack(bs)

def plot(history, model_type, dataset_type, source, ordering, units, reg_strength):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    fname = 'acc_model{}-data{}{}-ordering{}-units{}-reg{}'.format(
               model_type, dataset_type, source, ordering, units, reg_strength)
    plt.title(fname)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/'+fname+'.jpg')
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    fname = 'loss_model{}-data{}{}-ordering{}-units{}-reg{}'.format(
               model_type, dataset_type, source, ordering, units, reg_strength)
    plt.title(fname)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/'+fname+'.jpg')

def main():

    #filename = 'NN_full_sampled_reshapebands_valtestoriginal_results.txt'
    filename = 'CNN_full_sampled_reshapebands_nostop_results.txt '
    model_type = 'cnn'
    dataset_type = 'full'
    use_pca = 0
    ordering = 'bytime'
    reverse_clouds = 0
    verbose = 1
    full_sampled = 1   
    reshape_bands = 1 

    for source in ['s2']:

        f = open(filename,'a+')
            
        f.write('--------------------------------------- \n')
        f.write('{} model, {} dataset, {} pca, {} source, {} ordering, {} clouds_reverse \n'.format(
          model_type, dataset_type, use_pca, source, ordering, reverse_clouds))

        f.close()

        if verbose:
            print('--------------------------------------- \n')
            print('{} model, {} dataset, {} pca, {} source, {} ordering, {} clouds_reverse \n'.format(
                model_type, dataset_type, use_pca, source, ordering, reverse_clouds))

        #units = [32, 64, 128, 256]
        #reg_strength = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        #max_dropout=0.6
        ##p_units = random.sample(range(0, len(), 3)
        #for units, reg_strength, dropout in zip([32, 64, 128, 256], [0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5], np.random.random((5,))*0.6)
        dropout=0
        #for units in [32, 64, 128, 256]:
        for units in [8, 16, 32, 64, 128, 256, 512, 1024]:
            for reg_strength in [0]: #0.01, 0.03, 0.05, 0.1, 0.3, 0.5]:
                f = open(filename, 'a+')

                f.write('--------- \n')
                f.write('{} units, {} regularization \n'.format(units, reg_strength))
            
                if verbose:
                    print('--------- \n')
                    print('{} units, {} regularization \n'.format(units, reg_strength))
 
                # Define NN model
                keras_model = DL_model()
                # Load data into model
                keras_model.load_data(dataset_type, use_pca, source, ordering, reverse_clouds, verbose, full_sampled, reshape_bands)
    
                # Define model 
                if model_type == 'nn':
                    keras_model.model = make_1d_nn_model(num_classes=5, 
                                         num_input_feats=keras_model.X_train.shape[1],
                                         units=units,reg_strength=reg_strength,
                                         input_bands=keras_model.X_train.shape[2],
                                         dropout=dropout)
                                         
                elif model_type == 'cnn':
                    keras_model.model = make_1d_cnn_model(num_classes=5, 
                                         num_input_feats=keras_model.X_train.shape[1],
                                         units=units,reg_strength=reg_strength,
                                         input_bands=keras_model.X_train.shape[2],
                                         dropout=dropout)
                # Fit model
                history = keras_model.fit()

                # Evaluate
                f.write('evaluate train: \n')
                #print('evaluate train: ', file=f)
                keras_model.evaluate('train', f, verbose)
                f.write('evaluate val: \n')
                #print('evaluate val: ', file=f)
                keras_model.evaluate('val', f, verbose)
                f.write('evaluate test: \n')
                #print('evaluate test: ', file=f)
                keras_model.evaluate('test', f, verbose)

                # Plot
                plot(history, model_type, dataset_type, source, ordering, units, reg_strength)
                
                f.close()

if __name__ == '__main__':
   main()
