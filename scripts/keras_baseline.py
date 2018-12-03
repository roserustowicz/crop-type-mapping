import numpy as np
import os
import itertools
import random
import sys

import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras import regularizers, optimizers
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

    def load_data(self, dataset_type, source, ordering, verbose, full_sampled, reshape_bands, binary):
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
            print('Small pixel arrays are not currently saved')

        elif dataset_type == 'full':
            if source == 's1': 
                self.X_train = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_Xtrain_g2260.npy')
                self.X_val = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_Xval_g298.npy')
                self.X_test = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_Xtest_g323.npy')
                            
                self.y_train = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_ytrain_g2260.npy')
                self.y_val = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_yval_g298.npy')
                self.y_test = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_ytest_g323.npy')
                    
            elif source == 's2':
                self.X_train = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtrain_g2260.npy')
                self.X_val = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_Xval_g298.npy')
                self.X_test = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtest_g323.npy')
                self.y_train = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_ytrain_g2260.npy')
                self.y_val = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_yval_g298.npy')
                self.y_test = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g323.npy')

            elif source == 's1_s2':
                # use np.hstack() to combine raw s1 and s2
                s2_train = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtrain_g2260.npy')
                s2_val = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_Xval_g298.npy')
                s2_test = np.load(base_dir + 
                          '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_Xtest_g323.npy')

                s1_train = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_Xtrain_g2260.npy')
                s1_val = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_Xval_g298.npy')
                s1_test = np.load(base_dir + '/full/raw/full_raw_s1_sample_bytime_Xtest_g323.npy')
                   
                self.X_train = np.hstack((s1_train, s2_train))
                self.X_val = np.hstack((s1_val, s2_val))
                self.X_test = np.hstack((s1_test, s2_test))
        
                self.y_train = np.load(base_dir + 
                      '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_ytrain_g2260.npy')
                self.y_val = np.load(base_dir + 
                      '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_yval_g298.npy')
                self.y_test = np.load(base_dir + 
                      '/full/raw/full_raw_s2_cloud_mask_reverseFalse_bytime_ytest_g323.npy')
                    
        elif dataset_type == 'dummy:':

            self.X_train = np.ones((10, 30))
            self.y_train = np.ones((10,5))

            self.X_val = np.ones((4, 30))
            self.y_val = np.ones((4,5))

            self.X_test = np.ones((2, 30))
            self.y_test = np.ones((2,5))

        # filter out values of classes greater than 4
        self.X_train = self.X_train[self.y_train<=4, :]
        self.y_train = self.y_train[self.y_train<=4]

        self.X_val = self.X_val[self.y_val<=4, :]
        self.y_val = self.y_val[self.y_val<=4]

        self.X_test = self.X_test[self.y_test<=4, :]
        self.y_test = self.y_test[self.y_test<=4]
        
        # Normalize by standard scalar
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)

        if reshape_bands:
            if 's1' in source and 's2' in source: num_bands = 3+11
            elif 's1' in source: num_bands = 3
            elif 's2' in source: num_bands = 11
            self.X_train = reshape_channels(self.X_train, num_bands, ordering)
            self.X_val = reshape_channels(self.X_val, num_bands, ordering)
            self.X_test = reshape_channels(self.X_test, num_bands, ordering)
        else:
            self.X_train = np.expand_dims(self.X_train, axis=2)
            self.X_val = np.expand_dims(self.X_val, axis=2)
            self.X_test = np.expand_dims(self.X_test, axis=2)

        if binary:
            self.y_train[self.y_train > 2] = 1
            self.y_val[self.y_val > 2] = 1
            self.y_test[self.y_test > 2] = 1

            self.y_train = to_categorical(self.y_train.astype(int)-1,num_classes=2)
            self.y_val = to_categorical(self.y_val.astype(int)-1,num_classes=2)
            self.y_test = to_categorical(self.y_test.astype(int)-1,num_classes=2)
        else:
            self.y_train = to_categorical(self.y_train.astype(int)-1,num_classes=4)
            self.y_val = to_categorical(self.y_val.astype(int)-1,num_classes=4)
            self.y_test = to_categorical(self.y_test.astype(int)-1,num_classes=4)

        if verbose:
            print('X train: ', self.X_train.shape) #, self.X_train)
            print('X val: ', self.X_val.shape)
            print('X test: ', self.X_test.shape)
            print('y train: ', self.y_train.shape)
            print('y val: ', self.y_val.shape, np.unique(self.y_val))
            print('y test: ', self.y_test.shape, np.unique(self.y_test))
            print('y max: ', np.max(self.y_train))

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

    def evaluate(self, data_split, f, lr, verbose):
        """ Evaluate the model accuracy
  
        Args: 
          data_split - (str) the data split to use for evaluation 
                        of options 'train', 'val', or 'test'
        Returns: 
          prints the accuracy score
        """
        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss = 'categorical_crossentropy', 
                          optimizer=adam, 
                          metrics=['accuracy'])


        if data_split == 'train': 
            score = self.model.evaluate(self.X_train, self.y_train)
            pred = np.argmax(self.model.predict(self.X_train), axis=1)
            cm = confusion_matrix(np.argmax(self.y_train, axis=1), pred)
            f1_avg = get_f1score(cm, avg=True)
            f1_cls = get_f1score(cm, avg=False)
        elif data_split == 'val': 
            score = self.model.evaluate(self.X_val, self.y_val)
            pred = np.argmax(self.model.predict(self.X_val), axis=1)
            cm = confusion_matrix(np.argmax(self.y_val, axis=1), pred)
            f1_avg = get_f1score(cm, avg=True)
            f1_cls = get_f1score(cm, avg=False)
        elif data_split == 'test': 
            score = self.model.evaluate(self.X_test, self.y_test)
            pred = np.argmax(self.model.predict(self.X_test), axis=1)
            cm = confusion_matrix(np.argmax(self.y_test, axis=1), pred)
            f1_avg = get_f1score(cm, avg=True)
            f1_cls = get_f1score(cm, avg=False)

        if f:
            f.write('%s: %.2f%% \n' % (self.model.metrics_names[1], score[1]*100))
            f.write('average f1: %.3f%% \n' % (f1_avg))
            f.write('per class f1: ' + str(f1_cls) + '\n')
            f.write('Confusion Matrix:\n {}\n'.format(cm)) 
        if verbose:
            print('%s: %.2f%%' % (self.model.metrics_names[1], score[1]*100))
            print('average f1: %.3f%% \n' % (f1_avg))
            print('per class f1: ' + str(f1_cls) + '\n')
            print('Confusion Matrix: ', cm) 
        return f1_avg

    def fit(self, epochs, batch_size, class_weight=None, patience=5):
        """ Trains the model
        """
        # Compile model
        self.model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
        
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto', 
                     baseline=None, restore_best_weights=True), ModelCheckpoint(filepath='best_model.h5',
                     monitor='val_loss', save_best_only=True)]
       
        # Fit model
        history = self.model.fit(self.X_train, 
                       self.y_train, 
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(self.X_val, self.y_val),
                       verbose=1,
                       callbacks=callbacks, 
                       class_weight=class_weight)
        return history

def get_f1score(cm, avg=True):
    """ Calculate f1 score from a confusion matrix
    Args:
      cm - (np array) input confusion matrix
      avg - (bool) if true, compute average of all classes
                   if false, return separately per class
    """
    # calculate per class f1 score
    f1 = np.zeros(cm.shape[0])
    for cls in range(cm.shape[0]):
        f1[cls] = 2.*cm[cls, cls]/(np.sum(cm[cls, :])+np.sum(cm[:, cls]))
    # average across classes
    if avg:
        f1 = np.mean(f1)
    return f1

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

def plot(history, model_type, dataset_type, source, ordering, units, reg_strength, dropout):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    fname = 'acc_model{}-data{}{}-ordering{}-units{}-reg{}-dropout{}'.format(
               model_type, dataset_type, source, ordering, units, reg_strength, dropout)
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

def generate_int_power_HP(base, minVal, maxVal):
    exp = np.random.randint(minVal, maxVal + 1)
    return base ** exp

def generate_real_power_HP(base, minVal, maxVal):
    exp = np.random.uniform(minVal, maxVal)
    return base ** exp

def generate_int_HP(minVal, maxVal):
    return np.random.randint(minVal, maxVal + 1)

def generate_float_HP(minVal, maxVal):
    return np.random.uniform(minVal, maxVal)

def generate_string_HP(choices):
    return np.random.choice(choices)

def get_best_model():

    filename = '20181202_NN_best.txt'
    class_weight = {0: 0.8205, 1: 0.4012, 2: 0.8299, 3: 0.8780}

    model_type = 'nn'
    dataset_type = 'full'
    use_pca = 0
    ordering = 'bytime'
    reverse_clouds = 0
    verbose = 1
    full_sampled = 0 
    reshape_bands = 1 
    binary = 0
    epochs = 20
    num_search_samples = 10

    source = 's2'
    batch_size = 128
    dropout = 0.317266
    units = 16
    lr = 0.1
    reg_strength = 0.000001
    weight = 1
    num_classes = 4

    best_f1 = 0

    f = open(filename,'a+')
    f.write('--------- \n')
    f.write('{} model, {} dataset, {} source \n'.format(model_type, dataset_type, source))
    f.write('{} batch, {} dropout, {} units, {}learning rate, {} regularization, {} weight classes \n'.format(batch_size, dropout, units, lr, reg_strength, weight))
    f.close()

    count = 0
    while count < num_search_samples:
        count += 1

        # Define NN model
        keras_model = DL_model()
        # Load data into model
        keras_model.load_data(dataset_type, source, ordering, verbose, full_sampled, reshape_bands, binary)
        
        # Define model 
        if model_type == 'nn':
            keras_model.model = make_1d_nn_model(num_classes=num_classes, 
                                             num_input_feats=keras_model.X_train.shape[1],
                                             units=units,reg_strength=reg_strength,
                                             input_bands=keras_model.X_train.shape[2],
                                             dropout=dropout)
                                         
        elif model_type == 'cnn':
            keras_model.model = make_1d_cnn_model(num_classes=num_classes, 
                                             num_input_feats=keras_model.X_train.shape[1],
                                             units=units,reg_strength=reg_strength,
                                             input_bands=keras_model.X_train.shape[2],
                                             dropout=dropout)
        # Fit model
        if weight:
            history = keras_model.fit(epochs, batch_size, class_weight=class_weight)
        else:
            history = keras_model.fit(epochs, batch_size)

        f = open(filename,'a+')
        # Evaluate
        f.write('evaluate train: \n')
        keras_model.evaluate('train', f, lr, verbose)
        f.write('evaluate val: \n')
        val_f1 = keras_model.evaluate('val', f, lr, verbose)
        f.write('evaluate test: \n')
        keras_model.evaluate('test', f, lr, verbose)
        f.write('-------------------')
        f.close()

        if val_f1 > best_f1:
            best_f1 = val_f1            
            print('TO BEST MODEL: {} batch, {} dropout, {} units, {}learning rate, {} regularization, {} weight classes \n'.format(batch_size, dropout, units, lr, reg_strength, weight))

            # serialize model to JSON
            model_json = keras_model.model.to_json()
            json_out = 'best_' + model_type + '_model.json'
            hdf5_out = 'best_' + model_type + '_model.hdf5'
            with open(json_out, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            keras_model.model.save_weights(hdf5_out)
            print("Saved model to disk")
 
            ## later... 
            ## load json and create model
            #json_file = open('model.json', 'r')
            #loaded_model_json = json_file.read()
            #json_file.close()
            #loaded_model = model_from_json(loaded_model_json)
            ## load weights into new model
            #loaded_model.load_weights("model.h5")
            #print("Loaded model from disk")
 
            ## evaluate loaded model on test data
            #loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            #score = loaded_model.evaluate(X, Y, verbose=0)
            #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


def main():

    filename = '20181201_NN_full_reshapebands_results_earlystopping.txt'

    class_weight = {0: 0.8205, 1: 0.4012, 2: 0.8299, 3: 0.8780}

    model_type = 'nn'
    dataset_type = 'full'
    use_pca = 0
    ordering = 'bytime'
    reverse_clouds = 0
    verbose = 1
    full_sampled = 0 
    reshape_bands = 1 
    binary = 0
    epochs = 5
    num_search_samples = 100

    dropmin, dropmax = 0, 0.5
    lrbase, lrmin, lrmax  = 10, -6, -1  # base, min, max
    regbase, regmin, regmax = 10, -6, -1  # base, min, max
    unitbase, unitmin, unitmax = 2, 2, 9  # base, min, max
    batchbase, batchmin, batchmax = 2, 7, 11  # base, min, max
    hpr_sources = ['s1', 's2', 's1_s2']

    count = 0
    while count < num_search_samples:
        count += 1
        source = generate_string_HP(hpr_sources)
        print('source: ', source)

        f = open(filename,'a+')
            
        f.write('--------------------------------------- \n')
        f.write('{} model, {} dataset, {} source \n'.format(
          model_type, dataset_type, source))

        f.close()

        if verbose:
            print('--------------------------------------- \n')
            print('{} model, {} dataset, {} source \n'.format(
                model_type, dataset_type, source))
        
        batch_size = generate_int_power_HP(batchbase, batchmin, batchmax)
        dropout = generate_float_HP(dropmin, dropmax)
        units = generate_int_power_HP(unitbase, unitmin, unitmax)
        lr = generate_int_power_HP(lrbase, lrmin, lrmax)
        reg_strength = generate_int_power_HP(regbase, regmin, regmax)
        weight = random.getrandbits(1)

        f = open(filename, 'a+')

        f.write('--------- \n')
        f.write('{} batch, {} dropout, {} units, {}learning rate, {} regularization, {} weight classes \n'.format(batch_size, dropout, units, lr, reg_strength, weight))
            
        if verbose:
            print('--------- \n')
            print('{} units, {} regularization, {} dropout \n'.format(units, reg_strength, dropout))
 
        # Define NN model
        keras_model = DL_model()
        # Load data into model
        keras_model.load_data(dataset_type, source, ordering, verbose, full_sampled, reshape_bands, binary)
        
        if binary:
            num_classes=2
        else:
            num_classes=4
        # Define model 
        if model_type == 'nn':
            keras_model.model = make_1d_nn_model(num_classes=num_classes, 
                                             num_input_feats=keras_model.X_train.shape[1],
                                             units=units,reg_strength=reg_strength,
                                             input_bands=keras_model.X_train.shape[2],
                                             dropout=dropout)
                                         
        elif model_type == 'cnn':
            keras_model.model = make_1d_cnn_model(num_classes=num_classes, 
                                             num_input_feats=keras_model.X_train.shape[1],
                                             units=units,reg_strength=reg_strength,
                                             input_bands=keras_model.X_train.shape[2],
                                             dropout=dropout)
        # Fit model
        if weight:
            history = keras_model.fit(epochs, batch_size, class_weight=class_weight)
        else:
            history = keras_model.fit(epochs, batch_size)

        # Evaluate
        f.write('evaluate train: \n')
        #print('evaluate train: ', file=f)
        keras_model.evaluate('train', f, lr, verbose)
        f.write('evaluate val: \n')
        #print('evaluate val: ', file=f)
        keras_model.evaluate('val', f, lr, verbose)
        f.write('evaluate test: \n')
        #print('evaluate test: ', file=f)
        keras_model.evaluate('test', f, lr, verbose)
        f.write('-------------------')

        # Plot
        plot(history, model_type, dataset_type, source, ordering, units, reg_strength, dropout)
        f.close()

if __name__ == '__main__':
    get_best_model()
