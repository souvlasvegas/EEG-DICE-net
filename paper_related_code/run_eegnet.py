# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:48:26 2023

@author: AndreasMiltiadous
"""
from pymatreader import read_mat
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import mne
from pathlib import Path
import os
import numpy as np
from EEGModels import EEGNet
from EEGModels import EEGNet_SSVEP
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import backend as K
import sys
from sklearn.model_selection import LeaveOneGroupOut

root = tk.Tk()
root.withdraw()
files = filedialog.askopenfilenames(filetypes=[("set file","*.set")], title="open .set files (must be preprocessed)")

alzheimer_files = []
frontotemporal_files =[]
control_files = []

X= np.empty((0, 19, 1000))
y=np.empty(0)
group=np.empty(0)

for file in files:
    filename = Path(file).name
    print(filename)
    split_tup = os.path.splitext(filename)
    file_name = split_tup[0]
    if "A" in file_name:
        alzheimer_files.append(file)
        cl=1
    elif "C" in file_name:
        control_files.append(file)
        cl=0
    elif "F" in file_name:
        frontotemporal_files.append(file)
        cl=2
    sub=file_name[1:]
    data=mne.io.read_raw_eeglab(file,preload=True)
    pos=data.info.ch_names
    
    sfreq=data.info['sfreq']
  
    data.resample(250)

    epochs=mne.make_fixed_length_epochs(data,duration=4,overlap=2)
    data_array=epochs.get_data()
    
    
    classes=np.full(data_array.shape[0],cl)
    groups=np.full(data_array.shape[0],sub)
    X= np.append(X, data_array, axis=0)
    y=np.append(y,classes,axis=0)
    group=np.append(group,groups,axis=0)
    
##################################################################################################
##################################################################################################

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    
kernels, chans, samples = 1, 19, 1000

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 1}))

logo = LeaveOneGroupOut()  # Create a LeaveOneGroupOut object
scores = []  # List to store validation scores

fold_accuracies=[]

confusion_table=np.array([[0, 0], [0, 0]])


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for the first GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

batch_size=32
num_epochs=400

n_groups = logo.get_n_splits(groups=group)
print(n_groups)

########################
del data
del groups
del data_array
#########################

import psutil

svmem = psutil.virtual_memory()

# Available memory in bytes
available_memory = svmem.available
print(available_memory/(1024**3))
##########################################

print(sys.getsizeof(X)/(1024**3))

for idx,(train_index, test_index) in enumerate(logo.split(X, y, group)):
    # Split the data into training and testing sets for the current fold
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    
    ## debug
    print(idx, X_train.nbytes , X_train.nbytes/(1024**3))
    
    
    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
    
    Y_train      = np_utils.to_categorical(Y_train)
    Y_test       = np_utils.to_categorical(Y_test)
    
    train_gen = DataGenerator(X_train, Y_train, batch_size)
    test_gen = DataGenerator(X_test, Y_test, batch_size)
    
    #train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    #train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0])
    #train_dataset = train_dataset.batch(batch_size)
    #train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Create a validation dataset from numpy arrays
    #val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    #val_dataset = val_dataset.batch(batch_size)
    #val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    #model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
              #dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
              #dropoutType = 'Dropout')
    
    model = EEGNet_SSVEP(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])
    
    #checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                   #save_best_only=True)
    
    fittedModel = model.fit(train_gen, batch_size = batch_size, epochs = num_epochs, 
                            verbose = 2)
    
    #model.load_weights('/tmp/checkpoint.h5')
    
    
    probs       = model.predict(test_gen)
    preds       = probs.argmax(axis = -1)
    Y_test_int = np.argmax(Y_test, axis=-1)
    acc         = np.mean(preds == Y_test_int)
    print("Classification accuracy: %f " % (acc))
    fold_accuracies.append(acc)
    
    cm = confusion_matrix(Y_test_int, preds)
    confusion_table = confusion_table + cm
    K.clear_session()


del X_train, X_test, Y_train, Y_test

# Calculate and print the average score across all folds
mean_acc=np.asarray(fold_accuracies).mean()
average_score = np.mean(scores)
print("Average score:", average_score)

def calc_scores_from_confusionmatrix(conf_matrix):
    '''
    Parameters
    ----------
    conf_matrix : numpy array (only for binary classification)
    
    Returns
    -------
    accuracy : 
    sensitivity : 
    specificity : 
    precision : 
    f1 : .
    
    '''
    accuracy=(conf_matrix[0][0]+conf_matrix[1][1])/np.sum(conf_matrix)
    sensitivity=conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1])
    specificity=conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[1][0])
    precision=conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][0])
    f1=2*(precision*sensitivity)/(precision+sensitivity)
    return accuracy,sensitivity,specificity,precision,f1

accuracy,sensitivity,specificity,precision,f1=calc_scores_from_confusionmatrix(confusion_table)

print("confmatrix results")
print('f1_score',f1)
print('sensitivity',sensitivity)
print('specificity',specificity)
print('precision',precision)
print('accuracy',accuracy)


