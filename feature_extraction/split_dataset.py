# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:03:25 2022

@author: AndreasMiltiadous
"""


import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import mne
import copy

def split_array_to_fixed_sized_arrays (data, splitsize=30, to_csv=False, filename=None):
    """
    Splits a numpy array (regardless of dimension), on fixed size arrays (at first dimension) and stores them in a list.
    If to_csv is set True, then saves the generated arrays in files (flatten to 2d matrixes if 3d)

    Parameters
    ----------
    data : numpy array
    splitsize : int, number of elements for fixed size arrays
    to_csv : bool, True if you want to write to .csv files
    filename: filename of .set file used to create epochs

    Returns
    -------
    list of arrays

    """
    lista=[]
    while (data.shape[0]>=splitsize):
        x=np.split(data,[splitsize])
        data=x[1]
        lista.append(x[0])
        
    #write to files
    if (to_csv==True):
        
        #naming bullshit
        if filename==None:
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(title="Choose .set file that was used")
        filepath=os.path.split(filename)[0]
        name=os.path.split(filename)[1]
        plain_name=os.path.splitext(name)[0]
        #####
        
        for i,arr in enumerate(lista):
            ##check dimension of nparray, because 3d array needs flatten
            if arr.ndim==3:
                print("needs flatten")
                flatt=True
                arr_reshaped=arr.reshape(arr.shape[0],-1)
            if flatt==True:
                np.savetxt(filepath+"/"+"flat_"+plain_name+"_"+str(i)+".csv",arr_reshaped)
            elif arr.ndim==2 or arr.ndim==1:
                np.savetxt(filepath+"/"+plain_name+"_"+str(i)+".csv",arr)
    return lista

def load_original_arr(arr,channels=19):
    """
    Gets flattened array of type 2D [Epochs, Bands * Channels] and returns 3D array [Epochs, Bands, Channels]

    Parameters
    ----------
    arr : numpy array flattened [Epochs, Bands * Channels]
    channels : number of channels in EEG recording
        DESCRIPTION. The default is 19.

    Returns
    -------
    numpy array deflatenned 3D [Epochs, Bands, Channels]

    """
    return arr.reshape(arr.shape[0], arr.shape[1] // channels, channels)

def read_flattened(filename,channels=19):
    """
    Gets filename of flattened 3d file of type [Epochs, Bands * Channels] , returns 3D array [Epochs, Bands, Channels]

    Parameters
    ----------
    filename : filename
    channels : TYPE, optional
        DESCRIPTION. The default is 19.

    Returns
    -------
    arr : numpy array deflatenned 3D [Epochs, Bands, Channels]

    """
    flat_arr=np.loadtxt(filename)
    arr=load_original_arr(flat_arr, channels=19)
    return arr

def split_setfile_fixed_size(filename, window=30):
    '''
    Gets a .set file, splits the file into fixed sized windows (not epochs), and returns list with RawArray
    objects of the splitted file
    Parameters
    ----------
    filename : path to .set file (can be modified)
    window : int , size of time-window splitting
        DESCRIPTION. The default is 30.

    Returns
    -------
    cuts : list of RawArrays. Each RawArray is a Raw Object of fixed size.

    '''
    data=mne.io.read_raw_eeglab(filename,preload=True)
    epochs=mne.make_fixed_length_epochs(data,duration=30)
    info=data.info
    info2=copy.deepcopy(info)
    cuts=[]
    for i,epoch in enumerate(epochs):
        info2=copy.deepcopy(info)
        obj=mne.io.RawArray(epoch,info2)
        cuts.append(obj)
    return cuts

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilenames(filetypes=[("set file","*.set")])
    for file in filename:
        data=mne.io.read_raw_eeglab(file,preload=True)
        epochs=mne.make_fixed_length_epochs(data,duration=30)
        
    for i,epoch in enumerate(epochs):
        print(i)
    
    info=data.info

##filename = filedialog.askopenfilenames(filetypes=[("set file","*.set")])    
##for data in filename:
##    data1=mne.io.read_raw_eeglab(data,preload=False)