# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:36:07 2022

@author: AndreasMiltiadous
"""
## the version of mne_connectivity used was 0.4.0

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import mne
import numpy as np
from mne_connectivity import spectral_connectivity_time
import split_dataset as sp


def read_data(file_path,epoch_size):
    """
    Parameters
    ----------
    file_path: .set filepath
    epoch_size: length of epoch in seconds

    Returns
    -------
    epochs : object of type mne.Epochs, 
    To get array of dimension: (E,C,T). call epochs.get_data() E= n of epochs, C= n of channels, T= time points

    """
    data=mne.io.read_raw_eeglab(file_path,preload=False,uint16_codec='latin1')
    epochs=mne.make_fixed_length_epochs(data,duration=epoch_size,overlap=epoch_size/2)
    #array=epochs.get_data()
    return epochs


def create_subject_epoch_list(files=None, epoch_size=2):
    """
    Parameters:
    ----------
    files: tuple of filenames of Matlab type (.set) Must be preprocessed
    epoch_size: length of size to split
    if files=None, prompts user to open files
    
    Returns: 
    -------
    list of mne.Epochs objects of epoched datasets
    +
    list of filenames
    """
    if files==None:
        root = tk.Tk()
        root.withdraw()
        files = filedialog.askopenfilenames(filetypes=[("set file","*.set")], title="open .set files (must be preprocessed)")
    subject_list = []
    subject_list=[read_data(file,epoch_size) for file in files]
    return subject_list, files

def avg_diag(A):
    """
    Parameters
    ----------
    A : numpy array of shape (X,X)

    Returns : Average of down triangular matrix
    -------

    """
    n=A.shape[0]-1
    pli8os = 0
    while(n > 0):
        pli8os=pli8os+n
        n=n-1
    return (A.sum() - np.diag(A).sum())/(2*pli8os)

##foi=np.array([[0.5,4],[4,8],[8,12],[12,25],[25,45]])
#freqs=[2,6,10,18,35]
#n_cycles=[2,4,7,7,7]


# functions for spectral coherence connectivity with morlet wavelets
#########################################################################################################################################
def calc_global_spectral_coherence_connectivity(subject,freqs=[2,6,10,18,35],n_cycles=[2,4,7,7,7]):
    """
    This function calculates the global spectral coherence connectivity of an epoched signal using the morlet wavelet transform.
    Parameters
    ----------
    subject : Obj of type mne.Epochs : The epochs of one subject.
    freqs : TYPE, optional
        DESCRIPTION. The default is [2,6,10,18,35].
    n_cycles : TYPE, optional
        DESCRIPTION. The default is [2,4,7,7,7].
    Returns
    -------
    Numpy array of size (Epochs, Bands). 
    Each value is the global spatial coherence coeficient for each epoch for each band (wavelet to be specific)

    """
    sfreq=subject.info['sfreq']
    arr=spectral_connectivity_time(subject,method='coh',mode='cwt_morlet',freqs=freqs,sfreq=sfreq,n_cycles=n_cycles)

    con_time = arr.get_data()
    con_avg_time = np.mean(con_time,axis=-1)
    con_avg_transpose=np.transpose(con_avg_time,(0,3,1,2))
    
    con_sum=pd.DataFrame(columns=['Theta','Delta','Alpha','Beta','Gamma'])
    for arr in con_avg_transpose:
        temp=[avg_diag(A) for A in arr]
        con_sum.loc[len(con_sum)] = temp
    return (con_sum.to_numpy())
    
def calc_spectral_coherence_connectivity(subject,freqs=[2,6,10,18,35],n_cycles=[2,4,7,7,7]):
    """
    This function calculates the spectral coherence connectivity of each channel of an epoched signal using the morlet wavelet transform.
    Parameters

    Parameters
    ----------
    subject : Obj of type mne.Epochs : The epochs of one subject.
    freqs : TYPE, optional
        DESCRIPTION. The default is [2,6,10,18,35].
    n_cycles : TYPE, optional
        DESCRIPTION. The default is [2,4,7,7,7].

    Returns
    -------
    numpy array of size (Epochs, Bands, Channels)
    Each value is the spatial coherence coeficient for each epoch for each band for each electrode

    """
    sfreq=subject.info['sfreq']
    arr=spectral_connectivity_time(subject,method='coh',mode='cwt_morlet',freqs=freqs,sfreq=sfreq,n_cycles=n_cycles)
    con_time = arr.get_data()
    con_avg_time = np.mean(con_time,axis=-1)
    con_avg_transpose=np.transpose(con_avg_time,(0,3,1,2))
    lista=[]
    for B in con_avg_transpose:
        lista.append([(np.sum(A,axis=0)-1)/(A.shape[0]-1) for A in B])
    return np.array(lista)

########################################################################################################################################
    
def calc_spectral_coherence_connectivity_2dmatrix(subject,freqs=[2,6,10,18,35],n_cycles=[2,4,7,7,7]):
    sfreq=subject.info['sfreq']
    arr=spectral_connectivity_time(subject,method='coh',mode='cwt_morlet',freqs=freqs,sfreq=sfreq,n_cycles=n_cycles)
    con_time = arr.get_data()
    con_avg_time = np.mean(con_time,axis=0) 
    con_avg_time=np.mean(con_avg_time,axis=-1)
    return con_avg_time



##############################################################################################################################################


if __name__ == "__main__":
    subject_list, filenames=create_subject_epoch_list()
    results=[calc_spectral_coherence_connectivity(subject,freqs=[2,6,10,18,35]) for subject in subject_list]
    for i,result in enumerate(results):
        listt=sp.split_array_to_fixed_sized_arrays(result,splitsize=30,to_csv=False,filename=filenames[i])
        arr_coherence = np.array(listt)

'''

if __name__ == "__main__":
    
    #### Ask user to open .set files for each condition & Reads data with MNE & splits to epochs
    subject_listA, filenamesA=create_subject_epoch_list()
    subject_listC, filenamesC=create_subject_epoch_list()
    subject_listF, filenamesF=create_subject_epoch_list()
    
    results=[calc_spectral_coherence_connectivity_2dmatrix(subject) for subject in subject_listA]
    mean_arrayA = np.mean(results, axis=0)
    
    results=[calc_spectral_coherence_connectivity_2dmatrix(subject) for subject in subject_listC]
    mean_arrayC = np.mean(results, axis=0)
    
    results=[calc_spectral_coherence_connectivity_2dmatrix(subject) for subject in subject_listF]
    mean_arrayF = np.mean(results, axis=0)
    
    ######################
    import pickle
    
    root = tk.Tk()
    root.withdraw()
    dire = filedialog.askdirectory()
    with open(dire+"//numpy_heatmap.pkl", 'wb') as f:
        pickle.dump((mean_arrayA,mean_arrayC,mean_arrayF), f)
    
'''