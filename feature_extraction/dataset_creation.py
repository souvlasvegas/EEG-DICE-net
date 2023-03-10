# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:40:50 2022

@author: AndreasMiltiadous
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np
import global_spectral_coherence_computation as gs
import relative_band_power_computation as rb
import split_dataset as sp
import os


def create_numpy_connAndband_files():

    filepath=filedialog.askdirectory(title="where to save the created .npy files")
    subject_list, filenames=gs.create_subject_epoch_list()
    conn_results=[gs.calc_spectral_coherence_connectivity(subject) for subject in subject_list]
    band_results=[rb.calc_relative_band_power(subject) for subject in subject_list]
    for i,result in enumerate(conn_results):
        listt=sp.split_array_to_fixed_sized_arrays(result,splitsize=30,to_csv=False,filename=filenames[i])
        arr_coherence = np.array(listt)
        
        ##naming
        path=os.path.split(filenames[i])[0]
        name=os.path.split(filenames[i])[1]
        plain_name=os.path.splitext(name)[0]
        with open( filepath + '/' + plain_name + "_conn.npy", 'wb') as f:
            np.save(f, arr_coherence)
    
    for i,result in enumerate(band_results):
        listt=sp.split_array_to_fixed_sized_arrays(result,splitsize=30,to_csv=False,filename=filenames[i])
        arr_band = np.array(listt)
        
        ##naming
        path=os.path.split(filenames[i])[0]
        name=os.path.split(filenames[i])[1]
        plain_name=os.path.splitext(name)[0]
        with open( filepath + '/' + plain_name + "_band.npy", 'wb') as f:
            np.save(f, arr_band)
        
def create_training_dataset(filelist=None):
    '''
    Gets (or asks the user for) a list of .npy files that MUST be named as A4_band.npy and A4_conn.npy (S0S: ordered alphabetically)
    returns training_dataframe with columns [subj, conn (numpy array), band (numpy array), class]

    Parameters
    ----------
    filelist : TYPE, list of filenames
        DESCRIPTION. The default is None.

    Returns
    -------
    training_dataframe : dataframe, each row is one training sample.

    '''
    if filelist==None:
        filelist=filedialog.askopenfilenames(filetypes=[("numpy file","*.npy")])
    band_list=[]
    conn_list=[]
    training_dataframe=pd.DataFrame(columns=['subj','conn','band','class'])    
    for file in filelist:
        path=os.path.split(file)[0]
        name=os.path.split(file)[1]
        plain_name=os.path.splitext(name)[0]
        if plain_name.split("_")[1]=="band":
            band_list.append(file)
        elif plain_name.split("_")[1]=="conn":
            conn_list.append(file)
        else:
            print("something wrong")
            #return -1;
    for i,band_file in enumerate(band_list):
        conn_file=conn_list[i]
        band_subj=os.path.splitext(os.path.split(band_file)[1])[0].split("_")[0]
        conn_subj=os.path.splitext(os.path.split(conn_file)[1])[0].split("_")[0]
        print(band_subj,conn_subj)
        if conn_subj!=band_subj:
            print("error in file selection")
            #return -1;
        Class=band_subj[0]
        subj=band_subj[1:]
        conn=np.load(conn_file)
        list_conn=[s for s in np.load(conn_file)]
        list_band=[s for s in np.load(band_file)]
        for j,conn in enumerate(list_conn):
            band=list_band[j]
            d={'subj': subj, 'conn':conn,'band':band,'class':Class}
            ser=pd.Series(data=d,index=['subj','conn','band','class'])
            training_dataframe=training_dataframe.append(ser,ignore_index=True)
    return training_dataframe
    
if __name__ == "__main__":
    training_dataframe=create_training_dataset()
    filepath=filedialog.askdirectory(title="where to save the .pkl total file")
    training_dataframe.to_pickle(filepath+'/AlzheimerTrainingDataset.pkl')