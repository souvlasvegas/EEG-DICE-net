# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:40:02 2023

@author: m_ant
"""

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file")


import pickle
with open(pklfile, 'rb') as f:
  data = pickle.load(f)
  
connA=data.conn[0]
connC=data.conn[1173]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


Amean=np.mean(connA,axis=0)
Cmean=np.mean(connC,axis=0)

AmeanT=Amean.T


listA = [row['conn'] for index,row in data.iterrows() if row['class']=="A" ]
listC = [row['conn'] for index,row in data.iterrows() if row['class']=="C" ]
listF = [row['conn'] for index,row in data.iterrows() if row['class']=="F" ]

listAmean=[np.mean(row,axis=0).T for row in listA]
listCmean=[np.mean(row,axis=0).T for row in listC]
listFmean=[np.mean(row,axis=0).T for row in listF]

resultA=np.zeros(listAmean[0].shape)
resultC=np.zeros(listCmean[0].shape)
resultF=np.zeros(listCmean[0].shape)


for arr in listAmean:
    resultA+=arr
    
for arr in listCmean:
    resultC+=arr
    
for arr in listFmean:
    resultF+=arr   

resultAscaled=resultA/len(listAmean)
resultCscaled=resultC/len(listCmean)
resultFscaled=resultF/len(listFmean)
##########################################################

A_average_of_electrodes= np.mean(resultAscaled,axis=0)
C_average_of_electrodes= np.mean(resultCscaled,axis=0)
F_average_of_electrodes= np.mean(resultFscaled,axis=0)


resultAconcat=np.concatenate((resultAscaled, A_average_of_electrodes[np.newaxis,:]),axis=0)
resultCconcat=np.concatenate((resultCscaled, A_average_of_electrodes[np.newaxis,:]),axis=0)
resultFconcat=np.concatenate((resultFscaled, A_average_of_electrodes[np.newaxis,:]),axis=0)


###########################################################

fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(8, 4))

cmap="Greys"

axes=[ax1,ax2,ax3]
im = ax1.imshow(resultAscaled, cmap=cmap)
ax1.set_title("AD group")
im = ax2.imshow(resultCscaled, cmap=cmap)
ax2.set_title("CN group")
im = ax3.imshow(resultFscaled, cmap=cmap)
ax3.set_title("FTD group")

ch_names=['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
bands=['Delta','Theta','Alpha','Beta','Gamma']

xtick_locs=[0,1,2,3,4]
ytick_locs=list(range(19))

ax1.set_xticklabels(bands,rotation=90)
ax1.set_xticks(xtick_locs)
ax2.set_xticklabels(bands,rotation=90)
ax2.set_xticks(xtick_locs)
ax3.set_xticklabels(bands,rotation=90)
ax3.set_xticks(xtick_locs)

ax1.set_yticklabels(ch_names)
ax1.set_yticks(ytick_locs)
ax2.set_yticklabels(ch_names)
ax2.set_yticks(ytick_locs)
ax3.set_yticklabels(ch_names)
ax3.set_yticks(ytick_locs)

#cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
plt.colorbar(im,ax=axes)


plt.show()