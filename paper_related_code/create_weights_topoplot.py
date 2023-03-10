# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:41:31 2023

@author: AndreasMiltiadous
"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
import mne



root = tk.Tk()
root.withdraw()
npyfile1 = filedialog.askopenfilename(filetypes=[("pkl file","*.npy")],title="select npy file")
npyfile2 = filedialog.askopenfilename(filetypes=[("pkl file","*.npy")],title="select npy file")

data1=np.load(npyfile1)
data2=np.load(npyfile2)

setfile = filedialog.askopenfilename(filetypes=[("set file","*.set")])
data=mne.io.read_raw_eeglab(setfile,preload=True)

info=data.info["ch_names"]
 
conn=dict(zip(info,data1))

band=dict(zip(info,data2))

mne_info=mne.create_info(info, sfreq=300, ch_types='eeg')
montage=mne.channels.make_standard_montage('standard_1020')
mne_info.set_montage(montage)


import matplotlib.pyplot as plt

cmap = plt.get_cmap('coolwarm') 


def find_min_max(arr1, arr2):
    min_val = np.min([np.min(arr1), np.min(arr2)])
    max_val = np.max([np.max(arr1), np.max(arr2)])
    return min_val, max_val

def normalize_array(arr1,arr2):
    min_val, max_val=find_min_max(arr1,arr2)
    arr1_norm = (arr1 - min_val) / (max_val - min_val)
    arr2_norm = (arr2 - min_val) / (max_val - min_val)
    return arr1_norm,arr2_norm,min_val,max_val

vmin,vmax=find_min_max(data1,data2)

conn_norm,band_norm,_,_=normalize_array(np.array(list(conn.values())),np.array(list(band.values())))


fig,(ax1,ax2) = plt.subplots(ncols=2)
ax1.set_title("Coherence",fontdict={'family': 'Times New Roman', 'size': 14})
ax2.set_title("Band Power",fontdict={'family': 'Times New Roman', 'size': 14})
im,cm   = mne.viz.plot_topomap(conn_norm, mne_info, cmap=cmap, axes=ax1,show=False,vmin=0,vmax=1)   
im,cm   = mne.viz.plot_topomap(band_norm, mne_info, cmap=cmap, axes=ax2,show=False,vmin=0,vmax=1)   
# manually fiddle the position of colorbar
ax_x_start = 0.95
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.9
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
plt.suptitle("Normalized Absolute Magnitute of Convolution layer weights",fontdict={'family': 'Times New Roman', 'size': 30})
plt.show()