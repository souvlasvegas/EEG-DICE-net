# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:29:33 2022

@author: AndreasMiltiadous
"""

import mne
import numpy as np
import split_dataset as sp
import global_spectral_coherence_computation as gs
from mne.time_frequency import psd_welch


def sum_welch_windows(psds,fmin,fmax,freqs):
    psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)]
    return np.sum(psds_band,axis=2)


def calc_relative_band_power (subject, FREQ_BANDS=None):
    """
    calculates relative band power of each channel

    Parameters
    ----------
    subject : mne.Epochs object
    FREQ_BANDS : dict of the frequency bands

    Returns
    -------
    arr : numpy array of size (Epochs, Bands, Channels)
    Each value is the relative band power for each epoch for each band for each electrode

    """
    if FREQ_BANDS==None:
        FREQ_BANDS = {"delta": [0.5, 4],
                      "theta": [4, 8],
                      "alpha": [8, 13],
                      "beta": [13, 25],
                      "gamma": [25, 45]}
    spectrum=subject.compute_psd(method="welch",fmin=0.5,fmax=45,verbose=1)
    x=spectrum.get_data(return_freqs=True)
    psds=x[0]
    freqs=x[1]     
    arr=np.array([sum_welch_windows(psds,values[0],values[1],freqs=freqs) for key,values in FREQ_BANDS.items()])
    arr_transpose=np.transpose(arr,(1,0,2))    
    arr=np.array([data/data.sum(axis=0) for data in arr_transpose])
    return arr    
    
######
if __name__ == "__main__":
    subject_list, filenames=gs.create_subject_epoch_list()
    for subject in subject_list:
        arr=calc_relative_band_power(subject)
