# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:57:08 2023

@author: AndreasMiltiadous
"""

"""
DO NOT USE
"""
## Some patients of group A and Group C have the same patient number.
## There we fix this. Needs to be run only one time.

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file that needs fixing")

import pickle
with open(pklfile, 'rb') as f:
    data = pickle.load(f)
  
def check_if_needs_fix(data):
    d=data.loc[data['class']=='C']
    a=d.loc[d['subj']=="1"]
    return not a.empty
    

if check_if_needs_fix(data):
    print("HI")
    for index,row in data.iterrows():
        if row['class']=='C':
            row['subj']=str(int(row['subj'])+40)
    
data.to_pickle("AlzheimerTrainingDatasetFixxed.pkl")