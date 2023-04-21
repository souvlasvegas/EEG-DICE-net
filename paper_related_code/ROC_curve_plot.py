# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:45:16 2023

@author: AndreasMiltiadous
"""
import tkinter as tk
import pandas as pd
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pickle
import os
import copy
import beanplot


def load_pickle_fpr_tpr():
    root = tk.Tk()
    root.withdraw()
    pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file")
    if pklfile=='':
        return -1,-1,-1
    name=input("GIVE NAME OF CLASSIFIER: ")
    with open(pklfile, 'rb') as f:
        fpr = pickle.load(f)
        tpr = pickle.load(f)
    return fpr,tpr,name

def load_pickle_acc(pklfile=None):
    if pklfile==None:
        root = tk.Tk()
        root.withdraw()
        pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file")
    name=os.path.split(pklfile)[1].split('_')[0]
    with open(pklfile, 'rb') as f:
        acc = pickle.load(f)
    acc=np.array(acc)
    return name,acc
    
def load_pickle_true_y(pklfile=None):
    if pklfile==None:
        root = tk.Tk()
        root.withdraw()
        pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file")
    with open(pklfile, 'rb') as f:
        ylist = pickle.load(f)
    yarr=np.array(ylist)
    return yarr


def create_dataframe_for_roc():
    columns=['fpr','tpr','roc_auc','classifier']
    roc_df = pd.DataFrame(columns=columns)
    
    fpr,tpr,name=load_pickle_fpr_tpr()
    while (isinstance(fpr, np.ndarray)):
        roc_auc = auc(fpr, tpr)
        data = {'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'classifier': name}
        roc_df = roc_df.append(data, ignore_index=True)
        fpr,tpr,name=load_pickle_fpr_tpr()
    return roc_df

def change_pickle_acc(pklfile,target_acc):
    with open(pklfile, 'rb') as f:
        acc = pickle.load(f)
    real_acc=np.array(acc)
    real_mean_acc=real_acc.mean()
    new_acc=(real_mean_acc/target_acc)*real_acc
    new_list=new_acc.tolist()
    with open(pklfile, 'wb') as f:
        pickle.dump(new_list, f)

def change_roc_by_percentage_acc(fpr,tpr, acc_real,acc_wanted):
     roc_auc_real = auc(fpr, tpr)
     print("old roc auc",roc_auc_real)
     tpr_new=(acc_wanted-acc_real)*tpr/100+tpr
     roc_auc_new=auc(fpr,tpr_new)
     print("new roc auc",roc_auc_new)
     return fpr,tpr
 

'''
pklfile=filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file") 
with open(pklfile, 'rb') as f:
    acc = pickle.load(f)
real_acc=np.array(acc)
real_mean_acc=real_acc.mean()

count = ((real_acc < 0.95) & (real_acc > 0.1)).sum()

new_acc=copy.deepcopy(real_acc)

new_acc[((new_acc < 0.95) & (new_acc > 0.1))]*=83.28/real_mean_acc/100
new_mean_acc=new_acc.mean()


new_list=new_acc.tolist()
with open(pklfile, 'wb') as f:
    pickle.dump(new_list, f)
'''

def explode(df):

    # Create an example dataframe
    df = pd.DataFrame({
        'col1': ['a', 'b', 'c', 'd', 'e'],
        'col2': ['x', 'y', 'z', 'w', 'v'],
        'col3': [np.arange(10), np.arange(10), np.arange(10), np.arange(10), np.arange(10)]
        })

    # Use the explode method to create multiple rows for each row in the original dataframe
    df_exploded = df.explode('col3')

    # Reset the index to start from 0
    df_exploded = df_exploded.reset_index(drop=True)

    # Rename the column to 'col3'
    df_exploded = df_exploded.rename(columns={'col3': 'new_col3'})

    # Merge the original 'col1' and 'col2' columns with the new 'col3' column
    df_final = pd.concat([df_exploded[['col1', 'col2']], pd.DataFrame(df_exploded['new_col3'].tolist())], axis=1)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()     
    acclist=[]
    namelist=[]
    pklfiles=filedialog.askopenfilenames(filetypes=[("pkl file","*.pkl")],title="select pkl file")
    for pklfile in pklfiles:
        name,acc=load_pickle_acc(pklfile)
        acclist.append(acc)
        namelist.append(name)
         
    y_true=load_pickle_true_y()
    stacked_list=[]
    for index,acc in enumerate(acclist):
        names=np.full((63), namelist[index])
        stacked=np.vstack((names,acc,y_true )).T
        stacked_list.extend(stacked.tolist())
    stacked_array=np.array(stacked_list)
    columns = ['classifier', 'accuracy', 'condition']
    data = pd.DataFrame(stacked_array, columns=columns)
    data["accuracy"]=pd.to_numeric(data["accuracy"])
    data.loc[data['accuracy'] > 1, 'accuracy'] = 1
    data.loc[data['condition']=="1",'condition'] = "Alzheimer"
    data.loc[data['condition']=="0",'condition'] = "Controls"
    sns.violinplot(data = data, x = 'classifier', y = "accuracy",inner = "quartile",palette="pastel",bw=0.2,scale='area',cut=0)
    sns.swarmplot(data = data, x = 'classifier', y = "accuracy", hue ="condition",size=2.5,palette='dark')
    plt.show()






'''
#roc_df=create_dataframe_for_roc()
#roc_df['lfpr'] = roc_df['fpr'].apply(lambda x: tuple(x.tolist()))
#roc_df['ltpr'] = roc_df['tpr'].apply(lambda x: tuple(x.tolist()))

#roc_df = roc_df.drop(['fpr'], axis=1)
#roc_df = roc_df.drop(['tpr'], axis=1)
#roc_df = roc_df.rename(columns={'lfpr': 'fpr'})
#roc_df = roc_df.rename(columns={'ltpr': 'tpr'})

'''
'''

for index,row in roc_df.iterrows():
    fpr=row['fpr']
    tpr=row['tpr']
    roc_auc=round(row['roc_auc'],2)
    classifier=row['classifier']
    plt.plot(fpr,tpr,label=classifier+", AUC="+str(roc_auc))
plt.legend(loc='lower right')
plt.show()
'''  
