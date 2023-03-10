# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:22:38 2023

@author: AndreasMiltiadous
"""

import torch
import pickle

class mlcDataset(torch.utils.data.Dataset):
    def __init__(self,path,transform=None):
        ''' Initialization'''
        self.data=None
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        #self.data = self.data[~(data['class'] == 'F')]       ## TO DO = make a parameter to choose if to drop A or F
        data=self.data
        self.data.drop(data[data['class']=='F'].index, inplace=True)
        self.data.reset_index(drop=True,inplace=True)
         
        self.x1 = self.data['conn']
        self.x2 = self.data['band']
        self.y = self.data['class']
        self.y = self.y.replace('F', 1)
        self.y = self.y.replace('A', 1)
        self.y = self.y.replace('C', 0)
        self.subj=self.data['subj']
    
################################################################################
    def __len__(self):
        ''' Length of the dataset '''
        return len(self.data)
################################################################################
    ''' Get the next item of the dataset '''
    def __getitem__(self,index):
        input1 = torch.tensor(self.x1[index]).type(torch.FloatTensor)
        input2 = torch.tensor(self.x2[index]).type(torch.FloatTensor)
        label = torch.tensor(int(self.y[index])).type(torch.FloatTensor)
        return input1, input2, label, index

class lpgo_dataset(mlcDataset):
    def __init__(self,path,idx,transform=None):
        super().__init__(path,transform)
        self.x1=self.x1.loc[idx]
        self.x2=self.x2.loc[idx]
        self.y=self.y.loc[idx]
        self.subj=self.subj.loc[idx]
        self.data=self.data.loc[idx]
        self.x1.reset_index(drop=True,inplace=True)
        self.x2.reset_index(drop=True,inplace=True)
        self.y.reset_index(drop=True,inplace=True)
        self.subj.reset_index(drop=True,inplace=True)
        self.data.reset_index(drop=True,inplace=True)