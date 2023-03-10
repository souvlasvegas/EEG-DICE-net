# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:56:37 2023

@author: AndreasMiltiadous
"""
import copy
import numpy
import warnings
import torch
import sys
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import dataset
import utils
import train_eval_func as tef
import dice_models
import calc_metrics as cm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import KFold

"""
This script performs early stopping to determine best epoch number.

STEPS:
    1) divide dataset to (train-test) - validation set on a 9/1 scale
    2) for each (train-test) perform LOSO validation and stop when no improvement for 20 epochs
    3) print results in txt file, average the epochs used and get a rough estimate
    4) Go to main.py and exclusively check epoch sizes around the rough estimate for a better epoch size estimate.

Upon execution, you will be asked to:
    1) Choose a .pkl file containing the input data in proper form (choose AlzheimerTrainingDatasetFixxed.pkl)
    2) Choose a directory to save the results in a .txt file (no need to create the txt file, it is automatically created)
    
Please read the code and check all STOP HERE: before execution
    
"""

class EarlyStopping():
    def __init__(self,patience=30, min_delta=0,restore_best_weights=True):
        self.patience=patience
        self.min_delta=min_delta
        self.restore_best_weights=restore_best_weights
        self.best_model=None
        self.best_loss=None
        self.counter=0
        self.status=""
        
    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss=val_loss
            self.best_model=copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter=0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter+=1
            if self.counter>=self.patience:
                self.status= f"stopped on {self.counter}"
                if self.restore_best_weights==True:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status=f"{self.counter}/{self.patience}"
        return False
    

def logo_run(path, logo_idx, val_idx, confusion_table,confusion_table_of_run,dire=None):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logo = LeaveOneGroupOut()
    train_dataset=dataset.lpgo_dataset(path,logo_idx)
    val_dataset=dataset.lpgo_dataset(path,val_idx)
    new_val_idx=[x for x in range(len(val_dataset))]
    confusion_table_of_run=numpy.array([[0, 0], [0, 0]])
    print(logo.get_n_splits(groups=train_dataset.subj))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        #######################################################################
        #STOP HERE: select one of 2 folds and comment out the other one
        
        #LEAVE ONE OUT
        for fold, (train_idx,test_idx) in enumerate(logo.split(X=numpy.arange(len(train_dataset)),y=train_dataset.y,groups=train_dataset.subj)):
            
        #KFOLD
        #for fold, (train_idx,val_idx) in enumerate(splits.split(numpy.arange(len(dataset)))):
        #######################################################################   
            #define early stopping
            es=EarlyStopping()
            
            ############################################
            utils.reproducability(seed=7)
            ############################################
            ###################################################################
            #STOP HERE: choose model from dice_models
            model = dice_models.Model_no_encoder()
            ###################################################################
            model = model.to(device)
            ############################################
            criterion = torch.nn.BCEWithLogitsLoss()
            criterion = criterion.to(device)
            ############################################
            learning_rate = 0.001      # optimizer step
            weight_decay = 0.01        # L2 regularization
            optimizer_params = {'params': model.parameters(),
                                'lr': learning_rate,
                                'weight_decay': weight_decay}
            optimizer = torch.optim.AdamW(**optimizer_params)
            ############################################
            batch_size = 32    # batch size
            #shuffle = True      # if you want to shuffle your data after each epoch
            drop_last = True   # the last batch (maybe) contains less samples than the batch_size maybe you do not want it's gradients
            num_workers = 0     # number of multi-processes data loading
            pin_memory = True   # enable fast data transfer to CUDA-enabled GPUs
                # ---------- Properties ----------
            loader_params = {'batch_size': batch_size,
                              'drop_last': drop_last,
                              'num_workers': num_workers,
                              'pin_memory': pin_memory}
            epochs = 200
            
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(new_val_idx)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
            test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, sampler=test_sampler,drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,drop_last=True)
            
            
            for epoch in tqdm(range(epochs)):
                train_loss, train_acc, train_f1, train_rec, train_pre = tef.train_epoch(train_loader, optimizer, model, criterion)
                test_loss, test_acc, test_f1, test_rec, test_pre, _ = tef.eval_epoch(model, val_loader, criterion)
                if es(model,test_loss)==True:
                    break
            ## Run eval_epoch again because EarlyStopping restored the best weights
            test_loss, test_acc, test_f1, test_rec, test_pre, conf_matrix = tef.eval_epoch(model, test_loader, criterion)
            confusion_table = confusion_table + conf_matrix
            confusion_table_of_run = confusion_table_of_run + conf_matrix
            with open(dire+"/results.txt", 'a') as f:
                sys.stdout = f # Change the standard output to the file we created.
                print("Patient: " + str(train_dataset.data.loc[test_idx,'subj'].iloc[0]) + " Condition: " + str(train_dataset.y.loc[test_idx].iloc[0]))
                print('epoch', epoch, 'train loss', train_loss, 'train acc', train_acc, 'train f1', train_f1, 'test loss', test_loss, 'test acc', test_acc, 'test f1', test_f1)
                sys.stdout = original_stdout # Reset the standard output to its original value
            print("Patient: " + str(train_dataset.data.loc[test_idx,'subj'].iloc[0]) + " Condition: " + str(train_dataset.y.loc[test_idx].iloc[0]))
            print('epoch', epoch, 'train loss', train_loss, 'train acc', train_acc, 'train f1', train_f1, 'test loss', test_loss, 'test acc', test_acc, 'test f1', test_f1)
    return confusion_table, confusion_table_of_run



##################################################
#CUSTOM LEAVEPGROUPSOUT

if __name__ == "__main__":
    original_stdout = sys.stdout # Save a reference to the original standard output
    root = tk.Tk()
    root.withdraw()
    pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file")


    root = tk.Tk()
    root.withdraw()
    dire = filedialog.askdirectory(title="where to save the results")
    dataset=dataset.mlcDataset(pklfile)
    
    percentage=1/10
    
    p=int(numpy.round(percentage*len(numpy.unique(dataset.subj.to_numpy()))))
    
    unique_groups=numpy.unique(dataset.subj.to_numpy())
    pool_of_groups=numpy.copy(unique_groups)
    n_iterations=int(len(unique_groups)/p)
    
    confusion_table=numpy.array([[0, 0], [0, 0]])
    for i in range(n_iterations):
        confusion_table_of_run=numpy.array([[0, 0], [0, 0]])
        groups_to_leave_out= numpy.random.choice(pool_of_groups,p,replace=False)
        A= [i for i in pool_of_groups if i not in groups_to_leave_out]
        pool_of_groups=numpy.asarray(A)
        mask=numpy.isin(dataset.subj.to_numpy(),groups_to_leave_out,invert=True)
        logo_idx=numpy.argwhere(mask == True).flatten()
        val_idx=numpy.argwhere(mask == False).flatten()
        with open(dire+"/results.txt", 'a') as f:
            sys.stdout = f
            print("Validation iteration: ", i ,"GROUPS TO BE LEFT OUT: ", groups_to_leave_out)
            sys.stdout = original_stdout
        print("GROUPS TO BE LEFT OUT: ", groups_to_leave_out)
        print("Validation iteration: ", i, "/", n_iterations)
        
        
        confusion_table,confusion_table_of_run = logo_run(path=pklfile,logo_idx=logo_idx,val_idx=val_idx,confusion_table=confusion_table,confusion_table_of_run=confusion_table_of_run,dire=dire)
        
        accuracy,sensitivity,specificity,precision,f1=cm.calc_scores_from_confusionmatrix(confusion_table_of_run)
        with open(dire+"/overall_results.txt", 'a') as f:
            sys.stdout = f
            print("Validation iteration: ", i ,"GROUPS TO BE LEFT OUT: ", groups_to_leave_out)
            print('accuracy',accuracy)
            print('f1_score',f1)
            print('sensitivity',sensitivity)
            print('specificity',specificity)
            print('precision',precision)
            sys.stdout = original_stdout