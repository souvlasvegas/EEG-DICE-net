# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:31:57 2023

@author: AndreasMiltiadous
"""

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import KFold
import sys
import tkinter as tk
from tkinter import filedialog
import warnings
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy
import torch
import pickle

from tqdm import tqdm

import dice_models as DICE
import dataset
import utils
import train_eval_func as tef
import calc_metrics as cm


################################################################################
'''
This is the main function that runs the DICE deep-learning methodology.
Before executing, please read the code and stop in every line that says STOP HERE:
There, follow the instructions.

Upon execution, you will be asked to:
    1) Choose a .pkl file containing the input data in proper form (choose AlzheimerTrainingDatasetFixxed.pkl)
    2) Choose a directory to save the results in a .txt file (no need to create the txt file, it is automatically created)

'''
################################################################################

if __name__ == "__main__": 
    original_stdout = sys.stdout # Save a reference to the original standard output
    root = tk.Tk()
    root.withdraw()
    pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file")

    root = tk.Tk()
    root.withdraw()
    dire = filedialog.askdirectory(title="where to save the results")

    with open(pklfile, 'rb') as f:
      data = pickle.load(f)
      
    features = data[['subj', 'conn', 'band']].to_numpy()
    labels = data['class'].to_numpy()
    group=data['subj'].to_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    maxAcc=0
    maxEpochs=0
    fold_accuracies=[]
    
    #########################################
    #STOP HERE:
    #This for loop is for finding the best epoch size and VERY time consuming
    #If you want to run it for specific epoch number, for example 90 just make a 1-time loop etc. range(90,95,10)
    for epochs in tqdm(range(80,85,10)):
    #########################################
    
    
        k=10
        conv1outs=[]
        conv2outs=[]
        ## ROC CURVES
        all_y=[]
        all_probs=[]
        ####
        splits=KFold(n_splits=k,shuffle=True,random_state=34)
        logo = LeaveOneGroupOut()
        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[],
                   'train_f1': [], 'test_f1': []}
        confusion_table=numpy.array([[0, 0], [0, 0]])
        cmatrixes=[]
        dataset = dataset.mlcDataset(path=pklfile)
        print(logo.get_n_splits(groups=dataset.subj))
        with warnings.catch_warnings():

            ###################################################################
            #STOP HERE:
            # Comment out one of the 2 following for loops
            # Choose if you want Leave-One-Group-Out OR K-FOLD
            
            #LEAVE ONE OUT
            for fold, (train_idx,val_idx) in enumerate(logo.split(X=numpy.arange(len(dataset)),y=dataset.y,groups=dataset.subj)):
            
            #KFOLD
            #for fold, (train_idx,val_idx) in enumerate(splits.split(numpy.arange(len(dataset)))):
                tqdm.write("FOLD: "+ str(fold))

            ###################################################################
                
                utils.reproducability(seed=7)
                ###############################################################
                # STOP HERE:
                # Choose what model you want to run from dice_models.py
                model = DICE.Model_cls_late_concat()
                ###############################################################
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
                shuffle = True      # if you want to shuffle your data after each epoch
                drop_last = True   # the last batch (maybe) contains less samples than the batch_size maybe you do not want it's gradients
                num_workers = 0     # number of multi-processes data loading
                pin_memory = True   # enable fast data transfer to CUDA-enabled GPUs
                    # ---------- Properties ----------
                loader_params = {'batch_size': batch_size,
                                  'drop_last': drop_last,
                                  'num_workers': num_workers,
                                  'pin_memory': pin_memory}
                train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
                test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
                train_loader = torch.utils.data.DataLoader(dataset,sampler=train_sampler, **loader_params)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=test_sampler, drop_last=True)
                
                for epoch in range(epochs):
                    train_loss, train_acc, train_f1, train_rec, train_pre = tef.train_epoch(train_loader, optimizer, model, criterion)
                    
                    ## 
                    #print("train loss: ", train_loss, "train_acc: ",train_acc)
                    #test_loss, test_acc, test_f1, test_rec, test_pre, _ = eval_epoch(model, test_loader, criterion)
                    #print ("EPOCHS TEST: ",epochs ,"FOLD: ",fold,"epoch: ", epoch,"loss: ", test_loss,"acc: ", test_acc)
                    ##
                
                test_loss, test_acc, test_f1, test_rec, test_pre, conf_matrix = tef.eval_epoch(model, test_loader, criterion)
                y_true,y_pred_prob=tef.eval_epoch_ROC(model, test_loader, criterion)
                all_y.extend(y_true)
                all_probs.extend(y_pred_prob)
                print("test_acc: ",test_acc)
                
                ##calculate accuracies for each fold
                accuracy,_,_,_,_=cm.calc_scores_from_confusionmatrix(conf_matrix)
                fold_accuracies.append(accuracy)
                cmatrixes.append(conf_matrix)
                confusion_table = confusion_table + conf_matrix
                conv1,conv2=cm.get_weights(model)
                conv1outs.append(conv1)
                conv2outs.append(conv2)
                
        
        
        accuracy,sensitivity,specificity,precision,f1=cm.calc_scores_from_confusionmatrix(confusion_table)
        
        with open(dire+"/results_no_encoder_AD_kfold.txt", 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print ("EPOCHS USED: ", epochs)
            print('accuracy',accuracy)
            print('f1_score',f1)
            print('sensitivity',sensitivity)
            print('specificity',specificity)
            print('precision',precision)
            sys.stdout = original_stdout # Reset the standard output to its original value
        
        print ("EPOCHS USED: ", epochs)
        print('f1_score',f1)
        print('sensitivity',sensitivity)
        print('specificity',specificity)
        print('precision',precision)
        print('accuracy',accuracy)
        
        if accuracy>maxAcc:
            maxEpochs=epochs
            maxAcc=accuracy
            maxSens=sensitivity
            maxSpec=specificity
            maxPrec=precision
            maxF1=f1
        
        ######################################################################
        # STOP HERE:
        # This will print the ROC curve of every epoch size.
        # If you dont want it, comment out this block
        fpr, tpr, thresholds = roc_curve(numpy.array(all_y), numpy.array(all_probs))
        roc_auc = auc(fpr, tpr)
    
        # plot the average ROC curve
        plt.plot(fpr, tpr, label='Average ROC (area = {:.2f})'.format(roc_auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()
        ######################################################################
    
    with open(dire+"//results_no_encoder_AD_kfold.txt", 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print ("HIGHEST RESULTS for epochs: ", maxEpochs)
        print('MAX accuracy',maxAcc)
        print('MAX f1_score',maxF1)
        print('MAX sensitivity',maxSens)
        print('MAX specificity',maxSpec)
        print('MAX precision',maxPrec)
        sys.stdout = original_stdout # Reset the standard output to its original value
        
    ###########################################################################
    #STOP HERE:
    # These lines are for creating pickle for saving the results, so as to be used in other functions
    # for visualization purposes for the paper.
    # Just leave them commented out if you dont know what they do
    
    
    #create_pickle_fpr_tpr(fpr,tpr,filename="2cls_late_concatFTD")
    #utils.create_pickle_accuracy(fold_accuracies,filename="DICE_AD_fold_accuracies")
    ###########################################################################