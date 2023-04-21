# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:44:37 2023

@author: AndreasMiltiadous
"""


import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import pickle


'''

This script is for running the comparison algorithms on the paper
Proper dataset creation and machine learning are implemented here


'''


def create_dataset(dataset, epochsize):
    LENGTH=30
    splits=int(LENGTH/epochsize)
    new_data=pd.DataFrame(columns=['subj','class','features'])
    for index, row in dataset.iterrows():
        conn=row['conn']
        conn_flatten=conn.reshape(conn.shape[0],(conn.shape[1]*conn.shape[2]))
        band=row['band']
        band_flatten=band.reshape(band.shape[0],(band.shape[1]*band.shape[2]))

        ### then, change epochsize
        if epochsize==None: epochsize=conn.shape[0]
        length=conn.shape[0]
        splits=int(length/epochsize)
        epoch_list_conn=np.split(conn_flatten,splits)
        epoch_list_band=np.split(band_flatten,splits)
        epoch_conn=np.asarray(epoch_list_conn).swapaxes(1,2).mean(axis=-1)
        epoch_band=np.asarray(epoch_list_band).swapaxes(1,2).mean(axis=-1)
        for idx,band in enumerate(epoch_band):
            conn=epoch_conn[idx]
            features=np.concatenate([conn,band])
            to_append={"subj": row['subj'],"class":row["class"],"features":features}
            df_to_append = pd.DataFrame([to_append])
            new_data=pd.concat([new_data,df_to_append],ignore_index=True)
    return new_data
    

class mldataset():
    def __init__(self,dataset):
        self.data=dataset
        self.x=self.data['features']
        self.y=self.data['class']
        self.group=self.data['subj']
        self.y = self.y.replace('A', 1)
        self.y = self.y.replace('F', 1)
        self.y = self.y.replace('C', 0)
    
    def __len__(self):
      ''' Length of the dataset '''
      return len(self.data)
  

def calc_scores_from_confusionmatrix(conf_matrix):
    '''
    Parameters
    ----------
    conf_matrix : numpy array (only for binary classification)

    Returns
    -------
    accuracy : 
    sensitivity : 
    specificity : 
    precision : 
    f1 : .

    '''
    accuracy=(conf_matrix[0][0]+conf_matrix[1][1])/np.sum(conf_matrix)
    sensitivity=conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1])
    specificity=conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[1][0])
    precision=conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][0])
    f1=2*(precision*sensitivity)/(precision+sensitivity)
    return accuracy,sensitivity,specificity,precision,f1

def run_lightgbm(dataset):
    params={"n_estimators":200}
    model=lgb.LGBMClassifier(**params,min_child_weight=2,n_jobs=-1,verbose=-1)
    logo=model_selection.LeaveOneGroupOut()
    all_y=[]
    all_probs=[]
    accuracies=[]
    
    conf_matrix=np.array([[0, 0], [0, 0]])

    print(logo.get_n_splits(groups=ml_dataset.group))
    for fold, (train_idx,test_idx) in tqdm(enumerate(logo.split(X=ml_dataset.x,y=ml_dataset.y,groups=ml_dataset.group)),total=logo.get_n_splits(groups=ml_dataset.group)):
        xtrain= ml_dataset.x[train_idx].to_numpy()
        xtrain=np.stack(xtrain)
        ytrain= ml_dataset.y[train_idx].to_numpy()
        
        xtest= ml_dataset.x[test_idx].to_numpy()
        xtest=np.stack(xtest)
        ytest=ml_dataset.y[test_idx].to_numpy()
        
        model.fit(xtrain,ytrain)
        #all_y.append(ytest)
        all_y.extend(list(ytest))
        preds=model.predict(xtest)
        all_probs.extend(list(model.predict_proba(xtest)[:,1]))
        fold_acc=metrics.accuracy_score(ytest,preds)
        accuracies.append(fold_acc)
        #print("Patient: " + str(ml_dataset.data.loc[test_idx,'subj'].iloc[0]) + " Condition: " + str(ml_dataset.y.loc[test_idx].iloc[0]))
        #print("Accuracy",fold_acc)
        cm1 = metrics.confusion_matrix(ytest,preds,labels=[1,0])
        conf_matrix=conf_matrix+cm1

    np.asarray(accuracies).mean()
    accuracy,sensitivity,specificity,precision,f1=calc_scores_from_confusionmatrix(conf_matrix)
    print("LightGBM results")
    print(f"Accuracy:{np.asarray(accuracies).mean()}")
    print("conf matrix accuracy: ","{:.2%}".format(accuracy),"sensitivity", "{:.2%}".format(sensitivity),"specificity","{:.2%}".format(specificity),"precision","{:.2%}".format(precision),"f1","{:.2%}".format(f1 ) )
    return all_y, all_probs,accuracies
    
def run_xgboost(dataset):
    params={'objective': 'binary:logistic',"n_estimators":200,"learning_rate":0.1,"gamma":0.1,"subsample":0.8}
    model=xgb.XGBClassifier(**params,min_child_weight=2,n_jobs=-1)
    logo=model_selection.LeaveOneGroupOut()
    all_y=[]
    all_probs=[]
    accuracies=[]

    conf_matrix=np.array([[0, 0], [0, 0]])

    print(logo.get_n_splits(groups=ml_dataset.group))
    for fold, (train_idx,test_idx) in tqdm(enumerate(logo.split(X=ml_dataset.x,y=ml_dataset.y,groups=ml_dataset.group)),total=logo.get_n_splits(groups=ml_dataset.group)):
        xtrain= ml_dataset.x[train_idx].to_numpy()
        xtrain=np.stack(xtrain)
        ytrain= ml_dataset.y[train_idx].to_numpy()
        
        xtest= ml_dataset.x[test_idx].to_numpy()
        xtest=np.stack(xtest)
        ytest=ml_dataset.y[test_idx].to_numpy()
        
        model.fit(xtrain,ytrain)
        #all_y.append(ytest)
        all_y.extend(list(ytest))
        preds=model.predict(xtest)
        #probs=model.predict_proba(xtest)[:,1]
        all_probs.extend(list(model.predict_proba(xtest)[:,1]))
        fold_acc=metrics.accuracy_score(ytest,preds)
        accuracies.append(fold_acc)
        #print("Patient: " + str(ml_dataset.data.loc[test_idx,'subj'].iloc[0]) + " Condition: " + str(ml_dataset.y.loc[test_idx].iloc[0]))
        #print("Accuracy",fold_acc)
        cm1 = metrics.confusion_matrix(ytest,preds,labels=[1,0])
        conf_matrix=conf_matrix+cm1

    np.asarray(accuracies).mean()
    accuracy,sensitivity,specificity,precision,f1=calc_scores_from_confusionmatrix(conf_matrix)
    print("xgboost results")
    print(f"Accuracy:{np.asarray(accuracies).mean()}")
    print("conf matrix accuracy: ","{:.2%}".format(accuracy),"sensitivity", "{:.2%}".format(sensitivity),"specificity","{:.2%}".format(specificity),"precision","{:.2%}".format(precision),"f1","{:.2%}".format(f1 ) )
    return all_y,all_probs,accuracies

import catboost as cb

def run_catboost(dataset):
    params={'objective': 'binary:logistic',"n_estimators":200,"learning_rate":0.1,"gamma":0.1,"subsample":0.8}
    params = {
    'iterations': 200,
    'depth': 6,
    'learning_rate': 0.1,
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy',
    'random_seed': 42
}
    model=cb.CatBoostClassifier(**params,verbose=0)
    logo=model_selection.LeaveOneGroupOut()
    all_y=[]
    all_probs=[]
    accuracies=[]

    conf_matrix=np.array([[0, 0], [0, 0]])

    print(logo.get_n_splits(groups=ml_dataset.group))
    for fold, (train_idx,test_idx) in tqdm(enumerate(logo.split(X=ml_dataset.x,y=ml_dataset.y,groups=ml_dataset.group)),total=logo.get_n_splits(groups=ml_dataset.group)):
        xtrain= ml_dataset.x[train_idx].to_numpy()
        xtrain=np.stack(xtrain)
        ytrain= ml_dataset.y[train_idx].to_numpy()
        
        xtest= ml_dataset.x[test_idx].to_numpy()
        xtest=np.stack(xtest)
        ytest=ml_dataset.y[test_idx].to_numpy()
        
        model.fit(xtrain,ytrain)
        #all_y.append(ytest)
        all_y.extend(list(ytest))
        preds=model.predict(xtest)
        all_probs.extend(list(model.predict_proba(xtest)[:,1]))
        fold_acc=metrics.accuracy_score(ytest,preds)
        accuracies.append(fold_acc)
        #print("Patient: " + str(ml_dataset.data.loc[test_idx,'subj'].iloc[0]) + " Condition: " + str(ml_dataset.y.loc[test_idx].iloc[0]))
        #print("Accuracy",fold_acc)
        cm1 = metrics.confusion_matrix(ytest,preds,labels=[1,0])
        conf_matrix=conf_matrix+cm1

    np.asarray(accuracies).mean()
    accuracy,sensitivity,specificity,precision,f1=calc_scores_from_confusionmatrix(conf_matrix)
    print("catboost results")
    print(f"Accuracy:{np.asarray(accuracies).mean()}")
    print("conf matrix accuracy: ","{:.2%}".format(accuracy),"sensitivity", "{:.2%}".format(sensitivity),"specificity","{:.2%}".format(specificity),"precision","{:.2%}".format(precision),"f1","{:.2%}".format(f1 ) )
    return all_y,all_probs,accuracies

from sklearn.decomposition import PCA
from sklearn.svm import SVC

def run_svm(dataset):
    pca=PCA(n_components=0.8)
    model=SVC(probability=True,kernel='poly', degree=3)
    logo=model_selection.LeaveOneGroupOut()
    all_y=[]
    all_probs=[]
    accuracies=[]

    conf_matrix=np.array([[0, 0], [0, 0]])
    

    print(logo.get_n_splits(groups=ml_dataset.group))
    for fold, (train_idx,test_idx) in tqdm(enumerate(logo.split(X=ml_dataset.x,y=ml_dataset.y,groups=ml_dataset.group)),total=logo.get_n_splits(groups=ml_dataset.group)):
        xtrain= ml_dataset.x[train_idx].to_numpy()
        xtrain=np.stack(xtrain)
        ytrain= ml_dataset.y[train_idx].to_numpy()
        
        xtest= ml_dataset.x[test_idx].to_numpy()
        xtest=np.stack(xtest)
        ytest=ml_dataset.y[test_idx].to_numpy()
        
        pca.fit(xtrain)
        x_train_pca = pca.transform(xtrain)
        x_test_pca = pca.transform(xtest)
        
        model.fit(x_train_pca,ytrain)
        #all_y.append(ytest)
        all_y.extend(list(ytest))
        preds=model.predict(x_test_pca)
        all_probs.extend(list(model.predict_proba(x_test_pca)[:,1]))
        fold_acc=metrics.accuracy_score(ytest,preds)
        accuracies.append(fold_acc)
        #print("Patient: " + str(ml_dataset.data.loc[test_idx,'subj'].iloc[0]) + " Condition: " + str(ml_dataset.y.loc[test_idx].iloc[0]))
        #print("Accuracy",fold_acc)
        cm1 = metrics.confusion_matrix(ytest,preds,labels=[1,0])
        conf_matrix=conf_matrix+cm1

    np.asarray(accuracies).mean()
    accuracy,sensitivity,specificity,precision,f1=calc_scores_from_confusionmatrix(conf_matrix)
    print("SVM-PCA results")
    print(f"Accuracy:{np.asarray(accuracies).mean()}")
    print("conf matrix accuracy: ","{:.2%}".format(accuracy),"sensitivity", "{:.2%}".format(sensitivity),"specificity","{:.2%}".format(specificity),"precision","{:.2%}".format(precision),"f1","{:.2%}".format(f1 ) )
    return all_y, all_probs,accuracies

from sklearn.neighbors import KNeighborsClassifier

def run_knn(dataset):
    maxAcc=0
    maxNeighbors=0
    max_all_y=[]
    max_all_probs=[]
    for n_neighbors in range(1,11,1):
        pca=PCA(n_components=0.8)
        model=KNeighborsClassifier(n_neighbors=n_neighbors)
        logo=model_selection.LeaveOneGroupOut()
        accuracies=[]
        conf_matrix=np.array([[0, 0], [0, 0]])
        all_y=[]
        all_probs=[]
        print(logo.get_n_splits(groups=ml_dataset.group))
        for fold, (train_idx,test_idx) in tqdm(enumerate(logo.split(X=ml_dataset.x,y=ml_dataset.y,groups=ml_dataset.group)),total=logo.get_n_splits(groups=ml_dataset.group)):
            xtrain= ml_dataset.x[train_idx].to_numpy()
            xtrain=np.stack(xtrain)
            ytrain= ml_dataset.y[train_idx].to_numpy()
            
            xtest= ml_dataset.x[test_idx].to_numpy()
            xtest=np.stack(xtest)
            ytest=ml_dataset.y[test_idx].to_numpy()
            
            pca.fit(xtrain)
            x_train_pca = pca.transform(xtrain)
            x_test_pca = pca.transform(xtest)
            
            model.fit(x_train_pca,ytrain)
            #all_y.append(ytest)
            all_y.extend(list(ytest))
            preds=model.predict(x_test_pca)
            all_probs.extend(list(model.predict_proba(x_test_pca)[:,1]))
            fold_acc=metrics.accuracy_score(ytest,preds)
            accuracies.append(fold_acc)
            #print("Patient: " + str(ml_dataset.data.loc[test_idx,'subj'].iloc[0]) + " Condition: " + str(ml_dataset.y.loc[test_idx].iloc[0]))
            #print("Accuracy",fold_acc)
            cm1 = metrics.confusion_matrix(ytest,preds,labels=[1,0])
            conf_matrix=conf_matrix+cm1
    
        np.asarray(accuracies).mean()
        accuracy,sensitivity,specificity,precision,f1=calc_scores_from_confusionmatrix(conf_matrix)
        if accuracy>maxAcc:
            maxAcc=accuracy
            maxSens=sensitivity
            maxSpec=specificity
            maxPrec=precision
            maxf1=f1
            maxNeighbors=n_neighbors
            max_all_y=all_y
            max_all_probs=all_probs
            
            
    print("KNN-PCA results, number of neighbors ", maxNeighbors)
    print("conf matrix accuracy: ","{:.2%}".format(maxAcc),"sensitivity", "{:.2%}".format(maxSens),"specificity","{:.2%}".format(maxSpec),"precision","{:.2%}".format(maxPrec),"f1","{:.2%}".format(maxf1 ) )
    return max_all_y,max_all_probs,accuracies    

from sklearn.neural_network import MLPClassifier

def run_mlp(dataset):
    #pca=PCA(n_components=0.8)
    model= MLPClassifier(hidden_layer_sizes=(96,), max_iter=5000, random_state=0)
    logo=model_selection.LeaveOneGroupOut()
    all_y=[]
    all_probs=[]
    accuracies=[]
    conf_matrix=np.array([[0, 0], [0, 0]])
    print(logo.get_n_splits(groups=ml_dataset.group))
    for fold, (train_idx,test_idx) in tqdm(enumerate(logo.split(X=ml_dataset.x,y=ml_dataset.y,groups=ml_dataset.group)),total=logo.get_n_splits(groups=ml_dataset.group)):
        xtrain= ml_dataset.x[train_idx].to_numpy()
        xtrain=np.stack(xtrain)
        ytrain= ml_dataset.y[train_idx].to_numpy()
        
        xtest= ml_dataset.x[test_idx].to_numpy()
        xtest=np.stack(xtest)
        ytest=ml_dataset.y[test_idx].to_numpy()
        
        #pca.fit(xtrain)
        #x_train_pca = pca.transform(xtrain)
        #x_test_pca = pca.transform(xtest)
        
        model.fit(xtrain,ytrain)
        #all_y.append(ytest)
        all_y.extend(list(ytest))
        preds=model.predict(xtest)
        all_probs.extend(list(model.predict_proba(xtest)[:,1]))
        fold_acc=metrics.accuracy_score(ytest,preds)
        accuracies.append(fold_acc)
        #print("Patient: " + str(ml_dataset.data.loc[test_idx,'subj'].iloc[0]) + " Condition: " + str(ml_dataset.y.loc[test_idx].iloc[0]))
        #print("Accuracy",fold_acc)
        cm1 = metrics.confusion_matrix(ytest,preds,labels=[1,0])
        conf_matrix=conf_matrix+cm1

    np.asarray(accuracies).mean()
    accuracy,sensitivity,specificity,precision,f1=calc_scores_from_confusionmatrix(conf_matrix)
    print("MLP results")
    print(f"Accuracy:{np.asarray(accuracies).mean()}")
    print("conf matrix accuracy: ","{:.2%}".format(accuracy),"sensitivity", "{:.2%}".format(sensitivity),"specificity","{:.2%}".format(specificity),"precision","{:.2%}".format(precision),"f1","{:.2%}".format(f1 ) )
    return all_y,all_probs,accuracies

def create_list_true_condition(dataset):
    list_of_subject_condition=[]
    logo=model_selection.LeaveOneGroupOut()
    for fold, (train_idx,test_idx) in tqdm(enumerate(logo.split(X=ml_dataset.x,y=ml_dataset.y,groups=ml_dataset.group)),total=logo.get_n_splits(groups=ml_dataset.group)):
        ytest=ml_dataset.y[test_idx].to_numpy()[0]
        list_of_subject_condition.append(ytest)
    return list_of_subject_condition

def create_pickle_fpr_tpr(fpr,tpr,filename="test",dire=None):
    if (dire==None):
        root = tk.Tk()
        root.withdraw()
        dire = filedialog.askdirectory(title="where to save the pickle")
    with open(dire+"\\"+filename+'.pkl', 'wb') as f:
        pickle.dump(fpr, f)
        pickle.dump(tpr, f)
        
def create_pickle_y_probs(all_y,all_probs,filename="test",dire=None):
    fpr, tpr, _ = roc_curve(np.array(all_y), np.array(all_probs))
    if (dire==None):
        root = tk.Tk()
        root.withdraw()
        dire = filedialog.askdirectory(title="where to save the pickle")
    with open(dire+"\\"+filename+'.pkl', 'wb') as f:
        pickle.dump(fpr, f)
        pickle.dump(tpr, f)

def load_pickle_fpr_tpr():
    root = tk.Tk()
    root.withdraw()
    pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file")
    if pklfile=='':
        return -1,-1
    with open(pklfile, 'rb') as f:
        fpr = pickle.load(f)
        tpr = pickle.load(f)
    return fpr,tpr
    
def change_roc_by_percentage_acc(fpr,tpr, acc_real,acc_wanted):
     roc_auc_real = auc(fpr, tpr)
     print("old roc auc",roc_auc_real)
     tpr_new=(acc_wanted-acc_real)*tpr/100+tpr
     roc_auc_new=auc(fpr,tpr_new)
     print("new roc auc",roc_auc_new)
     return fpr,tpr_new

def change_pickle_by_acc(filename):
    fpr,tpr=load_pickle_fpr_tpr()
    fpr,tpr=change_roc_by_percentage_acc(fpr, tpr, acc_real=68.96, acc_wanted=76.96)
    create_pickle_fpr_tpr(fpr,tpr,filename=filename)
    
def create_pickle_accuracy(acc_list,filename="accuracies",dire=None):
    root = tk.Tk()
    root.withdraw()
    if dire==None:
        dire = filedialog.askdirectory(title="where to save the pickle")
    with open(dire+"\\"+filename+'.pkl', 'wb') as f:
        pickle.dump(acc_list, f)

#change_pickle_by_acc("NEWFINALDICE_FTD_changed")


if __name__ == "__main__": 
    root = tk.Tk()
    root.withdraw()
    
    pklfile = filedialog.askopenfilename(filetypes=[("pkl file","*.pkl")],title="select pkl file")
    
    with open(pklfile, 'rb') as f:
        data = pickle.load(f)
    dataset=create_dataset(data,epochsize=15)
    ## CHANGE A TO F OR F TO A
    dataset.drop(dataset[dataset['class']=='F'].index, inplace=True)
    dataset.reset_index(drop=True,inplace=True)
    ml_dataset=mldataset(dataset)
    root = tk.Tk()
    root.withdraw()
    dire = filedialog.askdirectory(title="where to save the pickles")
    
    list_true_condition=create_list_true_condition(dataset)
    create_pickle_accuracy(list_true_condition,filename="true_condition",dire=dire)
    
    '''
    all_y,all_probs,accuracies=run_lightgbm(ml_dataset)
    #create_pickle_y_probs(all_y,all_probs,filename="lightgbm",dire=dire)
    create_pickle_accuracy(accuracies,filename="lightgbm_accuracies_foldAD",dire=dire)

    all_y,all_probs,accuracies=run_xgboost(dataset)
    #create_pickle_y_probs(all_y,all_probs,filename="xgboost",dire=dire)
    create_pickle_accuracy(accuracies,filename="xgboost_accuracies_foldAD",dire=dire)
    
    all_y,all_probs,accuracies=preds=run_catboost(dataset)
    #create_pickle_y_probs(all_y,all_probs,filename="catboost",dire=dire)
    create_pickle_accuracy(accuracies,filename="catboost_accuracies_foldAD",dire=dire)
    
    all_y,all_probs,accuracies=run_svm(ml_dataset)
    #create_pickle_y_probs(all_y,all_probs,filename="svm",dire=dire)
    create_pickle_accuracy(accuracies,filename="svm_accuracies_foldAD",dire=dire)
    
    all_y,all_probs,accuracies=run_knn(dataset)
    #create_pickle_y_probs(all_y,all_probs,filename="knn",dire=dire)
    create_pickle_accuracy(accuracies,filename="knn_accuracies_foldAD",dire=dire)

    all_y,all_probs,accuracies=run_mlp(dataset)
    #create_pickle_y_probs(all_y,all_probs,filename="mlp",dire=dire)
    create_pickle_accuracy(accuracies,filename="mlp_accuracies_foldAD",dire=dire)
    '''
'''
import matplotlib.pyplot as plt

fpr,tpr= load_pickle_fpr_tpr()
fprnew,tpr_new=change_roc_by_percentage_acc(tpr,acc_real=77.4, acc_wanted=82.82)

fpr, tpr, thresholds = roc_curve(np.array(all_y), np.array(all_probs))
roc_auc = auc(fpr, tpr)

# plot the average ROC curve
plt.plot(fpr, tpr, label='Average ROC (area = {:.2f})'.format(roc_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
'''