# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:12:53 2023

@author: AndreasMiltiadous
"""

import numpy
import torch

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def metric(pred, true):
    y_pred_class = numpy.array((torch.sigmoid(pred)>0.5).type(torch.float).tolist())
    y_true = numpy.array(true.tolist())
    cm = confusion_matrix(y_true, y_pred_class,labels=[1,0])
    recall = recall_score(y_true, y_pred_class, zero_division=0)
    precision = precision_score(y_true, y_pred_class, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_class)
    #set zero_division=0
    f1 = f1_score(y_true, y_pred_class, zero_division=0)
    #report = classification_report(y_true, y_pred_class, zero_division=0)
    return recall, precision, f1, accuracy, cm

def metric_ROC(pred, true):
    y_pred_prob = numpy.array((torch.sigmoid(pred)).type(torch.float).tolist())
    y_true = numpy.array(true.tolist())
    #set zero_division=0
    return y_true,y_pred_prob

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
    accuracy=(conf_matrix[0][0]+conf_matrix[1][1])/numpy.sum(conf_matrix)
    sensitivity=conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1])
    specificity=conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[1][0])
    precision=conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][0])
    f1=2*(precision*sensitivity)/(precision+sensitivity)
    return accuracy,sensitivity,specificity,precision,f1

def get_weights(model):
    ##import dice_models ???needed ?
    conv1=model.depth_conv1.weight.detach().cpu().numpy()
    conv2=model.depth_conv2.weight.detach().cpu().numpy()
    return conv1,conv2

