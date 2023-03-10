# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:40:17 2023

@author: AndreasMiltiadous
"""

import numpy
import torch
from calc_metrics import metric, metric_ROC

device = "cuda" if torch.cuda.is_available() else "cpu"


def eval_epoch(model, dataloader, criterion):
    with torch.no_grad():   # disable gradient backpropagation
        model.eval()          # convert model parameters to evaluation mode
        losses = []
        recall_list = []
        precision_list = []
        f1_list = []
        accuracy_list = []
        conf_matrix=numpy.array([[0, 0], [0, 0]]) 
        for batch_index, (input1, input2, label, index) in enumerate(dataloader):
            input1 = input1.to(device)
            input2 = input2.to(device)
            label = label.to(device)
            out = model(input1, input2)
            loss = criterion(out.squeeze(), label)
            recall, precision, f1, accuracy, cm = metric(out.squeeze().to('cpu'), label.to('cpu'))
            conf_matrix=conf_matrix+cm
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            losses.append(loss.item())
    return round(numpy.mean(losses), 4), round(numpy.mean(accuracy_list), 2), round(numpy.mean(f1_list), 2), round(numpy.mean(recall_list), 2), round(numpy.mean(precision_list) ,2), conf_matrix
    # print('epoch', epoch, 'test loss', round(numpy.mean(losses), 4), 'acc', round(numpy.mean(accuracy_list), 2), 'f1', round(numpy.mean(f1_list), 2), 'rec', round(numpy.mean(recall_list), 2), 'pre', round(numpy.mean(precision_list) ,2))

##################################
# FUNCTIONALITY FOR ROC CURVE


def eval_epoch_ROC(model, dataloader, criterion):
    y_true=[]
    y_pred_prob=[]
    with torch.no_grad():   # disable gradient backpropagation
        model.eval()          # convert model parameters to evaluation mode
        for batch_index, (input1, input2, label, index) in enumerate(dataloader):
            input1 = input1.to(device)
            input2 = input2.to(device)
            label = label.to(device)
            out = model(input1, input2)
            y_true_npy,y_pred_prob_npy = metric_ROC(out.squeeze().to('cpu'), label.to('cpu'))
            y_true.extend(list(y_true_npy))
            y_pred_prob.extend(list(y_pred_prob_npy))
    return y_true,y_pred_prob
####################################

def train_epoch(dataloader, optimizer, model, criterion):
  model.train()            # train mode
  losses = []
  recall_list = []
  precision_list = []
  f1_list = []
  accuracy_list = []
  for batch_index, (input1, input2, label, index) in enumerate(dataloader):
      optimizer.zero_grad()
      input1 = input1.to(device)
      input2 = input2.to(device)
      label = label.to(device)
      out = model(input1, input2)
      loss = criterion(out.squeeze(), label)
      recall, precision, f1, accuracy, _ = metric(out.squeeze().to('cpu'), label.to('cpu'))
      recall_list.append(recall)
      precision_list.append(precision)
      f1_list.append(f1)
      accuracy_list.append(accuracy)
      loss.backward()
      optimizer.step()
      losses.append(loss.item())
  #print('epoch', epoch, 'loss', round(numpy.mean(losses), 4), 'acc', round(numpy.mean(accuracy_list), 2), 'f1', round(numpy.mean(f1_list), 2), 'rec', round(numpy.mean(recall_list), 2), 'pre', round(numpy.mean(precision_list) ,2))
  return round(numpy.mean(losses), 4), round(numpy.mean(accuracy_list), 2), round(numpy.mean(f1_list), 2), round(numpy.mean(recall_list), 2), round(numpy.mean(precision_list) ,2)

