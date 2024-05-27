# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 20:14:46 2021

@author: 菜菜
"""

from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
from module import GAT
import torch.nn.functional as F
import pandas as pd
import dgl
import dgl.nn as dglnn
import random
from sklearn.model_selection import KFold
import math
import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import interp
import os
import matplotlib.pyplot as plt
from pylab import *
# import utils.tools as utils
from sklearn.metrics import average_precision_score, precision_recall_curve

def get_shuffle(dataset,label):    
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label  

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-06)
    f1=float(tp*2)/(tp*2+fp+fn+1e-06)
    return acc, precision,npv, sensitivity, specificity, mcc, f1
def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def load_data():
    # data_=pd.read_csv(r'RBP_kPCA.csv')
    data_=pd.read_csv(r'RBP_KPLS_22.csv')
    # data_=pd.read_csv(r'/tmp/pycharm_project_900/DRBPPred-GAT/Feature_selection/PCA.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\PCA.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\GA.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\CT.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPaPred-GAT\Feature_extraction\NMBroto.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\GTPC.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\CTD.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\PseAAC.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\EBGW.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\MMI.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\PsePSSM.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_auto_new.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_LLE.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_SE.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_Elastic_net.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_LR.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_Lasso.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_ET.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_XGB.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\DDE.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\Kmer.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\RCKmer.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\MonoDiKGap.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\test_ALL_Auto.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_selection\ALL_auto_RN.csv')
    # data_=pd.read_csv(r'D:\PycharmProjects\DRBPPred-GAT\Feature_extraction\EBGW.csv')
    data1=np.array(data_)




    data=data1[:,:]
    [m1,n1]=np.shape(data)
    # label1=np.ones((int(3846),1))
    label1=np.ones((int(2616),1))
    label2=np.zeros((int(4175),1))
    # label1=np.ones((int(7129),1))
    # label2 = np.zeros((int(7060), 1))
    labels=np.append(label1,label2)
    shu=data
    X,y=get_shuffle(shu,labels)
    # print(X)
    features = torch.FloatTensor(X)
    y_tensor = torch.from_numpy(y)
    labels = torch.squeeze(y_tensor)
    g = dgl.knn_graph(features, 5, algorithm='bruteforce-blas', dist='cosine')
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    return g, features, labels

g, features, labels = load_data()
# adj = g.adjacency_matrix().to_dense()

sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

[sample_num,input_dim]=np.shape(features)
out_dim=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_cnn=[]
tprs_cnn = []
sepscore_cnn = []



# Kfold = KFold(n_splits = 10, random_state = False)
Kfold = KFold(n_splits = 10, random_state = None)
index = Kfold.split(X=features ,y=labels)
dur = []


for train_index,test_index in index:

    net = GAT(g,in_dim=features.size()[1],num_layers=3,
              num_hidden=16,num_classes=2,
              heads=torch.tensor([32,32,32]),activation=F.relu,
              feat_drop=0,attn_drop=0,
              negative_slope=0.1,                          #LeakyReLU角度  默认0.2
              residual=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # print(train_index)
    for epoch in range(180):
        t0 = time.time()

        logits = net(features)
        m = nn.LogSoftmax(dim=1)
        criteria = nn.NLLLoss()
        # loss = criteria(m(logits[train_index,:]), labels[train_index])
        loss = criteria(m(logits[train_index, :]), labels[train_index].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#        if epoch >= 3:
        dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(epoch, loss.item(), np.mean(dur)))
        
    a=logits[test_index]
    probas = F.softmax(a, dim=1)
    # Assuming `probas` is your PyTorch tensor
    probas_detached = probas.detach().numpy()
    y_class = categorical_probas_to_classes(probas_detached)

    # y_class= utils.categorical_probas_to_classes(probas)
    y_test=to_categorical(labels[test_index])#generate the test
    # print(y_test.shape)
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=labels[test_index]  
    # yscore=np.vstack((yscore,probas))
    yscore=np.vstack((yscore,probas.detach().numpy()))

    acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class,labels[test_index])
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(labels[test_index], probas[:, 1].detach().numpy())
    tprs_cnn.append(interp(mean_fpr, fpr, tpr))
    tprs_cnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aupr = average_precision_score(labels[test_index], probas[:, 1].detach().numpy())
    sepscore_cnn.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc,aupr])
    print('NB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f,aupr=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc,aupr))
scores=np.array(sepscore_cnn)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
print("aupr=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[8]*100,np.std(scores, axis=0)[8]*100))
row=ytest.shape[0]
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum2_GAT_KPLS.csv')
yscore_=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore_)
yscore_sum.to_csv('yscore_sum2_GAT_KPLS.csv')

scores=np.array(sepscore_cnn)
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscore_cnn.append(H1)
result=sepscore_cnn
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('GAT_RBP_KPLS.csv')

    
