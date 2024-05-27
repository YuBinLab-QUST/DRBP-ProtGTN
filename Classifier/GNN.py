# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:50:42 2023

@author: dell
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import time
from scipy import interp
# import utils.tools as utils
from sklearn.model_selection import StratifiedKFold
import argparse
import dgl
from sklearn.preprocessing import StandardScaler
#from layers import GraphConvolution
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import roc_curve, auc
import math
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
def get_shuffle(dataset,label):    
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label  

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y




data_=pd.read_csv(r'RBP_KPLS_22.csv')
data1=np.array(data_)
data=data1[:,:]
[m1,n1]=np.shape(data)
trans = StandardScaler()
data = trans.fit_transform(data)
# label1=np.ones((int(7129),1)) 
# label2=np.zeros((int(7060),1))
label1=np.ones((int(2616),1)) 
label2=np.zeros((int(4175),1))
labels=np.append(label1,label2)
shu=data
y=labels
X,y=get_shuffle(shu,labels)
features = torch.FloatTensor(X)
y = torch.tensor(y).long()
labels = torch.squeeze(y)


skf= StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)
dur = []

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)    #max(1)返回每一行中最大值的那个元素所构成的一维张量，且返回对应的一维索引张量（返回最大元素在这一行的列索引）
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# class GNN(Module):
#     def __init__(self, hidden_size, step=1):
#         super(GNN, self).__init__()
#         self.step = step
#         self.hidden_size = hidden_size
#         self.input_size = hidden_size * 2
#         self.gate_size = 3 * hidden_size
#         self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
#         self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
#         self.b_ih = Parameter(torch.Tensor(self.gate_size))
#         self.b_hh = Parameter(torch.Tensor(self.gate_size))
#         self.b_iah = Parameter(torch.Tensor(self.hidden_size))
#         self.b_oah = Parameter(torch.Tensor(self.hidden_size))

#         self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
#         self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

#     def GNNCell(self, A, hidden):
#         input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
#         input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
#         inputs = torch.cat([input_in, input_out], 2)
#         gi = F.linear(inputs, self.w_ih, self.b_ih)
#         gh = F.linear(hidden, self.w_hh, self.b_hh)
#         i_r, i_i, i_n = gi.chunk(3, 2)
#         h_r, h_i, h_n = gh.chunk(3, 2)
#         resetgate = torch.sigmoid(i_r + h_r)
#         inputgate = torch.sigmoid(i_i + h_i)
#         newgate = torch.tanh(i_n + resetgate * h_n)
#         hy = newgate + inputgate * (hidden - newgate)
#         return hy



#     def forward(self, A, hidden):
#         for i in range(self.step):
#             hidden = self.GNNCell(A, hidden)
#         return hidden

class GNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x
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

best_accuracy = 0  
sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
for train_index,test_index in skf.split(X,y):
    
     # net = GNN(hidden_size=32) 
     net = GNN(in_feats=features.shape[1], hidden_size=8, num_classes=2)
     optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
     #把训练集图表示
     features11 = torch.FloatTensor(X[train_index])
     g11 = dgl.knn_graph(features11, 5, algorithm='bruteforce', dist='cosine')

     #把验证集图表示
     features22 = torch.FloatTensor(X[test_index])
     g22 = dgl.knn_graph(features22, 5, algorithm='bruteforce', dist='cosine')

     for epoch in range(30):
         #训练模型，训练集用于训练    
         net.train()
         optimizer.zero_grad()
         t0 = time.time()
         logits = net(g11,features11)
         loss = nn.CrossEntropyLoss()
         logits = logits.float()
         loss_train =loss(logits, labels[train_index])
         acc_train = accuracy(logits, labels[train_index])
         loss_train.backward()
         optimizer.step()        
        # 验证模型，验证集用于验证，不参与参数更新
         net.eval()
         with torch.no_grad():
            logits22=net(g22,features22) 
            loss_val = loss(logits22, labels[test_index])
            acc_val = accuracy(logits22, labels[test_index])             
         dur.append(time.time() - t0)
         if acc_val > best_accuracy:
             best_accuracy = acc_val
     # net = torch.load('')
     with torch.no_grad():
         a=net(g22,features22)   
         probas = F.softmax(a, dim=1)   # 按行SoftMax,行和为1
         y_class = np.argmax(probas.detach().numpy(), axis=1)
         y_test=to_categorical(labels[test_index])#generate the test 
         ytest=np.vstack((ytest,y_test))
         y_test_tmp=labels[test_index]     
         # y_score=cv_clf.predict(X[test_index])#the output of  probability
         yscore=np.vstack((yscore,probas.detach().numpy()))
         acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class,labels[test_index])
         fpr, tpr, thresholds = roc_curve(labels[test_index], probas.detach().numpy()[:, 1])
         roc_auc = auc(fpr, tpr)
         aupr = average_precision_score(labels[test_index], probas.detach().numpy()[:, 1])
         sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc,aupr])
         print('NB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f,aupr=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc,aupr))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))
print("aupr=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[8]*100,np.std(scores, axis=0)[8]*100))
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)

result=sepscores
data_csv_zhibiao = pd.DataFrame(data=result)
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)

ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)

data_csv_zhibiao.to_csv('GNN_RBP_KPLS.csv')
yscore_sum.to_csv('yscore_sum2_GNN_KPLS.csv')
ytest_sum.to_csv('ytest_sum2_GNN_KPLS.csv')


fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='Ada ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(r'Curve_pr_auc1.png',format='png',dpi=600)
plt.show()