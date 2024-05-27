# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:07:30 2023

@author: dell
"""

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import scale

def calcE(X,coli,colj):
    sum1 = np.sum((X[:,coli]-X[:,colj])**2)
    return math.sqrt(sum1)

def Euclidean(X,n):
    Euclideandata=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            Euclideandata[i,j]=calcE(X,i,j)
            Euclideandata[j,i]=Euclideandata[i,j]
    Euclidean_distance=[]

    for i in range(n):
        sum1 = np.sum(Euclideandata[i,:])
        Euclidean_distance.append(sum1/n)
    return Euclidean_distance

def varience(data,avg1,col1,avg2,col2):
    return np.average((data[:,col1]-avg1)*(data[:,col2]-avg2))

def Person(X,y,n):
    feaNum=n
    #label_num=len(y[0,:])
    label_num=1
    PersonData=np.zeros([n])
    for i in range(feaNum):
        for j in range(feaNum,feaNum+label_num):
            #print('. ', end='')
            average1 = np.average(X[:,i])
            average2 = np.average(y)
            yn=(X.shape)[0]
            y=y.reshape((yn,1))
            dataset = np.concatenate((X,y),axis=1)
            numerator = varience(dataset, average1, i, average2, j);
            denominator = math.sqrt(
                varience(dataset, average1, i, average1, i) * varience(dataset, average2, j, average2, j));
            if (abs(denominator) < (1E-10)):
                PersonData[i]=0
            else:
                PersonData[i]=abs(numerator/denominator)
    return list(PersonData)

def mrmd(X,y,n_selected_features=200):
    n=X.shape[1]
    e=Euclidean(X,n)
    p = Person(X,y,n)
    mrmrValue=[]
    for i,j in zip(p,e):
        mrmrValue.append(i+j)
    mrmr_max=max(mrmrValue)
    features_name=np.array(range(n))
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name,mrmrValue)]
    mrmd_order=sorted(mrmrValue,key=lambda x:x[1],reverse=True)
    mrmd_order =[int(x[0]) for x in mrmd_order]
    mrmd_end=mrmd_order[:n_selected_features]
    return mrmd_end

data_train=pd.read_csv(r'D:\PycharmProjects\my\feature extraction\RBPhunhe2.csv')
data_=np.array(data_train)
data=data_[:,:]
[m1,n1]=np.shape(data)
label1=np.ones((int(2616),1))#Value can be changed
label2=np.zeros((int(4175),1))
# label1=np.ones((int(7129),1))#Value can be changed
# label2=np.zeros((int(7060),1))
label=np.append(label1,label2)
shu=scale(data)
feature=pd.DataFrame(shu)
X=np.array(feature)
y=label.astype('int64')
MRMDresult=mrmd(X,y,n_selected_features=100)
MRMDresult = feature[feature.columns[MRMDresult]]
MRMDresult.to_csv("RBP_MRMD.csv")