
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense,Input,Dropout
from keras.layers import Flatten
from keras.models import Model
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import scale
import utils.tools as utils
from keras.layers import Dense, merge,Input,Dropout
from keras.models import Model
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, average_precision_score
def to_class(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y
# Origanize data
def get_shuffle(dataset,label):    
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label 

data_=pd.read_csv(r'D:\PycharmProjects\my\feature selection\PDB14189_KPLS_22.csv')
data=np.array(data_)
data=data[:,:]
[m1,n1]=np.shape(data)
# label1=np.ones((int(3846),1))
# label1=np.ones((int(2616),1))
# label2=np.zeros((int(4175),1))
label1=np.ones((int(7129),1))
label2=np.zeros((int(7060),1))
#label1=np.ones((int(m1/2),1))
#label2=np.zeros((int(m1/2),1))
label=np.append(label1,label2)
X_=data
y_=label
X,y=get_shuffle(X_,y_)
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

def get_DNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(Dense(int(64), activation='relu', kernel_initializer='glorot_normal', name='High_dim_feature_1',input_shape=(1,input_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(int(64), activation='relu', kernel_initializer='glorot_normal', name='High_dim_feature_2'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(int(32), activation='relu', kernel_initializer='glorot_normal', name='High_dim_feature'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', name='output'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])#rmsprop
    return model

[sample_num,input_dim]=np.shape(X)
out_dim=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_rnn=[]
tprs_rnn = []
sepscore_rnn = []
skf= StratifiedKFold(n_splits=10)
for train, test in skf.split(X,y):
    clf_rnn = get_DNN_model(input_dim,out_dim)
    X_train_rnn=np.reshape(X[train],(-1,1,input_dim))
    X_test_rnn=np.reshape(X[test],(-1,1,input_dim))
    clf_list = clf_rnn.fit(X_train_rnn, to_categorical(y[train]),epochs=10)
    y_rnn_probas=clf_rnn.predict(X_test_rnn)
    probas_rnn.append(y_rnn_probas)
    y_class= utils.categorical_probas_to_classes(y_rnn_probas)
    
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]  
    yscore=np.vstack((yscore,y_rnn_probas))
    
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class,y[test])
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(y[test], y_rnn_probas[:, 1])
    tprs_rnn.append(interp(mean_fpr, fpr, tpr))
    tprs_rnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aupr = average_precision_score(y[test], y_rnn_probas[:, 1])
    sepscore_rnn.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc,aupr])
    print('NB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f,aupr=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc,aupr))
    clf_rnn=[]
    clf_list=[]
scores=np.array(sepscore_rnn)
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
ytest_sum.to_csv('ytest_sum_DNN_KPLS.csv')

yscore_=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore_)
yscore_sum.to_csv('yscore_sum_DNN_KPLS.csv')

scores=np.array(sepscore_rnn)
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscore_rnn.append(H1)
result=sepscore_rnn
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('DNN_PDB14189_KPLS.csv')
