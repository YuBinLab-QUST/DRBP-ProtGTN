import scipy.io as sio
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale,StandardScaler
from sklearn.metrics import roc_curve, auc
from dimensional_reduction1 import LLE
#import utils.tools as utils
data_=pd.read_csv(r'')
data=np.array(data_)
data=data[:,:]
print(data.shape)
[m1,n1]=np.shape(data)
label1=np.ones((int(2616),1))#Value can be changed
label2=np.zeros((int(4175),1))
# label1=np.ones((int(7129),1))#Value can be changed
# label2=np.zeros((int(7060),1))
label=np.append(label1,label2)
shu=scale(data)	
# new_X=LLE(shu,n_components=300)
new_X=LLE(shu,n_components=100)
shu=new_X
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('')
