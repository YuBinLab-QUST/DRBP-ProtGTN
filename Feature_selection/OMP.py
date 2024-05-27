import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.linear_model import OrthogonalMatchingPursuit

#Lasso中的正交匹配追踪
def omp_omp(data,label,n_nonzero_coefs=100):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(data, label)
    coef = omp.coef_
    idx_r, = coef.nonzero()
    new_data=data[:,idx_r]
    return new_data,idx_r

data_input = pd.read_csv(r'D:\PycharmProjects\my\feature extraction\RBPhunhe2.csv')
data_ = np.array(data_input)
data = data_[:,:]
label1=np.ones((int(2616),1))#Value can be changed
label2=np.zeros((int(4175),1))
# label1=np.ones((int(7129),1))#Value can be changed
# label2=np.zeros((int(7060),1))
label=np.append(label1,label2)
Zongshu = scale(data)


new_RNA_data,index_RNA = omp_omp(Zongshu,label,n_nonzero_coefs=100)

data_new = np.hstack((new_RNA_data,))
optimal_RPI_features = pd.DataFrame(data=data_new)
optimal_RPI_features.to_csv('RBP_OMP.csv')


