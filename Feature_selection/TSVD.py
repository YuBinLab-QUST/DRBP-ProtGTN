import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD

def TSVD(data,n_components=300):
    svd = TruncatedSVD(n_components=n_components)
    new_data=svd.fit_transform(data)
    return new_data

data_input = pd.read_csv(r'')
data_ = np.array(data_input)
data = data_[:,:]
# label1=np.ones((int(7129),1))#Value can be changed
# label2=np.zeros((int(7060),1))
label3=np.ones((int(2616),1))#Value can be changed
label4=np.zeros((int(4175),1))
label=np.append(label3,label4)
Zongshu = scale(data)

new_RNA_data = TSVD(Zongshu,n_components=100)

data_new = np.hstack((new_RNA_data,))
optimal_RPI_features = pd.DataFrame(data=data_new)
optimal_RPI_features.to_csv('')

