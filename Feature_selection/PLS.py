import pandas as pd
from sklearn.decomposition import KernelPCA,FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

file_path = r''  # 请替换为您的 CSV 文件路径
data = pd.read_csv(file_path)
data1=np.array(data)
data=data1[:,:]
print(data.shape)
# Standardize the data (important for kPCA)
scaler = StandardScaler()
X_std = scaler.fit_transform(data)
# label1=np.ones((int(7129),1))
# label2=np.zeros((int(7060),1))
# label3=np.ones((int(93),1))
# label4=np.zeros((int(93),1))
label1=np.ones((int(2616),1))
label2=np.zeros((int(4175),1))
label3=np.ones((int(966),1))
label4=np.zeros((int(597),1))
# y = np.append(label3, label4)
y = np.concatenate((label1, label2, label3, label4))
pls = PLSRegression(n_components=100)


pls.fit(X_std, y)


X_transformed = pls.transform(X_std)


# Create a new DataFrame with the kPCA results
kpca_df = pd.DataFrame(data=X_transformed)

# Save the DataFrame to a new CSV file
kpca_df.to_csv( '', index=False)
