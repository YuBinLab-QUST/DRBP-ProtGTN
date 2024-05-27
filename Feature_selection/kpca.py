# #kpca
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import numpy as np

file_path = r'' 
data = pd.read_csv(file_path)
data1=np.array(data)
data=data1[:,:]
print(data.shape)
# Standardize the data (important for kPCA)
scaler = StandardScaler()
X_std = scaler.fit_transform(data)

# Apply kPCA
kpca = KernelPCA(n_components=100, kernel='rbf')  # You can choose the number of components and kernel type
X_kpca = kpca.fit_transform(X_std)

# Create a new DataFrame with the kPCA results
kpca_df = pd.DataFrame(data=X_kpca)

# # Concatenate the kPCA results with the original data (optional)
# result_df = pd.concat([data, kpca_df], axis=1)

# Save the DataFrame to a new CSV file
kpca_df.to_csv('', index=False)



