
import pandas as pd
from sklearn.decomposition import KernelPCA,FastICA
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load data from CSV file
file_path = r'' 
data = pd.read_csv(file_path)
data1=np.array(data)
data=data1[:,:]
print(data.shape)
# Standardize the data (important for kPCA)
scaler = StandardScaler()
X_std = scaler.fit_transform(data)

ica = FastICA(n_components=100)
X_kpca = ica.fit_transform(X_std) 
A_ = ica.mixing_  
# Create a new DataFrame with the kPCA results
kpca_df = pd.DataFrame(data=X_kpca)

# Save the DataFrame to a new CSV file
kpca_df.to_csv('', index=False)
