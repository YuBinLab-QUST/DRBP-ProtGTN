# #kpca
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load data from CSV file
file_path = r'D:\PycharmProjects\my\feature extraction\RBPhunhe2.csv'  # 请替换为您的 CSV 文件路径
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
kpca_df.to_csv('RBP_kPCA.csv', index=False)



#MDS
# import pandas as pd
# from sklearn.manifold import MDS
# import matplotlib.pyplot as plt
#
# # 读取数据，假设数据保存在 data.csv 中
# data = pd.read_csv('ALL_protbert_DBP.csv')
#
# # 创建 MDS 模型，指定降维后的维度
# mds = MDS(n_components=200)
#
# # 对数据进行降维
# data_transformed = mds.fit_transform(data)
# kpca_df = pd.DataFrame(data=data_transformed)
#
# # # Concatenate the kPCA results with the original data (optional)
# # result_df = pd.concat([data, kpca_df], axis=1)
#
# # Save the DataFrame to a new CSV file
# kpca_df.to_csv('MDS_DBP.csv', index=False)
# # 可视化降维结果
# plt.scatter(data_transformed[:, 0], data_transformed[:, 1])
# plt.title('MDS Visualization')
# plt.show()
