# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from sklearn.decomposition import FastICA, PCA
# # 生成观测模拟数据
# np.random.seed(0)
# n_samples = 2000
# time = np.linspace(0, 8, n_samples)
# s1 = np.sin(2 * time)  # 信号源 1 : 正弦信号
# s2 = np.sign(np.sin(3 * time))  # 信号源 2 : 方形信号
# s3 = signal.sawtooth(2 * np.pi * time)  # 信号源 3: 锯齿波信号
# S = np.c_[s1, s2, s3]
# S += 0.2 * np.random.normal(size=S.shape)  # 增加噪音数据
# S /= S.std(axis=0)  # 标准化
#
# # 混合数据
# A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # 混合矩阵
# X = np.dot(S, A.T)  # 生成观测信号源
#
# # ICA模型
# ica = FastICA(n_components=3)
# S_ = ica.fit_transform(X)  # 重构信号
# A_ = ica.mixing_  # 获得估计混合后的矩阵
#
# # PCA模型
# pca = PCA(n_components=3)
# H = pca.fit_transform(X)  # 基于PCA的成分正交重构信号源
#
# # 图形展示
# plt.figure()
# models = [X, S, S_, H]
# names = ['Observations (mixed signal)',
#          'True Sources',
#          'ICA recovered signals',
#          'PCA recovered signals']
# colors = ['red', 'steelblue', 'orange']
# for ii, (model, name) in enumerate(zip(models, names), 1):
#     plt.subplot(4, 1, ii)
#     plt.title(name)
#     for sig, color in zip(model.T, colors):
#         plt.plot(sig, color=color)
# plt.subplots_adjust(0, 0.1, 1.2, 1.5, 0.26, 0.46)
# plt.show()
#



import pandas as pd
from sklearn.decomposition import KernelPCA,FastICA
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

ica = FastICA(n_components=100)
X_kpca = ica.fit_transform(X_std)  # 重构信号
A_ = ica.mixing_  # 获得估计混合后的矩阵
# Create a new DataFrame with the kPCA results
kpca_df = pd.DataFrame(data=X_kpca)

# Save the DataFrame to a new CSV file
kpca_df.to_csv('RBP_ICA.csv', index=False)