from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
import pandas as pd
import dgl
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data


# 1. 从CSV文件加载蛋白质特征向量数据
csv_file = r'D:\PycharmProjects\my\feature extraction\RBPhunhe2.csv'  # 替换为你的CSV文件路径
data_df = pd.read_csv(csv_file)
data1=np.array(data_df)

data=data1[:,:]




k = 5  # 选择K值，即每个节点连接的最近邻节点数量
nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(data)
distances, indices = nbrs.kneighbors(data)


# 3. 创建一个图
graph = nx.Graph()
graph.add_nodes_from(range(len(data)))  # 添加节点

# 根据KNN结果，添加边
for i in range(len(data)):
    for j in indices[i]:
        if i != j:
            graph.add_edge(i, j)

# 4. 创建一个 PyTorch Geometric Data 对象
edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
data = Data(x=torch.tensor(data, dtype=torch.float32), edge_index=edge_index)




class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)



# parameters
out_channels = 100
num_features = 1522
epochs = 180

# 创建 GAE 模型
model = GAE(GCNEncoder(num_features, out_channels))

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)

# 训练循环
for epoch in range(epochs):
    loss = train()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss}')

# 获取编码后的低维表示
Z = model.encode(data.x, data.edge_index).detach().numpy()
df = pd.DataFrame(Z)
csv_file_path = 'RBP_GAE.csv'
df.to_csv(csv_file_path, index=False)



