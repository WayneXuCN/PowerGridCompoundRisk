import pandas as pd
import igraph as ig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# -------------------
# 1. 数据预处理
# -------------------
# 读取网络数据（与你提供的一致）
edge_file = "european_powergrids/powergridEU_E.csv"
coord_file = "european_powergrids/powergridEU_V.csv"
edges = pd.read_csv(edge_file, header=None).values.tolist()
coords = pd.read_csv(coord_file, header=None)
G = ig.Graph.TupleList(edges, directed=False)
G.vs["name"] = [int(v) for v in G.vs["name"]]
name_to_index = {v["name"]: idx for idx, v in enumerate(G.vs)}
latitudes = [None] * G.vcount()
longitudes = [None] * G.vcount()
for _, row in coords.iterrows():
    node_id = int(row[0])
    if node_id in name_to_index:
        idx = name_to_index[node_id]
        latitudes[idx] = row[1]
        longitudes[idx] = row[2]
G.vs["latitude"] = latitudes
G.vs["longitude"] = longitudes

# 转换为 PyTorch Geometric 格式
edge_index = torch.tensor(G.get_edgelist(), dtype=torch.long).t().contiguous()  # [2, 16922]
x = torch.tensor([[lat, lon] for lat, lon in zip(latitudes, longitudes)], dtype=torch.float)  # [13478, 2]

# 计算节点间地理距离矩阵（用于动态传播因子）
distances = np.zeros((G.vcount(), G.vcount()))
for i in range(G.vcount()):
    for j in range(i + 1, G.vcount()):
        lat1, lon1 = latitudes[i], longitudes[i]
        lat2, lon2 = latitudes[j], longitudes[j]
        dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # 简化为欧几里得距离（km）
        distances[i, j] = dist
        distances[j, i] = dist

# 模拟气候风险特征（洪水、热浪、干旱）
climate_features = []
for lat, lon in zip(latitudes, longitudes):
    flood_risk = max(0, (55 - lat) / 10)  # 南部洪水风险高
    heat_risk = max(0, (lat - 40) / 15)   # 中部热浪风险高
    drought_risk = max(0, (lon + 10) / 20) # 东部干旱风险高
    climate_features.append([flood_risk, heat_risk, drought_risk])
climate_tensor = torch.tensor(climate_features, dtype=torch.float)  # [13478, 3]

# 合并特征
x = torch.cat([x, climate_tensor], dim=1)  # [13478, 5]（纬度、经度、洪水、热浪、干旱）

# 生成一阶和二阶邻接矩阵
adj_matrix = np.array(G.get_adjacency().data)  # [13478, 13478]
adj_matrix_2 = np.dot(adj_matrix, adj_matrix) * (adj_matrix > 0)  # 二阶邻接（三角形关系）

# 创建 PyTorch Geometric 数据对象
data = Data(x=x, edge_index=edge_index)
print("Data Prepared: Nodes:", data.num_nodes, "Edges:", data.num_edges)

# -------------------
# 2. 图神经网络（GNN）建模
# -------------------
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)  # 输出失效概率 [0,1]

# 初始化 GNN 模型
model = GCN(input_dim=5, hidden_dim=16, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# 模拟训练标签（南部 10% 节点因洪水失效）
labels = torch.zeros(data.num_nodes, dtype=torch.float)
southern_nodes = [i for i, lat in enumerate(latitudes) if lat < 45]
np.random.shuffle(southern_nodes)
fail_nodes = southern_nodes[:int(0.1 * len(southern_nodes))]
labels[fail_nodes] = 1
labels = labels.unsqueeze(1)  # [13478, 1]

# 训练 GNN
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)  # [13478, 1]
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 预测初始失效概率
model.eval()
with torch.no_grad():
    failure_probs = model(data).squeeze().numpy()  # [13478]
print("GNN Failure Probabilities (Top 5):", failure_probs[:5])

# -------------------
# 3. 动态系统建模与连锁效应模拟（创新点 1）
# -------------------
def dynamic_system(state, t, probs, climate_intensity, adj_matrix, adj_matrix_2, distances):
    S = state  # 节点状态 [13478]
    k = 0.1    # 失效速率
    r = 0.05   # 恢复速率
    sigma = 50 # 距离衰减参数 (km)
    w1, w2 = 0.7, 0.3  # 一阶和二阶权重

    # 计算动态传播因子（DPF）
    N1 = np.zeros_like(S)  # 一阶邻居影响
    N2 = np.zeros_like(S)  # 二阶三角形影响
    for i in range(len(S)):
        neighbors_1 = adj_matrix[i] > 0
        if np.sum(neighbors_1) > 0:
            N1[i] = np.sum((1 - S[neighbors_1]) * np.exp(-distances[i, neighbors_1] / sigma)) / np.sum(neighbors_1)
        neighbors_2 = adj_matrix_2[i] > 0
        if np.sum(neighbors_2) > 0:
            N2[i] = np.sum((1 - S[neighbors_2]) * (1 - S[neighbors_2])) / np.sum(neighbors_2)
    DPF = w1 * N1 + w2 * N2  # 动态传播因子

    # 状态变化率
    dSdt = -k * probs * climate_intensity * S + r * (1 - S) * (1 - DPF)
    return dSdt

# 定义单一灾害情景强度（72小时）
t = np.linspace(0, 72, 73)
flood_intensity = climate_tensor[:, 0].numpy() * (1 + t / 72 * 4)  # 南部水位升至 5m
heat_intensity = climate_tensor[:, 1].numpy() * (25 + t / 72 * 15) / 40  # 中部温度升至 40°C
drought_intensity = climate_tensor[:, 2].numpy() * (1 - t / 72)  # 东部降雨降至 0

# 初始状态：全正常
S0 = np.ones(data.num_nodes)

# 模拟单一情景
S_flood = odeint(dynamic_system, S0, t, args=(failure_probs, flood_intensity[-1], adj_matrix, adj_matrix_2, distances))
S_heat = odeint(dynamic_system, S0, t, args=(failure_probs, heat_intensity[-1], adj_matrix, adj_matrix_2, distances))
S_drought = odeint(dynamic_system, S0, t, args=(failure_probs, drought_intensity[-1], adj_matrix, adj_matrix_2, distances))

# 计算失效比例
failed_flood = np.mean(S_flood[-1] < 0.2)
failed_heat = np.mean(S_heat[-1] < 0.2)
failed_drought = np.mean(S_drought[-1] < 0.2)
print(f"Failed Nodes: Flood: {failed_flood:.3f}, Heat: {failed_heat:.3f}, Drought: {failed_drought:.3f}")

# -------------------
# 4. 复合风险建模（创新点 2）
# -------------------
class ARIN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ARIN, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.attn = nn.Linear(3 + 1, 1)  # 灾害强度 + 平均距离

    def forward(self, intensities, avg_dist):
        # intensities: [3, 73]（洪水、热浪、干旱强度随时间变化）
        # avg_dist: 平均地理距离标量
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long)  # 全连接图
        h = self.gcn(intensities, edge_index)  # [3, hidden_dim]

        # 计算注意力权重
        attn_input = torch.cat([intensities, avg_dist * torch.ones(3, intensities.shape[1])], dim=0)  # [4, 73]
        alpha = torch.sigmoid(self.attn(attn_input.t()).t())  # [3, 73]

        # 复合强度
        C_composite = torch.sum(alpha * intensities, dim=0)  # [73]
        return C_composite

# 计算受影响节点的平均距离（简化假设）
affected_nodes = failure_probs > 0.5
avg_dist = np.mean(distances[affected_nodes, :][:, affected_nodes]) if np.sum(affected_nodes) > 1 else 50.0
avg_dist_tensor = torch.tensor(avg_dist, dtype=torch.float)

# 复合情景：洪水+热浪
intensities = torch.stack([
    torch.tensor(flood_intensity[-1], dtype=torch.float),
    torch.tensor(heat_intensity[-1], dtype=torch.float),
    torch.tensor(drought_intensity[-1], dtype=torch.float) * 0  # 仅洪水+热浪
])  # [3, 13478]

# 初始化 ARIN 并计算复合强度
arin = ARIN(input_dim=data.num_nodes, hidden_dim=16)
C_composite = arin(intensities, avg_dist_tensor)  # [13478]

# 模拟复合情景
S_composite = odeint(dynamic_system, S0, t, args=(failure_probs, C_composite.numpy(), adj_matrix, adj_matrix_2, distances))
failed_composite = np.mean(S_composite[-1] < 0.2)
print(f"Failed Nodes (Composite Flood+Heat): {failed_composite:.3f}")

# -------------------
# 5. 结果分析与可视化
# -------------------
plt.figure(figsize=(10, 6))
plt.plot(t, np.mean(S_flood, axis=1), label="Flood")
plt.plot(t, np.mean(S_heat, axis=1), label="Heat")
plt.plot(t, np.mean(S_drought, axis=1), label="Drought")
plt.plot(t, np.mean(S_composite, axis=1), label="Composite (Flood+Heat)")
plt.xlabel("Time (hours)")
plt.ylabel("Average Node State")
plt.legend()
plt.title("Node State Evolution under Different Scenarios")
plt.show()

# 关键节点（失效概率 > 0.7 且最终状态 < 0.2）
critical_nodes = np.where((failure_probs > 0.7) & (S_composite[-1] < 0.2))[0]
print(f"Critical Nodes (Top 5): {critical_nodes[:5]}")