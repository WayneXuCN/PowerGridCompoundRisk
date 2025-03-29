import torch
import torch.nn as nn
import torch.nn.functional as F


# HoSC 单层模块：高阶单纯形卷积
class HigherOrderSimplicialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels)  # 权重矩阵 Θ_H [in_channels, out_channels]
        self.bn = nn.BatchNorm1d(out_channels)            # 批量归一化层，稳定训练

    def forward(self, Z_H, L1_tilde):
        # Z_H [M, in_channels]：输入边嵌入
        # L1_tilde [M, M]：Hodge 1-Laplacian
        Z_theta = self.theta(Z_H)                     # 线性变换 [M, out_channels]
        Z_conv = torch.sparse.mm(L1_tilde, Z_theta) if L1_tilde.is_sparse else torch.mm(L1_tilde, Z_theta)  # 卷积 [M, out_channels]
        Z_psi = F.relu(self.bn(Z_conv))               # 非线性变换（BatchNorm + ReLU）[M, out_channels]
        Z_max, _ = torch.max(Z_psi, dim=1, keepdim=True)  # 特征维度最大值 [M, 1]
        return Z_max

# HoSC 模块：多层高阶单纯形卷积
class HoSC(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()                 # 存储多层 HoSC
        current_dim = input_dim                       # 初始输入维度
        for dim in hidden_dims:
            self.layers.append(HigherOrderSimplicialConv(current_dim, dim))  # 添加单层 [current_dim, dim]
            current_dim = 1                           # 每层输出固定为 1

    def forward(self, edge_attr, L1_tilde):
        # edge_attr [M, input_dim]：边特征
        # L1_tilde [M, M]：Hodge 1-Laplacian
        Z_list = []                                   # 存储各层输出
        Z_H = edge_attr                               # 初始边嵌入 [M, input_dim]
        for layer in self.layers:
            Z_H = layer(Z_H, L1_tilde)                # 单层 HoSC 输出 [M, 1]
            Z_list.append(Z_H)
        return torch.cat(Z_list, dim=1)               # 拼接所有层输出 [M, len(hidden_dims)]

# GNN 模块：节点特征提取
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()                 # 存储多层 GCN
        current_dim = input_dim                       # 初始输入维度
        for dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, dim))  # 线性层 [current_dim, dim]
            current_dim = dim                         # 更新维度

    def forward(self, X_n, A_tilde):
        # X_n [n, input_dim]：节点特征
        # A_tilde [n, n]：归一化邻接矩阵
        H = X_n                                       # 初始节点嵌入 [n, input_dim]
        for layer in self.layers:
            H = torch.mm(A_tilde, H)                  # 邻域聚合 [n, current_dim]
            H = F.relu(H @ layer.weight.T + layer.bias)  # 线性变换与激活 [n, next_dim]
        return H                                      # 最终节点嵌入 [n, hidden_dims[-1]]

# PPGN-HoSC 主模型
class PPGN_HoSC(nn.Module):
    def __init__(self, n_nodes, node_features, edge_features, node_hidden, edge_hidden):
        super().__init__()
        self.n_nodes = n_nodes                        # 节点数 n，例如 100
        self.gnn = GNN(node_features, node_hidden)    # GNN 模块，输入 node_features，输出 hidden_dims[-1]
        self.hosc = HoSC(edge_features, edge_hidden)  # HoSC 模块，输入 edge_features，输出 len(edge_hidden)
        # 全连接层
        self.fc1 = nn.Linear(node_hidden[-1] + len(edge_hidden), 2 * n_nodes)  # [d_{L_n} + L_e, 2n]
        self.fc2 = nn.Linear(2 * n_nodes, n_nodes)    # [2n, n]

    def forward(self, data):
        # 输入数据提取
        X_n = data.x                                  # 节点特征 [n, node_features]，例如 [100, 3]
        X_e = data.edge_attr                          # 边特征 [M, edge_features]，例如 [150, 1]
        A_tilde = data.A_tilde                        # 邻接矩阵 [n, n]，例如 [100, 100]
        L1_tilde = data.L1_tilde                      # Hodge 1-Laplacian [M, M]，例如 [150, 150]
        B1 = data.B1                                  # 关联矩阵 [n, M]，例如 [100, 150]

        # 节点特征提取
        H_n = self.gnn(X_n, A_tilde)                  # GNN 输出 [n, d_{L_n}]，例如 [100, 8]

        # 边特征与高阶拓扑建模
        Z_H = self.hosc(X_e, L1_tilde)                # HoSC 输出 [M, L_e]，例如 [150, 3]

        # 边到节点映射
        H_e = torch.sparse.mm(B1, Z_H) if B1.is_sparse else torch.mm(B1, Z_H)  # [n, L_e]，例如 [100, 3]

        # 特征融合
        H = torch.cat([H_n, H_e], dim=1)              # 拼接 [n, d_{L_n} + L_e]，例如 [100, 11]

        # 全局变换
        f = F.relu(self.fc1(H))                       # 第一层全连接 [n, 2n]，例如 [100, 200]
        f = self.fc2(f)                               # 第二层全连接 [n, n]，例如 [100, 100]

        # 概率输出
        z_p = F.sigmoid(f)                            # 独立概率 [n, n]，每个元素在 [0,1]
        return z_p[0]                                 # 单样本输出 [n]，例如 [100]


if __name__ == "__main__":ß
    n, M = 100, 150  # 节点数和边数
    data = type('Data', (), {
        'x': torch.randn(n, 3),          # 节点特征 [100, 3]
        'edge_attr': torch.randn(M, 1),  # 边特征 [150, 1]
        'A_tilde': torch.randn(n, n),    # 邻接矩阵 [100, 100]
        'L1_tilde': torch.randn(M, M),   # Hodge 1-Laplacian [150, 150]
        'B1': torch.randn(n, M)          # 关联矩阵 [100, 150]
    })()
    model = PPGN_HoSC(n_nodes=n, node_features=3, edge_features=1, node_hidden=[32, 16, 8], edge_hidden=[16, 8, 1])
    z_p = model(data)
    print(f"Output shape: {z_p.shape}")  # 输出形状：[100]
    print(f"Sample probabilities: {z_p[:5]}")  # 示例前五个概率
    print(f"Sum of probabilities: {z_p.sum().item()}")  # 概率总和（无约束）