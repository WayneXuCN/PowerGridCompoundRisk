from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from pathlib import Path

# HoSC 单层模块：高阶单纯形卷积
class HigherOrderSimplicialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, Z_H, L1_tilde):
        Z_theta = self.theta(Z_H)
        Z_conv = torch.sparse.mm(L1_tilde, Z_theta) if L1_tilde.is_sparse else torch.mm(L1_tilde, Z_theta)
        Z_psi = F.relu(self.bn(Z_conv))
        Z_max, _ = torch.max(Z_psi, dim=1, keepdim=True)
        return Z_max

# HoSC 模块：多层高阶单纯形卷积
class HoSC(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(HigherOrderSimplicialConv(current_dim, dim))
            current_dim = 1

    def forward(self, edge_attr, L1_tilde):
        Z_list = []
        Z_H = edge_attr
        for layer in self.layers:
            Z_H = layer(Z_H, L1_tilde)
            Z_list.append(Z_H)
        return torch.cat(Z_list, dim=1)

# GNN 模块：节点特征提取
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, dim))
            current_dim = dim

    def forward(self, X_n, A_tilde):
        H = X_n
        for layer in self.layers:
            H = torch.mm(A_tilde, H)
            H = F.relu(H @ layer.weight.T + layer.bias)
        return H

# HoT_GNN：多任务高阶拓扑GNN
class HoT_GNN(nn.Module):
    def __init__(self, n_nodes, node_features, edge_features, node_hidden, edge_hidden):
        super().__init__()
        self.n_nodes = n_nodes
        self.gnn = GNN(node_features, node_hidden)  # 节点特征提取
        self.hosc = HoSC(edge_features, edge_hidden)  # 边特征与高阶拓扑提取
        
        # 输出头
        self.node_head = nn.Linear(node_hidden[-1] + len(edge_hidden), 1)  # 节点失效概率
        self.edge_head = nn.Linear(len(edge_hidden), 1)  # 边失效概率
        self.hyperedge_head = nn.Linear(node_hidden[-1] + len(edge_hidden), 1)  # 超边交互概率
        
        # 嵌入维度
        self.embed_dim = node_hidden[-1] + len(edge_hidden)

    def forward(self, X_n, X_e, A_tilde, L1_tilde, B1, hyperedges=None):
        # 节点特征提取
        H_n = self.gnn(X_n, A_tilde)  # [n, d_{L_n}]

        # 边特征与高阶拓扑提取
        Z_H = self.hosc(X_e, L1_tilde)  # [M, L_e]

        # 边到节点映射
        H_e = torch.sparse.mm(B1, Z_H) if B1.is_sparse else torch.mm(B1, Z_H)  # [n, L_e]

        # 特征融合
        H = torch.cat([H_n, H_e], dim=1)  # [n, d_{L_n} + L_e]

        # 节点失效概率
        node_prob = torch.sigmoid(self.node_head(H)).squeeze(-1)  # [n]

        # 边失效概率
        edge_prob = torch.sigmoid(self.edge_head(Z_H)).squeeze(-1)  # [M]

        # 超边交互概率
        if hyperedges is not None:
            hyperedge_probs = []
            for hyperedge in hyperedges:
                # 聚合超边内节点特征
                hyperedge_features = H[hyperedge].mean(dim=0)  # [d_{L_n} + L_e]
                prob = torch.sigmoid(self.hyperedge_head(hyperedge_features))  # [1]
                hyperedge_probs.append(prob)
            hyperedge_prob = torch.stack(hyperedge_probs)  # [n_hyperedges]
        else:
            hyperedge_prob = None

        return {
            'node_prob': node_prob,  # [9168]
            'edge_prob': edge_prob,  # [11667]
            'hyperedge_prob': hyperedge_prob,  # [n_hyperedges] or None
            'node_embedding': H  # [9168, d_{L_n} + L_e]
        }

if __name__ == "__main__":
    n, M = 1000, 1500
    X_n = torch.randn(n, 3)
    X_e = torch.randn(M, 3)
    A_tilde = torch.randn(n, n)
    L1_tilde = torch.randn(M, M)
    B1 = torch.randn(n, M)
    hyperedges = [[0, 1, 2], [3, 4, 5]]  # 示例超边

    # 模型初始化
    model = HoT_GNN(
        n_nodes=n,
        node_features=3,
        edge_features=3,
        node_hidden=[32, 16, 8],
        edge_hidden=[16, 8, 1]
    )

    # 前向传播
    outputs = model(X_n, X_e, A_tilde, L1_tilde, B1, hyperedges)
    print(f"Node prob shape: {outputs['node_prob'].shape}")  # [1000]
    print(f"Edge prob shape: {outputs['edge_prob'].shape}")  # [1500]
    print(f"Hyperedge prob shape: {outputs['hyperedge_prob'].shape}")  # [2]
    print(f"Node embedding shape: {outputs['node_embedding'].shape}")  # [1000, 9]

    logger.add(log_file, mode="w")
    logger.info("=== Model Summary ===")
    logger.info(
        "\n" + str(
            summary(
                model,
                input_data=(X_n, X_e, A_tilde, L1_tilde, B1, hyperedges),
                depth=4,
                col_names=["input_size", "output_size", "num_params"],
                verbose=0
            )
        )
    )