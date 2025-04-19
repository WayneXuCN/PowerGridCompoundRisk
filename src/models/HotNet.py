from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from src.config import LOGS_DIR


# HoSC 单层模块：高阶单纯形卷积
class HigherOrderSimplicialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.theta = nn.Linear(
            in_channels, out_channels
        )  # 权重矩阵 Θ_H [in_channels, out_channels]
        self.bn = nn.BatchNorm1d(out_channels)  # 批量归一化层，稳定训练

    def forward(self, Z_H, L1_tilde):
        # Z_H [M, in_channels]：输入边嵌入
        # L1_tilde [M, M]：Hodge 1-Laplacian
        Z_theta = self.theta(Z_H)  # 线性变换 [M, out_channels]
        Z_conv = (
            torch.sparse.mm(L1_tilde, Z_theta)
            if L1_tilde.is_sparse
            else torch.mm(L1_tilde, Z_theta)
        )  # 卷积 [M, out_channels]
        Z_psi = F.relu(self.bn(Z_conv))  # 非线性变换（BatchNorm + ReLU）[M, out_channels]
        Z_max, _ = torch.max(Z_psi, dim=1, keepdim=True)  # 特征维度最大值 [M, 1]
        return Z_max


# HoSC 模块：多层高阶单纯形卷积
class HoSC(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()  # 存储多层 HoSC
        current_dim = input_dim  # 初始输入维度
        for dim in hidden_dims:
            self.layers.append(
                HigherOrderSimplicialConv(current_dim, dim)
            )  # 添加单层 [current_dim, dim]
            current_dim = 1  # 每层输出固定为 1

    def forward(self, edge_attr, L1_tilde):
        # edge_attr [M, input_dim]：边特征
        # L1_tilde [M, M]：Hodge 1-Laplacian
        Z_list = []  # 存储各层输出
        Z_H = edge_attr  # 初始边嵌入 [M, input_dim]
        for layer in self.layers:
            Z_H = layer(Z_H, L1_tilde)  # 单层 HoSC 输出 [M, 1]
            Z_list.append(Z_H)
        return torch.cat(Z_list, dim=1)  # 拼接所有层输出 [M, len(hidden_dims)]


# GNN 模块：节点特征提取
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()  # 存储多层 GCN
        current_dim = input_dim  # 初始输入维度
        for dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, dim))  # 线性层 [current_dim, dim]
            current_dim = dim  # 更新维度

    def forward(self, X_n, A_tilde):
        # X_n [n, input_dim]：节点特征
        # A_tilde [n, n]：归一化邻接矩阵
        H = X_n  # 初始节点嵌入 [n, input_dim]
        for layer in self.layers:
            H = torch.mm(A_tilde, H)  # 邻域聚合 [n, current_dim]
            H = F.relu(H @ layer.weight.T + layer.bias)  # 线性变换与激活 [n, next_dim]
        return H  # 最终节点嵌入 [n, hidden_dims[-1]]


# HoT_GNN
class HoT_GNN(nn.Module):
    def __init__(self, n_nodes, node_features, edge_features, node_hidden, edge_hidden):
        super().__init__()
        self.n_nodes = n_nodes  # 节点数 n，例如 100
        self.gnn = GNN(
            node_features, node_hidden
        )  # GNN 模块，输入 node_features，输出 hidden_dims[-1]
        self.hosc = HoSC(
            edge_features, edge_hidden
        )  # HoSC 模块，输入 edge_features，输出 len(edge_hidden)
        # 全连接层
        self.fc1 = nn.Linear(
            node_hidden[-1] + len(edge_hidden), 2 * n_nodes
        )  # [d_{L_n} + L_e, 2n]
        self.fc2 = nn.Linear(2 * n_nodes, 1)

    def forward(self, X_n, X_e, A_tilde, L1_tilde, B1):
        # 节点特征提取
        H_n = self.gnn(X_n, A_tilde)  # GNN 输出 [n, d_{L_n}]

        # 边特征与高阶拓扑提取
        Z_H = self.hosc(X_e, L1_tilde)  # HoSC 输出 [M, L_e]

        # 边到节点映射
        H_e = torch.sparse.mm(B1, Z_H) if B1.is_sparse else torch.mm(B1, Z_H)  # [n, L_e]

        # 特征融合
        H = torch.cat([H_n, H_e], dim=1)  # 拼接 [n, d_{L_n} + L_e]

        # 全局变换
        f = F.relu(self.fc1(H))  # 特征扩展：第一层全连接 [n, 2n]
        f = self.fc2(f)  # 特征映射回节点空间：第二层全连接 [n, n]

        # 概率输出
        z_p = F.sigmoid(f).squeeze(-1)
        return z_p  # 单样本输出 [n]，例如 [100]


if __name__ == "__main__":
    n, M = 1000, 1500  # 节点数和边数
    X_n = torch.randn(n, 3)  # 节点特征 [n, 3]
    X_e = torch.randn(M, 3)  # 边特征 [M, 3]
    A_tilde = torch.randn(n, n)  # 归一化邻接矩阵 [n, n]
    L1_tilde = torch.randn(M, M)  # Hodge 1-Laplacian [M, M]
    B1 = torch.randn(n, M)  # 边到节点映射 [n, M]

    # 模型初始化
    model = GNN(
        n_nodes=n,
        node_features=3,
        edge_features=3,
        node_hidden=[32, 16, 8],  # 节点隐藏层维度
        edge_hidden=[16, 8, 1],  # 边隐藏层维度
    )

    log_file = LOGS_DIR / "HotNet_summary.log"
    logger.add(log_file, mode="w")  # 表示覆盖保存
    logger.info("=== Model Summary ===")
    logger.info(
        "\n"
        + str(
            summary(
                model,
                input_data=(X_n, X_e, A_tilde, L1_tilde, B1),
                depth=4,
                col_names=["input_size", "output_size", "num_params"],
                verbose=0,
            )
        )
    )

    z_p = model(X_n, X_e, A_tilde, L1_tilde, B1)
    print(f"Output shape: {z_p.shape}")  # 输出形状：[100]
