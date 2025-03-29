import torch
import torch.nn as nn
import torch.nn.functional as F

class HigherOrderSimplicialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels)  # 可训练权重 Θ_H
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, Z_H, L1_tilde):
        # 公式: Z_H^{(ℓ+1)} = max(Ψ(˜L1 Z_H^{(ℓ)} Θ))
        # 1. 线性变换 Θ_H
        Z_theta = self.theta(Z_H)  # [M, out_channels]
        # 2. Hodge 1-Laplacian 矩阵乘法
        if L1_tilde.is_sparse:
            Z_conv = torch.sparse.mm(L1_tilde, Z_theta)
        else:
            Z_conv = torch.mm(L1_tilde, Z_theta)  # [M, out_channels]

        # 3. 非线性变换 Ψ (BatchNorm + ReLU)
        Z_psi = F.relu(self.bn(Z_conv))

        # 4. 元素级最大池化（沿特征维度取每个边的最大值）
        Z_max, _ = torch.max(Z_psi, dim=1, keepdim=True)  # [M, 1]

        return Z_max

class HoSC(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(HigherOrderSimplicialConv(current_dim, dim))
            current_dim = 1  # 每层输出维度为1

    def forward(self, edge_attr, L1_tilde):
        Z_list = []
        Z_H = edge_attr
        for layer in self.layers:
            Z_H = layer(Z_H, L1_tilde)
            Z_list.append(Z_H)

        # 沿特征维度拼接所有层的输出
        return torch.cat(Z_list, dim=1)  # [M, num_layers]

class HOTNet(nn.Module):
    def __init__(self, edge_features, hidden_dims, num_classes=1):
        super().__init__()
        self.hosc = HoSC(edge_features, hidden_dims)
        self.fc = nn.Linear(len(hidden_dims), num_classes)

    def forward(self, data):
        # 提取边特征和Hodge矩阵
        edge_attr = data.edge_attr  # [M, edge_features]
        L1_tilde = data.L1_tilde    # [M, M]

        # 通过HoSC模块得到边级嵌入
        Z_H = self.hosc(edge_attr, L1_tilde)  # [M, num_layers]

        # 全局池化（对边特征进行全局平均）
        graph_embedding = Z_H.mean(dim=0).unsqueeze(0)  # [1, num_layers]

        # 分类头
        out = self.fc(graph_embedding)
        return torch.sigmoid(out)