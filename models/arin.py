import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class ARIN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=16):
        super().__init__()
        self.gcn = GCNConv(num_nodes, hidden_dim)
        self.attn = nn.Linear(3 + 1, 1)  # 灾害强度 + 地理距离

    def forward(self, intensities, avg_dist):
        # intensities: [3, time_steps] (洪水、热浪、干旱)
        # avg_dist: 受影响节点平均距离
        edge_index = torch.tensor([[0,1,2], [1,2,0]], dtype=torch.long)  # 灾害交互图
        h = self.gcn(intensities, edge_index)

        # 注意力机制（考虑灾害强度和地理因素）
        attn_input = torch.cat([
            intensities,
            avg_dist.expand(3, intensities.size(1))
        ], dim=0)
        alpha = torch.softmax(self.attn(attn_input.T), dim=1).T

        # 加权复合强度
        composite = (alpha * intensities).sum(dim=0)
        return composite.unsqueeze(0)  # [1, time_steps]