# 电力网络气候复合风险

## 项目简介
本项目构建动态系统模型与复合风险分析框架，研究洪水/热浪/干旱对欧洲电网（13,478节点）的连锁失效效应。核心功能包括：
- 基于图神经网络（GNN）的初始失效概率预测
- 动态传播因子（DPF）驱动的时空演化模拟
- 注意力增强风险交互网络（ARIN）的复合风险建模

## 项目结构
```text
project/
├── config/            # 配置管理
├── data/              # 数据生命周期
│   ├── raw/           # 原始电网数据
│   ├── processed/     # 预处理数据
│   └── results/       # 实验结果
├── models/            # 核心算法实现
├── utils/             # 工具函数
├── experiments/       # 独立实验脚本
├── scripts/           # 辅助脚本
├── notebooks/         # 探索性分析（可选）
└── requirements.txt   # 依赖清单
```

## 快速开始
### 1. 环境配置
```bash
# 克隆仓库
git clone https://github.com/your-repo.git

# 安装依赖（支持GPU加速）
pip install -r requirements.txt
```

### 2. 数据准备
```bash
# 执行预处理（生成邻接矩阵/气候特征）
python scripts/data_preprocess.py

# 验证数据完整性
ls data/processed/adj_sparse.npz
ls data/processed/climate_features.pt
```

### 3. 模型训练
```bash
# 训练GNN模型（预测初始失效概率）
python scripts/train_gnn.py

# 查看训练结果
tensorboard --logdir=runs/
```

### 4. 运行实验
```bash
# 单一灾害模拟（洪水示例）
python experiments/single_hazard.py --disaster flood

# 复合风险模拟（洪水+热浪）
python experiments/composite_hazard.py --disasters flood heat
```

## 核心功能
### 动态系统建模
```python
# 微分方程核心逻辑（models/dynamic.py）
def dynamic_system(state, t, probs, intensity_matrix, adj_sparse, adjacency_list, distances):
    # 实现动态传播因子（DPF）计算
    N1 = calculate_first_order_effects(...)  # 一阶邻居效应
    N2 = calculate_second_order_effects(...) # 二阶环路效应
    DPF = w1*N1 + w2*N2
    dSdt = -k*probs*intensity*S + r*(1-S)*(1-DPF)
    return dSdt
```

### 复合风险网络
```python
# 注意力机制实现（models/arin.py）
class ARIN(nn.Module):
    def forward(self, intensities, avg_dist):
        edge_index = self.build_interaction_graph()
        h = self.gcn(intensities, edge_index)
        alpha = self.attention_mechanism(h, avg_dist)
        C_composite = torch.sum(alpha * intensities, dim=0)
        return C_composite
```

## 结果示例
![状态演化曲线](docs/state_evolution.png)
_Figure: 不同灾害情景下的节点状态演化（72小时模拟）_

## 项目特点
- **高性能计算**：采用稀疏矩阵加速邻接计算，支持13k+节点规模
- **动态可视化**：自动生成风险扩散热力图（utils/vis_utils.py）
- **可扩展架构**：新增灾害类型仅需修改配置文件

## 贡献指南
1. Fork本仓库
2. 创建feature分支 (`git checkout -b feature/fooBar`)
3. 提交代码 (`git commit -m 'Add some fooBar'`)
4. 推送分支 (`git push origin feature/fooBar`)
5. 提交Pull~ Request

## 引用
如使用本项目请引用：
```bibtex
@article{your2025paper,
  title={Dynamic Cascading Failure Analysis in European Power Grids under Compound Climate Risks},
  author={Your Name},
  journal={Nature Energy},
  year={2025}
}
```

## 许可证
本项目采用MIT许可证，详见[LICENSE](LICENSE)文件