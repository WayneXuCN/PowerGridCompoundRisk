import torch
from models.arin import ARIN
from utils.metrics import find_critical_nodes

def run_composite_hazard():
    # 加载数据
    processor = PowerGridProcessor()
    adj_sparse, _, distances, climate_tensor = processor.load_processed_data()

    # 计算平均距离
    affected_mask = failure_probs > 0.5
    avg_dist = np.mean(distances[affected_mask][:, affected_mask]) if np.any(affected_mask) else 50.0

    # 构建ARIN模型
    arin = ARIN(num_nodes=adj_sparse.shape[0])
    intensities = torch.stack([
        torch.tensor(flood_intensity[:, -1]),
        torch.tensor(heat_intensity[:, -1]),
        torch.zeros_like(torch.tensor(drought_intensity[:, -1]))
    ])

    # 复合强度预测
    composite_intensity = arin(intensities, torch.tensor(avg_dist))

    # 动态模拟
    simulator = DynamicSimulator(adj_sparse, adjacency_list, distances)
    S_composite = simulator.simulate(failure_probs, composite_intensity.numpy(), t_span)

    # 关键节点识别
    critical_nodes = find_critical_nodes(failure_probs, S_composite[-1])
    print(f"Critical Nodes: {critical_nodes[:5]}")