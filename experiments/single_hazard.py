import numpy as np
from data.processor import PowerGridProcessor
from models.gnn import GCNTrainer
from models.dynamic import DynamicSimulator
from utils.metrics import calculate_failure_rate

def run_single_hazard():
    # 数据加载
    processor = PowerGridProcessor()
    adj_sparse, adjacency_list, distances, climate_tensor = processor.load_processed_data()

    # GNN预测
    trainer = GCNTrainer()
    failure_probs = trainer.train_and_predict()

    # 动态模拟
    simulator = DynamicSimulator(adj_sparse, adjacency_list, distances)
    t_span = np.linspace(0, 72, 73)

    # 洪水情景
    flood_intensity = climate_tensor[:, 0].numpy().reshape(-1, 1) * (1 + t_span/72 * 4)
    S_flood = simulator.simulate(failure_probs, flood_intensity, t_span)

    # 结果分析
    flood_failure = calculate_failure_rate(S_flood)
    print(f"Flood Failure Rate: {flood_failure[-1]:.3f}")