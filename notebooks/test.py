import igraph as ig
import numpy as np
import random
from scipy.stats import expon, norm, lognorm
import matplotlib.pyplot as plt

class TemporalHypergraph:
    def __init__(self):
        self.graph = ig.Graph(directed=True)  # 创建有向超图
        self.vertices = []  # 超图顶点（灾害源）
        self.edges = []     # 超图边（灾害交互）

    def add_vertex(self, name, attributes=None):
        """添加顶点（灾害源）"""
        self.vertices.append(name)
        self.graph.add_vertex(name=name, **(attributes or {}))

    def add_edge(self, source, targets, attributes=None):
        """添加边（灾害交互），并初始化边属性"""
        for target in targets:
            self.graph.add_edge(source, target, time=None, severity=None)

    def update_edge_attributes(self, source, target, time, severity):
        """更新边的时间和严重性属性"""
        try:
            edge_id = self.graph.get_eid(source, target)
            self.graph.es[edge_id]["time"] = time
            self.graph.es[edge_id]["severity"] = severity
        except ValueError:
            print(f"Edge between {source} and {target} does not exist.")


def simulate_multihazard_scenario(hypergraph, T, hazard_models):
    """模拟多灾害场景"""
    t_current = 0
    while t_current < T:
        # 模拟主灾害发生
        for vertex in hypergraph.vertices:
            if vertex.startswith("Earthquake"):
                rate = hazard_models[vertex]["rate"]
                earthquake_times = simulate_earthquake_occurrence(rate, T)
                for t_eq in earthquake_times:
                    # 更新超图边属性
                    for edge in hypergraph.edges:
                        if edge[0] == vertex:
                            for target in edge[1]:  # 遍历目标顶点
                                severity = hazard_models[vertex]["severity"](t_eq)
                                hypergraph.update_edge_attributes(vertex, target, t_eq, severity)

        # 模拟次生灾害触发
        for edge in hypergraph.edges:
            source, targets = edge
            for target in targets:
                if target.startswith("Liquefaction"):
                    # 液化触发模型
                    pass
                elif target.startswith("Tsunami"):
                    # 海啸触发模型
                    pass
                elif target.startswith("Flood"):
                    # 洪水触发模型
                    pass

        t_current += 1  # 时间步进


def sliding_window_analysis(hypergraph, T, w, s):
    """滑动窗口分析"""
    windows = []
    for t_start in range(0, T, s):
        t_end = t_start + w
        window_events = []
        for edge in hypergraph.edges:
            source, targets = edge
            for target in targets:
                try:
                    # 获取边索引
                    edge_id = hypergraph.graph.get_eid(source, target)
                    # 检查时间是否在窗口内
                    if t_start <= hypergraph.graph.es[edge_id]["time"] <= t_end:
                        window_events.append((source, target, hypergraph.graph.es[edge_id]["severity"]))
                except ValueError:
                    print(f"Edge between {source} and {target} does not exist.")
        windows.append((t_start, t_end, window_events))
    return windows


# 初始化时间超图
thg = TemporalHypergraph()
thg.add_vertex("Earthquake_Fault1", {"rate": 0.01})
thg.add_vertex("Earthquake_Fault2", {"rate": 0.025})
thg.add_vertex("Liquefaction_Site")
thg.add_vertex("Tsunami_Ocean")
thg.add_vertex("Flood_River")

thg.add_edge("Earthquake_Fault1", ["Liquefaction_Site"])
thg.add_edge("Earthquake_Fault2", ["Liquefaction_Site", "Tsunami_Ocean"])
thg.add_edge("Tsunami_Ocean", ["Flood_River"])

# 定义灾害模型
hazard_models = {
    "Earthquake_Fault1": {"rate": 0.01, "severity": lambda t: simulate_tsunami_runup(7.1)},
    "Earthquake_Fault2": {"rate": 0.025, "severity": lambda t: simulate_tsunami_runup(7.5)},
    "Flood_River": {"severity": simulate_flood_depth},
}

# 模拟多灾害场景
simulate_multihazard_scenario(thg, T=50, hazard_models=hazard_models)

# 滑动窗口分析
windows = sliding_window_analysis(thg, T=50, w=2, s=1)

# 打印结果
for t_start, t_end, events in windows:
    print(f"Window [{t_start}, {t_end}]: {events}")