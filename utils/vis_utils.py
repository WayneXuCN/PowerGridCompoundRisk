import matplotlib.pyplot as plt
from config.paths import RESULT_DIR

def plot_scenario_comparison(t, S_list, labels, title="Scenario Comparison"):
    plt.figure(figsize=(12, 6))
    for S, label in zip(S_list, labels):
        plt.plot(t, np.mean(S, axis=1), label=label)
    plt.xlabel("Time (hours)")
    plt.ylabel("Average Node State")
    plt.legend()
    plt.title(title)
    plt.savefig(RESULT_DIR / f"{title.replace(' ', '_')}.png")
    plt.close()

def plot_critical_nodes(critical_nodes, G, title="Critical Nodes"):
    # 可视化关键节点在电网中的位置
    pass  # 需要结合igraph的绘图功能实现