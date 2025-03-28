import numpy as np
from scipy.integrate import odeint
from config.parameters import DynamicParams

class DynamicSimulator:
    def __init__(self, adj_sparse, adjacency_list, distances):
        self.adj_sparse = adj_sparse
        self.adjacency_list = adjacency_list
        self.distances = distances

    def simulate(self, probs, intensity_matrix, t_span):
        S0 = np.ones(self.adj_sparse.shape[0])
        return odeint(
            self._dynamic_system,
            S0,
            t_span,
            args=(probs, intensity_matrix),
            mxstep=5000  # 防止积分发散
        )

    def _dynamic_system(self, state, t, probs, intensity_matrix):
        k = DynamicParams.k
        r = DynamicParams.r
        sigma = DynamicParams.sigma
        w1, w2 = DynamicParams.w1, DynamicParams.w2

        # 时间索引计算
        t_idx = min(int(t / 72 * (intensity_matrix.shape[1] - 1)),
                    intensity_matrix.shape[1] - 1)
        current_intensity = intensity_matrix[:, t_idx]

        # 一阶邻居影响（向量化优化）
        N1 = np.zeros_like(state)
        for i in range(len(state)):
            neighbors = self.adjacency_list[i]
            if neighbors:
                neighbor_effects = (1 - state[neighbors]) * np.exp(
                    -self.distances[i, neighbors] / sigma
                )
                N1[i] = neighbor_effects.mean() if neighbor_effects.size else 0

        # 二阶影响（稀疏矩阵加速）
        adj_sq = self.adj_sparse.T.dot(self.adj_sparse).tolil()
        adj_sq.setdiag(0)
        adj_sq = adj_sq.tocsr()
        N2 = np.array(adj_sq.multiply(1 - state).power(2).mean(axis=1)).flatten()

        DPF = w1 * N1 + w2 * N2
        dSdt = -k * probs * current_intensity * state + r * (1 - state) * (1 - DPF)
        return dSdt