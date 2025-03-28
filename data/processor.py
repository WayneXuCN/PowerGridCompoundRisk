from itertools import combinations

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, diags, eye
from scipy.sparse.linalg import inv

from config.parameters import ClimateParams
from config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


class PowerGridProcessor:
    def __init__(self):
        self.edge_file = RAW_DATA_DIR / "european/powergridEU_E.csv"
        self.coord_file = RAW_DATA_DIR / "european/powergridEU_V.csv"

    def process(self):
        """完整数据处理流程"""
        # 1. 构建图结构
        g = self._build_graph()
        # 2. 计算节点间地理距离矩阵
        distances = self._compute_distances_matrix(g)
        # 3. 生成邻接矩阵和高阶拓扑
        adj_sparse, B1, L1_tilde = self._compute_hodge_laplacian(g) # 生成稀疏邻接矩阵
        # 4. 生成节点风险特征矩阵 X（包含经纬度和气候风险）
        X = self._generate_climate_features(g)
        # 5. 生成边特征矩阵 Z（基于节点特征）
        edge_features = self._generate_edge_features(g, X)
        # 6. 保存预处理结果
        self._save_processed_data(distances, adj_sparse,B1, L1_tilde, X, edge_features)

        return adj_sparse, B1, L1_tilde, distances, X, edge_features

    def _build_graph(self):
        """构建图结构"""
        edges = pd.read_csv(self.edge_file, header=None).values.tolist() # 读取边数据
        coords = pd.read_csv(self.coord_file, header=None) # 读取节点坐标数据

        g = ig.Graph.TupleList(edges, directed=False)  # 生成无向图
        g.vs["name"] = [int(v) for v in g.vs["name"]]  # 节点名转为整数

        # 批量添加坐标属性
        name_to_index = {v["name"]: idx for idx, v in enumerate(g.vs)}
        latitudes = [None] * g.vcount()
        longitudes = [None] * g.vcount()
        for _, row in coords.iterrows():
            node_id = int(row[0])
            if node_id in name_to_index:
                idx = name_to_index[node_id]
                latitudes[idx] = row[1]
                longitudes[idx] = row[2]
        # 添加坐标属性
        g.vs["latitude"] = latitudes
        g.vs["longitude"] = longitudes

        # 数据完整性检查
        assert all(lat is not None for lat in latitudes), "纬度数据缺失"
        assert all(lon is not None for lon in longitudes), "经度数据缺失"
        return g

    def _compute_distances_matrix(self, g):
        """
        计算节点间 Haversine 地理距离矩阵（用于动态传播因子）
        """
        # 转换为弧度
        latitudes = g.vs["latitude"]
        longitudes = g.vs["longitude"]
        lats_rad = np.radians(latitudes)
        lons_rad = np.radians(longitudes)
        # 使用广播计算差值
        lat_matrix = lats_rad[:, np.newaxis] - lats_rad[np.newaxis, :]
        lon_matrix = lons_rad[:, np.newaxis] - lons_rad[np.newaxis, :]
        # Haversine 公式
        a = np.sin(lat_matrix / 2) ** 2 + np.cos(lats_rad[:, np.newaxis]) * np.cos(lats_rad[np.newaxis, :]) * np.sin(
            lon_matrix / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # 添加 clip 避免数值误差

        # 地球半径 (km)
        R = 6371
        distances = R * c
        print(f"距离矩阵计算完成，形状：{distances.shape}")
        return distances

    def _compute_hodge_laplacian(self, g):
        """生成稀疏邻接矩阵和高阶拓扑表示"""
        # 构建邻接矩阵和边列表
        edges = np.array(g.get_edgelist())
        n_nodes, n_edges = g.vcount(), len(edges)
        adj_sparse = coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(g.vcount(), g.vcount()),
        ).tocsr()
        edge_dict = {tuple(sorted(e)): idx for idx, e in enumerate(edges)}
        # 构建 B1
        B1 = lil_matrix((n_nodes, n_edges))  # lil_matrix 更适合动态插入
        for idx, (i, j) in enumerate(edges):
            B1[i, idx] = 1  # 起点+1
            # B1[j, idx] = -1  # 终点-1
            B1[j, idx] = 1  # 无向图简化：两个节点均标记为+1
        B1 = B1.tocsr()  # 转换为CSR用于后续计算

        # 检测所有3节点环路
        triangles = []
        for node in g.vs:
            neighbors = node.neighbors()
            for pair in combinations(neighbors, 2):
                if g.are_adjacent(pair[0], pair[1]):
                    triangle = sorted([node.index, pair[0].index, pair[1].index])
                    triangles.append(tuple(triangle))
        triangles = list(set(triangles))  # 去重

        # 构建 B2（统一顺时针定向）
        B2 = lil_matrix((n_edges, len(triangles)))
        for t_idx, triangle in enumerate(triangles):
            # 定义顺时针边顺序：A→B, B→C, C→A
            ordered_edges = [
                tuple(sorted((triangle[0], triangle[1]))),
                tuple(sorted((triangle[1], triangle[2]))),
                tuple(sorted((triangle[2], triangle[0]))),
            ]
            for i, edge in enumerate(ordered_edges):
                if edge in edge_dict:
                    B2[edge_dict[edge], t_idx] = 1 if i == 0 else -1
        B2 = B2.tocsr()

        # 计算 Hodge 1-Laplacian
        L1 = B1.T @ B1 + B2 @ B2.T

        # 归一化 Hodge 1-Laplacian
        # 计算 D2（边度矩阵）
        D2_diag = np.array(np.abs(B2).sum(axis=1)).flatten()
        D2_diag = np.maximum(D2_diag, 1)  # 避免除零，最小值为1
        D2 = diags(D2_diag)

        # 计算 D1（节点度矩阵）
        node_degree = np.abs(B1).dot(D2_diag)  # |B1| D2 1
        D1_diag = 2 * node_degree
        D1 = diags(D1_diag)

        # 计算 D3（面度矩阵）
        D3 = (1 / 3) * eye(B2.shape[1])  # 面数量为B2的列数

        # 计算稀疏矩阵的逆时，确保矩阵为 CSC 格式
        D1_inv = inv(D1.tocsc())  # 转换为 CSC 格式
        D2_inv = inv(D2.tocsc())  # 转换为 CSC 格式

        # 构建归一化 Hodge 1-Laplacian
        L1_down = D2 @ B1.T @ D1_inv @ B1
        L1_up = B2 @ D3 @ B2.T @ D2_inv
        L1_tilde = L1_down + L1_up

        print(f"检测到 {len(triangles)} 个三角形环路")
        print(f"B1 shape: {B1.shape}")
        print(f"B2 shape: {B2.shape}")
        print(f"L1_tilde shape: {L1_tilde.shape}")

        return adj_sparse, B1, L1_tilde

    def _generate_climate_features(self, g):
        """生成节点风险特征（洪水、热浪、干旱）"""
        latitudes = np.array(g.vs["latitude"], dtype=np.float32)
        longitudes = np.array(g.vs["longitude"], dtype=np.float32)
        # 洪水风险：纬度越低风险越高
        F = np.maximum(0, (ClimateParams.flood_max_lat - latitudes) / ClimateParams.flood_scale)
        # 热浪风险：纬度越高风险越高
        H = np.maximum(0, (latitudes - ClimateParams.heat_min_lat) / ClimateParams.heat_scale)
        # 干旱风险：经度越东风险越高
        D = np.maximum(0, (longitudes + ClimateParams.drought_offset) / ClimateParams.drought_scale)
        """
        # 归一化风险值到 [0, 1] 范围
        scaler = MinMaxScaler()
        F = scaler.fit_transform(F.reshape(-1, 1)).flatten()
        H = scaler.fit_transform(H.reshape(-1, 1)).flatten()
        D = scaler.fit_transform(D.reshape(-1, 1)).flatten()
        """

        # 合并经纬度和气候风险
        X = np.column_stack([latitudes, longitudes, F, H, D])
        print(f"节点风险特征特征矩阵生成完成（包含经纬度和气候风险），形状：{X.shape}")
        return X

    def _generate_edge_features(self, g, X):
        """将节点特征转换为边特征"""
        edges = np.array(g.get_edgelist())  # 边列表 (16922×2)
        edge_features = []
        for i, j in edges:
            # 确保节点索引有效
            if i >= X.shape[0] or j >= X.shape[0]:
                raise ValueError(f"无效节点索引：i={i}, j={j}, X.shape={X.shape}")
            # 取节点i和j的特征的平均值
            edge_feature = (X[i] + X[j]) / 2
            edge_features.append(edge_feature)
        edge_features = np.array(edge_features, dtype=np.float32)
        print(f"边特征生成完成，形状：{edge_features.shape}")
        return edge_features

    def _save_processed_data(self, distances, adj_sparse, B1, L1_tilde, X, edge_features):
        """保存预处理结果"""
        np.save(PROCESSED_DATA_DIR / "distances.npy", distances)
        sp.save_npz(PROCESSED_DATA_DIR / "adj_sparse.npz", adj_sparse)
        sp.save_npz(PROCESSED_DATA_DIR / "B1.npz", B1)
        sp.save_npz(PROCESSED_DATA_DIR / "L1_tilde.npz", L1_tilde)
        np.save(PROCESSED_DATA_DIR / "X.npy", X)
        np.save(PROCESSED_DATA_DIR / "edge_features.npy", edge_features)
        print("预处理数据已保存")


if __name__ == "__main__":
    processor = PowerGridProcessor()
    adj_sparse, B1, L1_tilde, distances, X, edge_features = processor.process()