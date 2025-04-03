from itertools import combinations
from pathlib import Path

import igraph as ig
from loguru import logger
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix, diags, eye, lil_matrix, triu
from scipy.sparse.linalg import inv
import typer
from typing_extensions import Annotated

from src.config import INTERIM_DATA_DIR, LOGS_DIR, RAW_DATA_DIR, ClimateParams


class PowerGridProcessor:
    def __init__(self, edge_file: Path, coord_file: Path, output_dir: Path):
        """
        初始化 PowerGridProcessor。

        Parameters
        ----------
        edge_file : Path
            边数据文件路径。
        coord_file : Path
            节点坐标数据文件路径。
        output_dir : Path
            输出目录路径，用于保存处理结果。
        """
        self.edge_file = edge_file
        self.coord_file = coord_file
        self.output_dir = output_dir

    def process(self):
        """
        执行数据处理的完整流程。

        该方法依次执行以下步骤：
        1. 构建图结构
        2. 计算地理距离矩阵
        3. 生成邻接矩阵和高阶拓扑
        4. 生成节点的风险特征矩阵
        5. 生成边的特征矩阵
        6. 保存处理结果到指定输出目录

        Returns
        -------
        tuple
            返回处理结果，包括邻接矩阵、B1矩阵、L1矩阵、L1_tilde矩阵、地理距离矩阵、节点特征矩阵、边特征矩阵。
        """
        logger.info("启动数据处理流程")

        # 1. 构建图结构
        g = self._build_graph()
        # 2. 计算节点间地理距离矩阵
        distances = self._compute_distances_matrix(g)
        # 3. 生成邻接矩阵和高阶拓扑
        adj_sparse, adj_tilde_sparse, B1, L1, L1_tilde = self._compute_hodge_laplacian(g)
        # 4. 生成节点风险特征矩阵 X（包含经纬度和气候风险）
        X = self._generate_climate_features(g)
        # 5. 生成边特征矩阵 Z（基于节点特征）
        edge_features = self._generate_edge_features(g, X)
        # 6. 保存预处理结果
        self._save_processed_data(
            distances, adj_sparse, adj_tilde_sparse, B1, L1, L1_tilde, X, edge_features
        )

        logger.success("数据处理流程已完成")
        return adj_sparse, adj_tilde_sparse, B1, L1, L1_tilde, distances, X, edge_features

    def _build_graph(self):
        """
        构建无向图结构。

        该方法从边数据文件和坐标数据文件构建一个无向图，并为每个节点分配坐标属性。

        Returns
        -------
        igraph.Graph
            返回构建好的图对象，包含节点和边的详细信息。
        """
        logger.info("正在构建图结构")

        edges = pd.read_csv(self.edge_file, header=None).values.tolist()  # 读取边数据
        coords = pd.read_csv(self.coord_file, header=None)  # 读取节点坐标数据

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

        logger.info("图结构构建完成")
        return g

    def _compute_distances_matrix(self, g):
        """
        计算节点间的 Haversine 地理距离矩阵。

        使用 Haversine 公式计算节点之间的地理距离，并返回上三角稀疏矩阵。

        Parameters
        ----------
        g : igraph.Graph
            输入的图结构，包含节点的经纬度信息。

        Returns
        -------
        scipy.sparse.csr_matrix
            返回计算得到的地理距离矩阵（上三角）。
        """
        logger.info("正在计算地理距离矩阵")

        latitudes = g.vs["latitude"]
        longitudes = g.vs["longitude"]
        lats_rad = np.radians(latitudes)
        lons_rad = np.radians(longitudes)
        lat_matrix = lats_rad[:, np.newaxis] - lats_rad[np.newaxis, :]
        lon_matrix = lons_rad[:, np.newaxis] - lons_rad[np.newaxis, :]
        a = (
            np.sin(lat_matrix / 2) ** 2
            + np.cos(lats_rad[:, np.newaxis])
            * np.cos(lats_rad[np.newaxis, :])
            * np.sin(lon_matrix / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # 添加 clip 避免数值误差
        R = 6371  # 地球半径 (km)
        distances = (R * c).astype(np.float32)  # 单精度

        # 只保留上三角
        distances = triu(distances, k=1)

        logger.info(f"地理距离矩阵计算完成，维度：{distances.shape}")
        return distances

    def _compute_hodge_laplacian(self, g):
        """
        计算 Hodge 1-Laplacian 和相关的拓扑矩阵。

        该方法生成稀疏邻接矩阵，构建 B1 和 B2 矩阵，并计算归一化的 Hodge 1-Laplacian。

        Parameters
        ----------
        g : igraph.Graph
            输入的图结构，包含图的边和节点。

        Returns
        -------
        tuple
            返回邻接矩阵、B1 矩阵、L1 矩阵和 L1_tilde 矩阵。
        """
        logger.info("正在计算 Hodge Laplacian 和高阶拓扑结构")

        edges = np.array(g.get_edgelist())
        n_nodes, n_edges = g.vcount(), len(edges)
        # 构建稀疏邻接矩阵
        adj_sparse = coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(g.vcount(), g.vcount()),
        ).tocsr()

        # 节点度矩阵
        D = diags(np.array(adj_sparse.sum(axis=1)).flatten())
        # 归一化邻接矩阵 计算 D^(-1/2) * A * D^(-1/2)
        # 先提取对角元素，计算-0.5次方，再构建对角矩阵
        D_diag = np.array(D.diagonal())
        # 防止除零错误，将零度节点的度设为1
        zero_degrees = D_diag == 0
        if np.any(zero_degrees):
            logger.warning(f"检测到 {np.sum(zero_degrees)} 个零度节点，已处理以避免除零错误")
            D_diag[zero_degrees] = 1.0
        D_pow_minus_half = diags(np.power(D_diag, -0.5))
        adj_tilde_sparse = D_pow_minus_half @ adj_sparse @ D_pow_minus_half

        edge_dict = {tuple(sorted(e)): idx for idx, e in enumerate(edges)}

        # 构建 B1
        B1 = lil_matrix((n_nodes, n_edges))
        for idx, (i, j) in enumerate(edges):
            B1[i, idx] = 1
            B1[j, idx] = 1
        B1 = B1.tocsr()

        # 检测所有3节点环路
        triangles = []
        for node in g.vs:
            neighbors = node.neighbors()
            for pair in combinations(neighbors, 2):
                if g.are_adjacent(pair[0], pair[1]):
                    triangle = sorted([node.index, pair[0].index, pair[1].index])
                    triangles.append(tuple(triangle))
        triangles = list(set(triangles))  # 去重

        B2 = lil_matrix((n_edges, len(triangles)))
        for t_idx, triangle in enumerate(triangles):
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
        logger.info(f"已识别 {len(triangles)} 个三角形环路")

        # 归一化 Hodge 1-Laplacian
        D2_diag = np.array(np.abs(B2).sum(axis=1)).flatten()
        D2_diag = np.maximum(D2_diag, 1)
        D2 = diags(D2_diag)

        node_degree = np.abs(B1).dot(D2_diag)
        D1_diag = 2 * node_degree
        D1 = diags(D1_diag)

        D3 = (1 / 3) * eye(B2.shape[1])

        D1_inv = inv(D1.tocsc())
        D2_inv = inv(D2.tocsc())

        L1_down = D2 @ B1.T @ D1_inv @ B1
        L1_up = B2 @ D3 @ B2.T @ D2_inv
        L1_tilde = L1_down + L1_up

        logger.info(f"B1 矩阵维度: {B1.shape}")
        logger.info(f"B2 矩阵维度: {B2.shape}")
        logger.info(f"L1_tilde 矩阵维度: {L1_tilde.shape}")

        return adj_sparse, adj_tilde_sparse, B1, L1, L1_tilde

    def _generate_climate_features(self, g):
        """
        生成节点的气候风险特征。

        该方法为每个节点计算洪水、热浪和干旱的风险，并返回合成的特征矩阵。

        Parameters
        ----------
        g : igraph.Graph
            输入的图结构，包含节点的经纬度信息。

        Returns
        -------
        numpy.ndarray
            返回包含气候风险的节点特征矩阵。
        """
        logger.info("正在生成节点气候风险特征")

        latitudes = np.array(g.vs["latitude"], dtype=np.float32)
        longitudes = np.array(g.vs["longitude"], dtype=np.float32)
        F = np.maximum(0, (ClimateParams.flood_max_lat - latitudes) / ClimateParams.flood_scale)
        H = np.maximum(0, (latitudes - ClimateParams.heat_min_lat) / ClimateParams.heat_scale)
        D = np.maximum(
            0, (longitudes + ClimateParams.drought_offset) / ClimateParams.drought_scale
        )

        X = np.column_stack([latitudes, longitudes, F, H, D])

        logger.info(f"节点气候风险特征生成完成，维度：{X.shape}")
        return X

    def _generate_edge_features(self, g, X):
        """
        生成边的特征矩阵。

        该方法计算每条边的特征，特征由边两端节点的特征值的平均值组成。

        Parameters
        ----------
        g : igraph.Graph
            输入的图结构，包含节点及其特征。
        X : numpy.ndarray
            节点特征矩阵。

        Returns
        -------
        numpy.ndarray
            返回生成的边特征矩阵。
        """
        logger.info("正在生成边特征矩阵")

        edges = np.array(g.get_edgelist())
        edge_features = []
        for i, j in edges:
            if i >= X.shape[0] or j >= X.shape[0]:
                raise ValueError(f"无效节点索引：i={i}, j={j}, X.shape={X.shape}")
            edge_feature = (X[i] + X[j]) / 2
            edge_features.append(edge_feature)
        edge_features = np.array(edge_features, dtype=np.float32)

        logger.info(f"边特征矩阵生成完成，维度：{edge_features.shape}")
        return edge_features

    def _save_processed_data(
        self, distances, adj_sparse, adj_tilde_sparse, B1, L1, L1_tilde, X, edge_features
    ):
        """
        保存处理结果到指定的输出目录。

        该方法将所有处理结果保存为稀疏矩阵和标准格式文件。

        Parameters
        ----------
        distances : scipy.sparse.csr_matrix
            节点间地理距离的稀疏矩阵。
        adj_sparse : scipy.sparse.csr_matrix
            稀疏邻接矩阵。
        adj_tilde_sparse: scipy.sparse.csr_matrix
            归一化邻接矩阵。
        B1 : scipy.sparse.csr_matrix
            B1 矩阵。
        L1 : scipy.sparse.csr_matrix
            L1 矩阵。
        L1_tilde : scipy.sparse.csr_matrix
            L1_tilde 矩阵。
        X : numpy.ndarray
            节点特征矩阵。
        edge_features : numpy.ndarray
            边特征矩阵。
        """
        logger.info("正在保存处理结果")

        sp.save_npz(self.output_dir / "distances.npz", distances)
        sp.save_npz(self.output_dir / "adj_sparse.npz", adj_sparse)
        sp.save_npz(self.output_dir / "adj_tilde_sparse.npz", adj_tilde_sparse)
        sp.save_npz(self.output_dir / "B1.npz", B1)
        sp.save_npz(self.output_dir / "L1.npz", L1)
        sp.save_npz(self.output_dir / "L1_tilde.npz", L1_tilde)
        np.save(self.output_dir / "X.npy", X)
        np.save(self.output_dir / "edge_features.npy", edge_features)

        logger.success("所有处理结果已保存到指定位置")


app = typer.Typer()


@app.command()
def main(
    edge_file: Annotated[
        Path,
        typer.Argument(help="边数据文件路径，包含电网边信息"),
    ] = RAW_DATA_DIR / "european/powergridEU_E.csv",
    coord_file: Annotated[
        Path,
        typer.Argument(help="节点坐标数据文件路径，包含节点位置信息"),
    ] = RAW_DATA_DIR / "european/powergridEU_V.csv",
    output_dir: Annotated[Path, typer.Argument(help="处理结果保存目录路径")] = INTERIM_DATA_DIR
    / "european/",
):
    """
    从原始数据文件生成电网图特征，并将结果保存至指定目录。
    """
    log_file = LOGS_DIR / "european_features.log"
    logger.add(log_file, mode="w")  # 表示覆盖保存
    logger.info("开始电网图数据特征处理")

    # 初始化 PowerGridProcessor 处理器
    processor = PowerGridProcessor(
        edge_file=edge_file, coord_file=coord_file, output_dir=output_dir
    )

    # 执行数据处理
    adj_sparse, adj_tilde_sparse, B1, L1, L1_tilde, distances, X, edge_features = (
        processor.process()
    )

    # 处理完成后记录日志
    logger.success("电网图特征处理完成")
    logger.info(f"处理结果已保存至: {output_dir}")


if __name__ == "__main__":
    app()
