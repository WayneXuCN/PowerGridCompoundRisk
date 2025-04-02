from pathlib import Path
from typing import Annotated
import igraph as ig
from loguru import logger
import numpy as np
import pandas as pd
import scipy.sparse as sp
import typer
from tabulate import tabulate  # 用于格式化表格输出

from src.config import INTERIM_DATA_DIR, LOGS_DIR, RAW_DATA_DIR


def load_data(edge_file: Path, coord_file: Path):
    """
    加载边数据和节点坐标数据

    Parameters
    ----------
    edge_file : Path
        边数据文件路径，包含电网边信息
    coord_file : Path
        节点坐标数据文件路径，包含节点位置信息

    Returns
    -------
    edges : list
        电网边的列表，每一项包含一对节点连接
    coords : pandas.DataFrame
        节点的坐标数据
    """
    logger.info("Loading edge and coordinate data...")
    edges = pd.read_csv(edge_file, header=None).values.tolist()
    coords = pd.read_csv(coord_file, header=None)
    return edges, coords


def compute_graph_properties(graph: ig.Graph):
    """
    计算图的一些基本属性

    Parameters
    ----------
    graph : ig.Graph
        igraph 图对象

    Returns
    -------
    dict
        包含图的基本属性，包括节点数、边数、密度、最大度数等
    """
    num_nodes = graph.vcount()
    num_links = graph.ecount()
    density = graph.density()
    max_degree = max(graph.degree())
    avg_degree = sum(graph.degree()) / num_nodes
    avg_clustering_coeff = graph.transitivity_avglocal_undirected()
    num_connected_components = len(graph.components())
    diameter = graph.diameter(directed=False)
    avg_shortest_path_length = graph.average_path_length(directed=False)

    return {
        "Num Nodes": num_nodes,
        "Num Links": num_links,
        "Density": density,
        "Max Degree": max_degree,
        "Avg Degree": avg_degree,
        "Avg Clustering Coeff": avg_clustering_coeff,
        "Num Connected Components": num_connected_components,
        "Diameter": diameter,
        "Avg Shortest Path Length": avg_shortest_path_length,
    }


def load_intermediate_matrices():
    """
    加载并返回所有中间计算的矩阵

    Returns
    -------
    dict
        包含中间计算结果的矩阵，包括邻接矩阵、节点-边关联矩阵等
    """
    logger.info("Loading intermediate matrices...")
    distances_upper = sp.load_npz(INTERIM_DATA_DIR / "distances.npz").toarray()
    distances = distances_upper + distances_upper.T
    adj_sparse = sp.load_npz(INTERIM_DATA_DIR / "adj_sparse.npz")
    B1 = sp.load_npz(INTERIM_DATA_DIR / "B1.npz")
    L1_tilde = sp.load_npz(INTERIM_DATA_DIR / "L1_tilde.npz")
    X = np.load(INTERIM_DATA_DIR / "X.npy", allow_pickle=True)

    return {
        "distances": distances,
        "adj_sparse": adj_sparse,
        "B1": B1,
        "L1_tilde": L1_tilde,
        "X": X,
    }


def log_results(graph_properties: dict, matrices: dict, log_file: Path):
    """
    将计算结果输出到日志中，并输出为表格格式

    Parameters
    ----------
    graph_properties : dict
        图的一些基本属性，包括节点数、边数、密度等
    matrices : dict
        包含所有中间矩阵的字典，例如邻接矩阵、节点-边关联矩阵等
    log_file : Path
        日志文件路径
    """
    logger.add(log_file, mode="w")  # 覆盖保存日志文件
    
    # 格式化输出图属性表
    logger.info("=== Graph Summary ===")
    graph_properties_table = pd.DataFrame(list(graph_properties.items()), columns=["Property", "Value"])
    logger.info("\n" + tabulate(graph_properties_table, headers="keys", tablefmt="grid", showindex=False))

    # 输出矩阵形状
    logger.info("=== Matrix Shapes ===")
    matrix_shapes_table = pd.DataFrame({
        "Matrix": list(matrices.keys()),
        "Shape": [str(matrix.shape) for matrix in matrices.values()]
    })
    logger.info("\n" + tabulate(matrix_shapes_table, headers="keys", tablefmt="grid", showindex=False))


app = typer.Typer()


@app.command()
def generate_graph_info(
    edge_file: Annotated[
        Path,
        typer.Argument(help="边数据文件路径，包含电网边信息"),
    ] = RAW_DATA_DIR / "european/powergridEU_E.csv",
    coord_file: Annotated[
        Path,
        typer.Argument(help="节点坐标数据文件路径，包含节点位置信息"),
    ] = RAW_DATA_DIR / "european/powergridEU_V.csv",
    log_file: Annotated[
        Path,
        typer.Argument(help="日志文件保存路径"),
    ] = LOGS_DIR / "graph_info.log",
):
    """
    生成并保存图的信息并展示中间矩阵形状
    """
    # 设置日志文件输出
    log_file = LOGS_DIR / "graph_info.log"
    
    # 加载数据
    edges, coords = load_data(edge_file, coord_file)

    # 创建无向图
    graph = ig.Graph.TupleList(edges, directed=False)
    graph.vs["name"] = [int(v) for v in graph.vs["name"]]  # 节点名转为整数

    # 计算图属性
    graph_properties = compute_graph_properties(graph)

    # 加载中间矩阵
    matrices = load_intermediate_matrices()

    # 输出日志和生成表格
    log_results(graph_properties, matrices, log_file)

    logger.success("Graph information has been saved successfully.")


if __name__ == "__main__":
    app()
