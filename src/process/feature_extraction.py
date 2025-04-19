"""
电力-天然气耦合网络节点和边特征处理模块

本模块负责处理电力-天然气耦合网络的节点和边特征，包括:
1. 读取电力和天然气网络的图结构
2. 提取和归一化节点特征
3. 处理和填充边特征
4. 保存处理后的特征矩阵
"""

import numpy as np
import pandas as pd
import igraph as ig
import scipy.sparse as sp
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from loguru import logger
import typer
from typing_extensions import Annotated

from src.config import INTERIM_DATA_DIR, LOGS_DIR


class FeatureExtractor:
    """
    电力-天然气耦合网络特征提取器

    该类负责提取和处理电力-天然气耦合网络的节点和边特征，
    包括特征归一化、缺失值处理、独热编码等。
    """

    def __init__(self, graph_path: Path, output_dir: Path):
        """
        初始化特征提取器

        Parameters
        ----------
        graph_path : Path
            电力-天然气耦合网络图结构文件路径
        output_dir : Path
            处理结果保存目录路径
        """
        self.graph_path = graph_path
        self.output_dir = output_dir
        
    def process(self):
        """
        执行特征处理的完整流程

        该方法依次执行以下步骤:
        1. 读取图结构
        2. 提取节点和边特征
        3. 归一化特征
        4. 处理缺失值
        5. 构建特征矩阵
        6. 保存处理结果

        Returns
        -------
        tuple
            返回处理结果，包括节点特征矩阵和边特征矩阵
        """
        logger.info("启动特征处理流程")

        # 1. 读取图结构
        g = self._load_graph()
        
        # 2. 提取节点和边数据
        node_df, edge_df = self._extract_node_edge_data(g)
        
        # 3. 归一化节点特征
        node_df = self._normalize_node_features(node_df)
        
        # 4. 独热编码分类特征
        node_df = self._one_hot_encode_categorical_features(node_df)
        
        # 5. 填充边特征缺失值
        edge_df = self._fill_edge_feature_missing_values(edge_df)
        
        # 6. 构建特征矩阵
        node_features, edge_features = self._build_feature_matrices(node_df, edge_df)
        
        # 7. 保存处理结果
        self._save_processed_data(node_features, edge_features)
        
        logger.success("特征处理流程已完成")
        return node_features, edge_features
    
    def _load_graph(self):
        """
        从保存的图结构文件中加载图
        
        Returns
        -------
        igraph.Graph
            返回加载的图对象，包含节点和边的详细信息
        """
        logger.info(f"正在从{self.graph_path}加载图结构")
        g = ig.Graph.Read_Pickle(self.graph_path)
        logger.info(f"图结构加载完成，包含{g.vcount()}个节点和{g.ecount()}个边")
        return g
    
    def _extract_node_edge_data(self, g):
        """
        从图结构中提取节点和边的数据
        
        Parameters
        ----------
        g : igraph.Graph
            输入的图结构
            
        Returns
        -------
        tuple
            返回节点DataFrame和边DataFrame
        """
        logger.info("正在提取节点和边数据")
        
        # 提取节点数据
        node_attributes = g.vs.attributes()
        node_data = {}
        for attr in node_attributes:
            node_data[attr] = g.vs[attr]
        node_df = pd.DataFrame(node_data)
        
        # 提取边数据
        edge_attributes = g.es.attributes()
        edge_data = {}
        for attr in edge_attributes:
            edge_data[attr] = g.es[attr]
        edge_df = pd.DataFrame(edge_data)
        
        # 统计电力/天然气节点和边数量
        num_electric_nodes = node_df['is_electric_node'].sum() if 'is_electric_node' in node_df else 0
        num_gas_nodes = node_df['is_gas_node'].sum() if 'is_gas_node' in node_df else 0
        
        logger.info(f"节点总数: {len(node_df)}, 电力节点: {num_electric_nodes}, 天然气节点: {num_gas_nodes}")
        logger.info(f"边总数: {len(edge_df)}")
        
        return node_df, edge_df
    
    def _normalize_node_features(self, node_df):
        """
        归一化节点特征
        
        Parameters
        ----------
        node_df : pandas.DataFrame
            节点特征DataFrame
            
        Returns
        -------
        pandas.DataFrame
            返回归一化后的节点特征DataFrame
        """
        logger.info("正在归一化节点特征")
        
        # 对电力节点特征进行归一化
        electric_features = ['base_kv', 'pd', 'qd', 'total_gen_capacity']
        electric_nodes = node_df['is_electric_node'] == True
        if electric_nodes.any() and all(feat in node_df for feat in electric_features):
            electric_scaler = StandardScaler()
            node_df.loc[electric_nodes, electric_features] = electric_scaler.fit_transform(
                node_df.loc[electric_nodes, electric_features]
            )
        
        # 对天然气节点特征进行归一化
        gas_features = ['gas_load_p', 'gas_load_qf', 'qf_max', 'qf_min', 'degree']
        gas_nodes = node_df['is_gas_node'] == True
        if gas_nodes.any() and all(feat in node_df for feat in gas_features):
            gas_scaler = StandardScaler()
            node_df.loc[gas_nodes, gas_features] = gas_scaler.fit_transform(
                node_df.loc[gas_nodes, gas_features]
            )
        
        # 对经纬度特征进行归一化
        location_features = ['latitude', 'longitude']
        if all(feat in node_df for feat in location_features):
            location_scaler = StandardScaler()
            node_df[location_features] = location_scaler.fit_transform(node_df[location_features])
        
        logger.info("节点特征归一化完成")
        return node_df
    
    def _one_hot_encode_categorical_features(self, node_df):
        """
        对分类特征进行独热编码
        
        Parameters
        ----------
        node_df : pandas.DataFrame
            节点特征DataFrame
            
        Returns
        -------
        pandas.DataFrame
            返回经过独热编码处理后的节点特征DataFrame
        """
        logger.info("正在对分类特征进行独热编码")
        
        # 对电力节点的bus_type进行独热编码
        if 'bus_type' in node_df:
            bus_type_encoded = pd.get_dummies(node_df['bus_type'], prefix='bus_type')
            node_df = pd.concat([node_df, bus_type_encoded], axis=1)
            logger.info(f"电力节点bus_type独热编码完成，生成{len(bus_type_encoded.columns)}个特征")
        
        # 对天然气节点的node_type进行独热编码
        if 'node_type' in node_df:
            node_type_encoded = pd.get_dummies(node_df['node_type'], prefix='node_type')
            node_df = pd.concat([node_df, node_type_encoded], axis=1)
            logger.info(f"天然气节点node_type独热编码完成，生成{len(node_type_encoded.columns)}个特征")
        
        # 添加节点类型标记特征
        node_df['node_domain'] = 0  # 默认为0
        if 'is_electric_node' in node_df:
            node_df.loc[node_df['is_electric_node'] == True, 'node_domain'] = 1  # 电力节点标记为1
        if 'is_gas_node' in node_df:
            node_df.loc[node_df['is_gas_node'] == True, 'node_domain'] = 2  # 天然气节点标记为2
        
        return node_df
    
    def _fill_edge_feature_missing_values(self, edge_df):
        """
        填充边特征的缺失值
        
        Parameters
        ----------
        edge_df : pandas.DataFrame
            边特征DataFrame
            
        Returns
        -------
        pandas.DataFrame
            返回填充缺失值后的边特征DataFrame
        """
        logger.info("正在填充边特征缺失值")
        
        # 电力边特征和天然气边特征分开填充缺失值
        if 'is_electric_edge' in edge_df and 'is_gas_edge' in edge_df:
            electric_edges = edge_df['is_electric_edge'] == True
            gas_edges = edge_df['is_gas_edge'] == True
            
            # 电力边特征缺失值填充
            for feat in ['r', 'x', 'b', 'capacity']:
                if feat in edge_df.columns:
                    # 用电力边的平均值填充电力边的缺失值
                    if electric_edges.any() and edge_df.loc[electric_edges, feat].isna().any():
                        mean_val = edge_df.loc[electric_edges, feat].mean()
                        edge_df.loc[electric_edges, feat] = edge_df.loc[electric_edges, feat].fillna(mean_val)
                    # 对于天然气边，这些特征应该填0
                    if gas_edges.any():
                        edge_df.loc[gas_edges, feat] = edge_df.loc[gas_edges, feat].fillna(0)
            
            # 天然气边特征缺失值填充
            for feat in ['q', 'length', 'diameter', 'friction_factor', 'k']:
                if feat in edge_df.columns:
                    # 用天然气边的平均值填充天然气边的缺失值
                    if gas_edges.any() and edge_df.loc[gas_edges, feat].isna().any():
                        mean_val = edge_df.loc[gas_edges, feat].mean()
                        edge_df.loc[gas_edges, feat] = edge_df.loc[gas_edges, feat].fillna(mean_val)
                    # 对于电力边，这些特征应该填0
                    if electric_edges.any():
                        edge_df.loc[electric_edges, feat] = edge_df.loc[electric_edges, feat].fillna(0)
        
        logger.info("边特征缺失值填充完成")
        
        # 检查是否还有缺失值
        remaining_nulls = edge_df.isnull().sum()
        if remaining_nulls.sum() > 0:
            logger.warning(f"边特征仍有缺失值: {remaining_nulls[remaining_nulls > 0]}")
        
        return edge_df
    
    def _build_feature_matrices(self, node_df, edge_df):
        """
        构建节点和边特征矩阵
        
        Parameters
        ----------
        node_df : pandas.DataFrame
            节点特征DataFrame
        edge_df : pandas.DataFrame
            边特征DataFrame
            
        Returns
        -------
        tuple
            返回节点特征矩阵和边特征矩阵
        """
        logger.info("正在构建特征矩阵")
        
        # 构建节点特征矩阵
        final_node_features = []
        
        # 添加位置特征
        location_features = ['latitude', 'longitude']
        final_node_features.extend(location_features)
        
        # 添加电力节点特征
        electric_feats = ['base_kv', 'pd', 'qd', 'has_generator', 'total_gen_capacity'] 
        final_node_features.extend(electric_feats)
        
        # 添加bus_type独热编码特征
        bus_type_cols = [col for col in node_df.columns if col.startswith('bus_type_')]
        final_node_features.extend(bus_type_cols)
        
        # 添加天然气节点特征
        gas_feats = ['gas_load_p', 'gas_load_qf', 'is_slack', 'qf_max', 'qf_min', 'degree']
        final_node_features.extend(gas_feats)
        
        # 添加node_type独热编码特征
        node_type_cols = [col for col in node_df.columns if col.startswith('node_type_')]
        final_node_features.extend(node_type_cols)
        
        # 添加节点类型标记特征
        final_node_features.append('node_domain')
        
        # 构建最终节点特征矩阵
        node_feature_matrix = node_df[final_node_features]
        logger.info(f"节点特征矩阵形状: {node_feature_matrix.shape}")
        
        # 构建边特征矩阵
        edge_features = ['r', 'x', 'b', 'capacity', 'q', 'length', 'diameter', 'friction_factor', 'k']
        edge_feature_matrix = edge_df[edge_features]
        logger.info(f"边特征矩阵形状: {edge_feature_matrix.shape}")
        
        # 检查节点特征矩阵是否有缺失值
        node_nulls = node_feature_matrix.isnull().sum()
        if node_nulls.sum() > 0:
            logger.warning(f"节点特征矩阵有缺失值: {node_nulls[node_nulls > 0]}")
            # 对节点特征矩阵中的缺失值进行填充
            for col in node_feature_matrix.columns:
                if node_feature_matrix[col].isna().any():
                    if node_feature_matrix[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        # 数值型特征用0填充
                        node_feature_matrix[col] = node_feature_matrix[col].fillna(0)
                    else:
                        # 非数值型特征用最频繁的值填充
                        node_feature_matrix[col] = node_feature_matrix[col].fillna(node_feature_matrix[col].mode()[0])
        
        # 检查边特征矩阵是否有缺失值
        edge_nulls = edge_feature_matrix.isnull().sum()
        if edge_nulls.sum() > 0:
            logger.warning(f"边特征矩阵有缺失值: {edge_nulls[edge_nulls > 0]}")
            # 对边特征矩阵中的缺失值填充为0
            edge_feature_matrix = edge_feature_matrix.fillna(0)
        
        logger.info("特征矩阵构建完成")
        
        return node_feature_matrix, edge_feature_matrix
    
    def _save_processed_data(self, node_features, edge_features):
        """
        保存处理结果到指定的输出目录
        
        Parameters
        ----------
        node_features : pandas.DataFrame
            节点特征矩阵
        edge_features : pandas.DataFrame
            边特征矩阵
        """
        logger.info("正在保存处理结果")
        
        # 创建输出目录（如果不存在）
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存节点特征矩阵
        node_features_path = self.output_dir / "node_features.npy"
        np.save(node_features_path, node_features.values)
        
        # 保存边特征矩阵
        edge_features_path = self.output_dir / "edge_features.npy"
        np.save(edge_features_path, edge_features.values)
        
        # 保存特征列名（用于未来的解释性）
        np.save(self.output_dir / "node_feature_names.npy", np.array(node_features.columns))
        np.save(self.output_dir / "edge_feature_names.npy", np.array(edge_features.columns))
        
        logger.success(f"所有处理结果已保存到: {self.output_dir}")


app = typer.Typer()


@app.command()
def main(
    graph_path: Annotated[
        Path, 
        typer.Argument(help="电力-天然气耦合网络图结构文件路径")
    ] = INTERIM_DATA_DIR / "Texas7k_Gas/merged_graph.pkl",
    output_dir: Annotated[
        Path, 
        typer.Argument(help="处理结果保存目录路径")
    ] = INTERIM_DATA_DIR / "Texas7k_Gas/",
):
    """
    处理电力-天然气耦合网络的节点和边特征，并将结果保存至指定目录。
    """
    log_file = LOGS_DIR / "feature_extraction.log"
    logger.add(log_file, mode="w")  # 表示覆盖保存
    logger.info("开始特征处理")
    
    # 初始化特征提取器
    extractor = FeatureExtractor(graph_path=graph_path, output_dir=output_dir)
    
    # 执行特征处理
    node_features, edge_features = extractor.process()
    
    # 处理完成后记录日志
    logger.success("特征处理完成")
    logger.info(f"处理结果已保存至: {output_dir}")


if __name__ == "__main__":
    app()