from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import igraph as ig
from IPython.display import Markdown, display
from loguru import logger
import typer

from src.config import INTERIM_DATA_DIR, REPORTS_DIR

# =============================================================================
# 数据加载函数
# =============================================================================


def load_texas_networks() -> Tuple[ig.Graph, ig.Graph, ig.Graph]:
    """
    加载德克萨斯州电力和天然气网络图结构

    Returns
    -------
    Tuple[ig.Graph, ig.Graph, ig.Graph]
        电力网络图, 天然气网络图, 融合网络图
    """
    logger.info("加载德克萨斯州电力-天然气网络数据...")

    # 图结构位置
    electric_graph_path = INTERIM_DATA_DIR / "Texas7k_Gas/electric_graph.pkl"
    gas_graph_path = INTERIM_DATA_DIR / "Texas7k_Gas/gas_graph.pkl"
    merged_graph_path = INTERIM_DATA_DIR / "Texas7k_Gas/merged_graph.pkl"

    # 读取图结构
    electric_graph = ig.Graph.Read_Pickle(electric_graph_path)
    gas_graph = ig.Graph.Read_Pickle(gas_graph_path)
    merged_graph = ig.Graph.Read_Pickle(merged_graph_path)

    logger.success(
        f"成功加载电力网络({electric_graph.vcount()}节点, {electric_graph.ecount()}边)和天然气网络({gas_graph.vcount()}节点, {gas_graph.ecount()}边)"
    )
    return electric_graph, gas_graph, merged_graph


# =============================================================================
# 图分析辅助函数
# =============================================================================


def safe_attr(vertex: ig.Vertex, attr: str, default=None) -> any:
    """
    安全地获取顶点属性，处理属性不存在的情况

    Parameters
    ----------
    vertex : ig.Vertex
        图顶点对象
    attr : str
        属性名称
    default : any, optional
        属性不存在时的默认值

    Returns
    -------
    any
        顶点属性值或默认值
    """
    try:
        return vertex[attr] if attr in vertex.attributes() else default
    except Exception:
        return default


def compute_graph_properties(graph: ig.Graph) -> Dict:
    """
    计算图的基本网络特性属性

    Parameters
    ----------
    graph : ig.Graph
        要分析的图

    Returns
    -------
    Dict
        包含图基本属性的字典
    """
    try:
        # 基础图论指标
        num_nodes = graph.vcount()
        num_edges = graph.ecount()
        density = graph.density()
        max_degree = max(graph.degree()) if num_nodes > 0 else 0
        avg_degree = sum(graph.degree()) / num_nodes if num_nodes > 0 else 0

        # 尝试计算可能较耗时的指标
        try:
            avg_clustering = graph.transitivity_avglocal_undirected()
        except Exception:
            avg_clustering = None

        try:
            num_components = len(graph.components())
        except Exception:
            num_components = None

        # 区分节点类型
        try:
            electric_nodes = sum(1 for v in graph.vs if safe_attr(v, "is_electric_node", False))
        except Exception:
            electric_nodes = 0

        try:
            gas_nodes = sum(1 for v in graph.vs if safe_attr(v, "is_gas_node", False))
        except Exception:
            gas_nodes = 0

        # 区分边类型
        try:
            electric_edges = sum(1 for e in graph.es if safe_attr(e, "is_electric_edge", False))
        except Exception:
            electric_edges = 0

        try:
            gas_edges = sum(1 for e in graph.es if safe_attr(e, "is_gas_edge", False))
        except Exception:
            gas_edges = 0

        try:
            coupling_edges = sum(1 for e in graph.es if safe_attr(e, "is_coupling_edge", False))
        except Exception:
            coupling_edges = 0

        return {
            "总节点数": num_nodes,
            "总边数": num_edges,
            "网络密度": density,
            "最大度": max_degree,
            "平均度": avg_degree,
            "平均聚类系数": avg_clustering,
            "连通分量数": num_components,
            "电力节点数": electric_nodes,
            "天然气节点数": gas_nodes,
            "电力边数": electric_edges,
            "天然气边数": gas_edges,
            "耦合边数": coupling_edges,
        }
    except Exception as e:
        logger.error(f"计算图属性时出错: {e}")
        return {}


def get_node_attributes_coverage(graph: ig.Graph) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    分析节点属性的覆盖情况并分类

    Parameters
    ----------
    graph : ig.Graph
        要分析的图

    Returns
    -------
    Tuple[Dict, Dict, Dict, Dict]
        通用属性列表, 电力特有属性, 天然气特有属性, 属性覆盖率
    """
    # 分离电力和天然气节点
    electric_nodes = [v for v in graph.vs if safe_attr(v, "is_electric_node", False)]
    gas_nodes = [v for v in graph.vs if safe_attr(v, "is_gas_node", False)]

    # 获取所有属性
    all_attributes = set()
    for v in graph.vs:
        all_attributes.update(v.attributes().keys())

    # 计算覆盖率
    electric_coverage = {}
    gas_coverage = {}

    for attr in sorted(all_attributes):
        electric_count = sum(
            1 for v in electric_nodes if attr in v.attributes() and safe_attr(v, attr) is not None
        )
        gas_count = sum(
            1 for v in gas_nodes if attr in v.attributes() and safe_attr(v, attr) is not None
        )

        electric_coverage[attr] = electric_count / len(electric_nodes) if electric_nodes else 0
        gas_coverage[attr] = gas_count / len(gas_nodes) if gas_nodes else 0

    # 分类属性
    common_attrs = []
    electric_attrs = []
    gas_attrs = []

    for attr in sorted(all_attributes):
        if electric_coverage[attr] > 0.5 and gas_coverage[attr] > 0.5:
            common_attrs.append(attr)
        elif electric_coverage[attr] > 0.5:
            electric_attrs.append(attr)
        elif gas_coverage[attr] > 0.5:
            gas_attrs.append(attr)
        else:
            # 低覆盖率属性也放入通用属性
            common_attrs.append(attr)

    # 合并覆盖率
    coverage = {}
    for attr in all_attributes:
        coverage[attr] = {"electric": electric_coverage[attr], "gas": gas_coverage[attr]}

    return common_attrs, electric_attrs, gas_attrs, coverage


def analyze_node_type_distribution(graph: ig.Graph) -> Tuple[Dict, Dict]:
    """
    分析发电机类型和天然气节点类型分布

    Parameters
    ----------
    graph : ig.Graph
        要分析的图

    Returns
    -------
    Tuple[Dict, Dict]
        发电机类型分布, 天然气节点类型分布
    """
    # 电力节点
    electric_nodes = [v for v in graph.vs if safe_attr(v, "is_electric_node", False)]

    # 发电机类型分布
    gen_types = {}
    for v in electric_nodes:
        if safe_attr(v, "has_generator", False):
            gen_type = safe_attr(v, "gen_type", "Unknown")
            gen_types[gen_type] = gen_types.get(gen_type, 0) + 1

    # 天然气节点
    gas_nodes = [v for v in graph.vs if safe_attr(v, "is_gas_node", False)]

    # 天然气节点类型分布
    node_types = {}
    for v in gas_nodes:
        node_type = safe_attr(v, "node_type", "Unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1

    return gen_types, node_types


def get_node_samples(graph: ig.Graph, sample_size: int = 5) -> Tuple[List, List]:
    """
    获取电力和天然气节点的样本

    Parameters
    ----------
    graph : ig.Graph
        要分析的图
    sample_size : int, optional
        样本大小

    Returns
    -------
    Tuple[List, List]
        电力节点样本, 天然气节点样本
    """
    electric_nodes = [v for v in graph.vs if safe_attr(v, "is_electric_node", False)]
    gas_nodes = [v for v in graph.vs if safe_attr(v, "is_gas_node", False)]

    electric_samples = electric_nodes[: min(sample_size, len(electric_nodes))]
    gas_samples = gas_nodes[: min(sample_size, len(gas_nodes))]

    return electric_samples, gas_samples


# =============================================================================
# 生成Markdown内容函数
# =============================================================================


def generate_attribute_description_tables(
    common_attrs: List[str], electric_attrs: List[str], gas_attrs: List[str], coverage: Dict
) -> List[str]:
    """
    生成属性描述表格的Markdown内容

    Parameters
    ----------
    common_attrs : List[str]
        通用属性列表
    electric_attrs : List[str]
        电力特有属性
    gas_attrs : List[str]
        天然气特有属性
    coverage : Dict
        属性覆盖率

    Returns
    -------
    List[str]
        Markdown内容列表
    """
    md_content = []

    # 属性说明字典
    attr_descriptions = {
        "name": "节点名称或标识符",
        "label": "节点描述性标签",
        "latitude": "节点纬度坐标",
        "longitude": "节点经度坐标",
        "substation_id": "关联变电站ID",
        "degree": "节点连接边的数量",
        "is_electric_node": "是否为电力节点",
        "is_gas_node": "是否为天然气节点",
        # 电力属性
        "base_kv": "基准电压(千伏)",
        "bus_type": "母线类型(1=PQ负荷节点, 2=PV发电节点, 3=平衡节点, 4=孤立节点)",
        "pd": "有功负荷(MW)",
        "qd": "无功负荷(MVar)",
        "has_generator": "是否有发电机",
        "total_gen_capacity": "节点发电机总容量(MW)",
        "gen_pg": "发电机有功出力(MW)",
        "gen_qg": "发电机无功出力(MVar)",
        "gen_pmax": "发电机最大有功出力(MW)",
        "gen_pmin": "发电机最小有功出力(MW)",
        "gen_type": "发电机类型",
        "gen_fuel": "发电机燃料类型",
        # 天然气属性
        "node_type": "天然气节点类型",
        "gas_load_p": "天然气节点压力",
        "gas_load_qf": "天然气流量负荷",
        "qf_max": "最大流量",
        "qf_min": "最小流量",
        "is_slack": "是否为天然气网络平衡节点",
    }

    # 通用属性表格
    md_content.append("### 2.1 通用属性\n")
    md_content.append("| 属性名 | 电力节点覆盖率 | 天然气节点覆盖率 | 说明 |")
    md_content.append("|--------|--------------|----------------|------|")

    for attr in common_attrs:
        desc = attr_descriptions.get(attr, "")
        md_content.append(
            f"| {attr} | {coverage[attr]['electric'] * 100:.1f}% | {coverage[attr]['gas'] * 100:.1f}% | {desc} |"
        )

    # 电力节点特有属性表格
    md_content.append("\n### 2.2 电力节点特有属性\n")
    md_content.append("| 属性名 | 电力节点覆盖率 | 说明 |")
    md_content.append("|--------|--------------|------|")

    for attr in electric_attrs:
        desc = attr_descriptions.get(attr, "")
        md_content.append(f"| {attr} | {coverage[attr]['electric'] * 100:.1f}% | {desc} |")

    # 天然气节点特有属性表格
    md_content.append("\n### 2.3 天然气节点特有属性\n")
    md_content.append("| 属性名 | 天然气节点覆盖率 | 说明 |")
    md_content.append("|--------|----------------|------|")

    for attr in gas_attrs:
        desc = attr_descriptions.get(attr, "")
        md_content.append(f"| {attr} | {coverage[attr]['gas'] * 100:.1f}% | {desc} |")

    return md_content


def generate_node_samples_tables(
    electric_samples: List[ig.Vertex], gas_samples: List[ig.Vertex]
) -> List[str]:
    """
    生成节点样本表格的Markdown内容

    Parameters
    ----------
    electric_samples : List[ig.Vertex]
        电力节点样本
    gas_samples : List[ig.Vertex]
        天然气节点样本

    Returns
    -------
    List[str]
        Markdown内容列表
    """
    md_content = []

    # 电力节点样本
    md_content.append("### 3.1 电力节点样本\n")
    if electric_samples:
        # 选择最有代表性的几个属性来展示
        key_electric_attrs = ["name", "bus_type", "base_kv", "pd", "qd", "has_generator"]
        if any(safe_attr(v, "has_generator", False) for v in electric_samples):
            key_electric_attrs.extend(["gen_type", "gen_pmax"])

        # 格式化表格
        md_content.append("| 节点ID | " + " | ".join(key_electric_attrs) + " |")
        md_content.append(
            "|" + "-" * 8 + "|" + "|".join(["-" * 12] * len(key_electric_attrs)) + "|"
        )

        for v in electric_samples:
            row = [str(v.index)]
            for attr in key_electric_attrs:
                value = safe_attr(v, attr, "")
                row.append(str(value))
            md_content.append("| " + " | ".join(row) + " |")

    # 天然气节点样本
    md_content.append("\n### 3.2 天然气节点样本\n")
    if gas_samples:
        # 选择最有代表性的几个属性来展示
        key_gas_attrs = ["name", "node_type", "gas_load_p", "gas_load_qf", "is_slack"]

        # 格式化表格
        md_content.append("| 节点ID | " + " | ".join(key_gas_attrs) + " |")
        md_content.append("|" + "-" * 8 + "|" + "|".join(["-" * 12] * len(key_gas_attrs)) + "|")

        for v in gas_samples:
            row = [str(v.index)]
            for attr in key_gas_attrs:
                value = safe_attr(v, attr, "")
                row.append(str(value))
            md_content.append("| " + " | ".join(row) + " |")

    return md_content


def generate_node_type_tables(gen_types: Dict, node_types: Dict) -> List[str]:
    """
    生成节点类型分布表格的Markdown内容

    Parameters
    ----------
    gen_types : Dict
        发电机类型分布
    node_types : Dict
        天然气节点类型分布

    Returns
    -------
    List[str]
        Markdown内容列表
    """
    md_content = []

    # 发电机类型分布
    if gen_types:
        md_content.append("### 4.1 发电机类型分布\n")
        md_content.append("| 发电机类型 | 数量 | 占比 |")
        md_content.append("|-----------|------|------|")

        total_gens = sum(gen_types.values())
        for gen_type, count in sorted(gen_types.items(), key=lambda x: x[1], reverse=True):
            gen_desc = {
                "CT": "燃气轮机 (Combustion Turbine)",
                "WT": "风力发电机 (Wind Turbine)",
                "GT": "燃气透平机 (Gas Turbine)",
                "ST": "蒸汽轮机 (Steam Turbine)",
                "CA": "联合循环机组 (Combined Cycle)",
                "UN": "未知类型 (Unknown)",
                "IC": "内燃机 (Internal Combustion)",
                "PV": "光伏发电 (Photovoltaic)",
                "HY": "水力发电 (Hydro)",
                "CS": "集中式太阳能 (Concentrated Solar)",
                "BA": "电池储能 (Battery)",
            }.get(gen_type, gen_type)

            md_content.append(f"| {gen_desc} | {count} | {count / total_gens * 100:.1f}% |")

    # 天然气节点类型分布
    if node_types:
        md_content.append("\n### 4.2 天然气节点类型分布\n")
        md_content.append("| 节点类型 | 数量 | 占比 |")
        md_content.append("|---------|------|------|")

        total_nodes = sum(node_types.values())
        for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            node_desc = {
                "Junction": "管道连接点",
                "Gas Load": "天然气负荷点",
                "Processing Plant": "天然气处理厂",
                "Electric Load": "电力负荷点(耦合点)",
                "Storage Reservoir": "天然气储存库",
                "Import/Export Point": "进出口点",
                "Trading Hub": "交易中心",
                "Processing Plant - Slack": "平衡节点处理厂",
            }.get(node_type, node_type)

            md_content.append(f"| {node_desc} | {count} | {count / total_nodes * 100:.1f}% |")

    return md_content


# =============================================================================
# 主要分析函数
# =============================================================================


def analyze_network_attributes(
    graph: ig.Graph, output_file: Optional[Path] = None, sample_size: int = 5
) -> str:
    """
    分析网络节点属性并以优雅的方式展示，可选择保存为Markdown文件

    Parameters:
    -----------
    graph : ig.Graph
        要分析的图
    output_file : Path, optional
        输出的Markdown文件路径
    sample_size : int
        每种节点类型要展示的样本数量

    Returns:
    --------
    str
        生成的Markdown内容
    """
    logger.info(f"开始分析图结构 ({graph.vcount()}节点, {graph.ecount()}边)...")

    # 创建Markdown内容
    md_content = []
    md_content.append("# 电力-天然气融合网络节点属性分析\n")

    # 1. 计算网络基本特性
    graph_properties = compute_graph_properties(graph)

    # 添加网络概述
    md_content.append("## 1. 网络概述\n")
    for prop, value in graph_properties.items():
        md_content.append(f"- **{prop}**: {value}")
    md_content.append("")

    # 2. 分析节点属性
    md_content.append("## 2. 节点属性分类\n")
    common_attrs, electric_attrs, gas_attrs, coverage = get_node_attributes_coverage(graph)
    md_content.extend(
        generate_attribute_description_tables(common_attrs, electric_attrs, gas_attrs, coverage)
    )

    # 3. 提取节点样本
    md_content.append("\n## 3. 节点样本分析\n")
    electric_samples, gas_samples = get_node_samples(graph, sample_size)
    md_content.extend(generate_node_samples_tables(electric_samples, gas_samples))

    # 4. 分析节点类型分布
    md_content.append("\n## 4. 节点类型分析\n")
    gen_types, node_types = analyze_node_type_distribution(graph)
    md_content.extend(generate_node_type_tables(gen_types, node_types))

    # 生成完整内容
    full_content = "\n".join(md_content)

    # 显示结果
    display(Markdown(full_content))

    # 保存为markdown文件
    if output_file:
        logger.info(f"保存分析结果至 {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_content)
        logger.success(f"分析结果已保存至: {output_file}")

    return full_content


# =============================================================================
# 命令行接口
# =============================================================================

app = typer.Typer()


@app.command()
def analyze_texas_network(
    output_dir: Annotated[
        Path,
        typer.Option(help="分析结果输出目录"),
    ] = REPORTS_DIR,
    sample_size: Annotated[
        int,
        typer.Option(help="节点样本展示数量"),
    ] = 5,
    log_file: Annotated[
        Optional[Path],
        typer.Option(help="日志文件路径"),
    ] = None,
):
    """
    分析Texas电力-天然气融合网络的结构和节点属性，并生成详细报告
    """
    # 设置日志
    if log_file:
        logger.add(log_file, rotation="10 MB")

    # 加载网络
    electric_graph, gas_graph, merged_graph = load_texas_networks()

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分析并保存结果
    merged_output = output_dir / "texas_merged_network.md"
    electric_output = output_dir / "texas_electric_network.md"
    gas_output = output_dir / "texas_gas_network.md"

    # 分析融合网络
    logger.info("分析融合网络...")
    analyze_network_attributes(merged_graph, merged_output, sample_size)

    # 分析电力网络
    logger.info("分析电力网络...")
    analyze_network_attributes(electric_graph, electric_output, sample_size)

    # 分析天然气网络
    logger.info("分析天然气网络...")
    analyze_network_attributes(gas_graph, gas_output, sample_size)

    logger.success("所有网络分析完成")


if __name__ == "__main__":
    app()
