import json
from pathlib import Path
import re

import igraph as ig
from loguru import logger
import numpy as np
import typer
from typing_extensions import Annotated

from src.config import INTERIM_DATA_DIR, LOGS_DIR, RAW_DATA_DIR


class NetworkProcessor:
    def __init__(self, electric_file: Path, gas_file: Path, aux_file: Path, output_dir: Path):
        """
        初始化 NetworkProcessor。
        Parameters
        ----------
        electric_file : Path
            电力系统数据文件路径（如 .RAW 文件）。
        gas_file : Path
            天然气系统数据文件路径（如 JSON 文件）。
        aux_file : Path
            电力系统辅助数据文件路径（如 .AUX 文件）。
        output_dir : Path
            输出目录路径，用于保存处理结果。
        """
        self.electric_file = electric_file
        self.gas_file = gas_file
        self.aux_file = aux_file
        self.output_dir = output_dir

    def process(self):
        """
        执行数据处理的完整流程。
        该方法依次执行以下步骤：
        1. 构建电力系统图
        2. 构建天然气系统图
        3. 合并两个图
        4. 提取基本特征
        5. 保存处理结果到指定输出目录
        Returns
        -------
        tuple
            返回处理结果，包括合并后的图
        """
        logger.info("启动数据处理流程")
        # 1. 构建电力系统图
        electric_graph = self._build_electric_graph()
        # 2. 构建天然气系统图
        gas_graph = self._build_gas_graph()
        # 3. 合并两个图
        merged_graph = self._merge_graphs(electric_graph, gas_graph)
        # 4. 提取基本特征
        # node_features, edge_features = self._extract_basic_features(merged_graph)
        # 5. 保存处理结果
        self._save_processed_data(electric_graph, gas_graph, merged_graph)

        # 添加详细的统计信息输出
        logger.info(
            f"电力系统图：{electric_graph.vcount()} 个节点，{electric_graph.ecount()} 条边"
        )
        logger.info(f"天然气系统图：{gas_graph.vcount()} 个节点，{gas_graph.ecount()} 条边")
        logger.info(f"合并后的图：{merged_graph.vcount()} 个节点，{merged_graph.ecount()} 条边")

        # 统计不同类型的节点
        try:
            electric_nodes = sum(1 for v in merged_graph.vs if v["is_electric_node"])
        except (KeyError, AttributeError):
            electric_nodes = 0

        try:
            gas_nodes = sum(1 for v in merged_graph.vs if v["is_gas_node"])
        except (KeyError, AttributeError):
            gas_nodes = 0

        gas_node_types = {}
        for v in merged_graph.vs:
            try:
                if v["is_gas_node"]:
                    try:
                        node_type = v["node_type"]
                    except (KeyError, AttributeError):
                        node_type = "Unknown"
                    gas_node_types[node_type] = gas_node_types.get(node_type, 0) + 1
            except (KeyError, AttributeError):
                continue

        logger.info(f"合并图中的电力节点数：{electric_nodes}")
        logger.info(f"合并图中的天然气节点数：{gas_nodes}")
        logger.info(f"天然气节点类型分布：{gas_node_types}")

        # 统计不同类型的边
        try:
            electric_edges = sum(1 for e in merged_graph.es if e["is_electric_edge"])
        except (KeyError, AttributeError):
            electric_edges = 0

        try:
            gas_edges = sum(1 for e in merged_graph.es if e["is_gas_edge"])
        except (KeyError, AttributeError):
            gas_edges = 0

        try:
            coupling_edges = sum(1 for e in merged_graph.es if e["is_coupling_edge"])
        except (KeyError, AttributeError):
            coupling_edges = 0

        logger.info(f"合并图中的电力边数：{electric_edges}")
        logger.info(f"合并图中的天然气边数：{gas_edges}")
        logger.info(f"合并图中的耦合边数：{coupling_edges}")

        logger.success("数据处理流程已完成")
        return electric_graph, gas_graph, merged_graph

    def _build_electric_graph(self):
        """
        构建电力系统图。
        从 .m 文件中提取节点和边信息，并构建图。
        Returns
        -------
        igraph.Graph
            返回构建好的电力系统图对象。
        """
        logger.info("正在构建电力系统图")

        # 从 .AUX 文件提取变电站的经纬度坐标
        substation_coords = self._extract_substation_coords(self.aux_file)

        # 打开并读取.m文件
        with open(self.electric_file, "r") as f:
            content = f.read()

        buses = []
        generators = []
        branches = []
        gen_types = {}  # noqa: F841
        gen_fuels = {}  # noqa: F841

        # 解析bus数据
        bus_pattern = r"mpc\.bus\s*=\s*\[(.*?)\];"
        bus_match = re.search(bus_pattern, content, re.DOTALL)

        if bus_match:
            bus_data = bus_match.group(1).strip()
            for line in bus_data.split("\n"):
                line = line.strip()
                if line and not line.startswith("%"):
                    parts = line.split()
                    if len(parts) >= 10:  # 确保有足够的列
                        try:
                            bus_id = int(float(parts[0]))
                            bus_type = int(float(parts[1]))
                            pd = float(parts[2])  # 有功负荷
                            qd = float(parts[3])  # 无功负荷
                            base_kv = float(parts[9])

                            # 获取变电站ID和坐标
                            substation_id = bus_id // 1000
                            latitude, longitude = substation_coords.get(substation_id, (0.0, 0.0))

                            # 获取总线名称（如果有bus_name部分）
                            bus_name = f"Bus {bus_id}"  # 默认名称

                            buses.append(
                                {
                                    "id": bus_id,
                                    "name": bus_name,
                                    "type": bus_type,  # 1=PQ, 2=PV, 3=slack, 4=isolated
                                    "pd": pd,  # 有功负荷
                                    "qd": qd,  # 无功负荷
                                    "base_kv": base_kv,
                                    "latitude": latitude,
                                    "longitude": longitude,
                                    "substation_id": substation_id,
                                }
                            )
                        except (ValueError, IndexError) as e:
                            logger.warning(f"解析总线数据时出错: {line}, 错误: {e}")
        else:
            logger.error("在.m文件中未找到总线数据部分")

        # 解析bus_name数据以获取更准确的名称
        bus_name_pattern = r"mpc\.bus_name\s*=\s*\{(.*?)\};"
        bus_name_match = re.search(bus_name_pattern, content, re.DOTALL)

        if bus_name_match:
            bus_names = bus_name_match.group(1).strip().split("\n")
            for i, name_line in enumerate(bus_names):
                if name_line.strip() and not name_line.strip().startswith("%"):
                    name = name_line.strip().strip("';")
                    if i < len(buses):
                        buses[i]["name"] = name

        # 解析generator数据
        gen_pattern = r"mpc\.gen\s*=\s*\[(.*?)\];"
        gen_match = re.search(gen_pattern, content, re.DOTALL)

        if gen_match:
            gen_data = gen_match.group(1).strip()
            for line in gen_data.split("\n"):
                line = line.strip()
                if line and not line.startswith("%"):
                    parts = line.split()
                    if len(parts) >= 10:
                        try:
                            bus_id = int(float(parts[0]))
                            pg = float(parts[1])  # 有功出力
                            qg = float(parts[2])  # 无功出力
                            qmax = float(parts[3])  # 最大无功输出
                            qmin = float(parts[4])  # 最小无功输出
                            vg = float(parts[5])  # 电压设定值
                            mbase = float(parts[6])  # 机组基准容量
                            status = int(float(parts[7]))  # 状态
                            pmax = float(parts[8])  # 最大有功输出
                            pmin = float(parts[9])  # 最小有功输出

                            generators.append(
                                {
                                    "bus_id": bus_id,
                                    "pg": pg,
                                    "qg": qg,
                                    "qmax": qmax,
                                    "qmin": qmin,
                                    "vg": vg,
                                    "mbase": mbase,
                                    "status": status,
                                    "pmax": pmax,
                                    "pmin": pmin,
                                    "type": "Unknown",  # 默认值，稍后可能更新
                                    "fuel": "Unknown",  # 默认值，稍后可能更新
                                }
                            )
                        except (ValueError, IndexError) as e:
                            logger.warning(f"解析发电机数据时出错: {line}, 错误: {e}")
        else:
            logger.warning("在.m文件中未找到发电机数据部分")

        # 解析gentype数据（发电机类型）
        gentype_pattern = r"mpc\.gentype\s*=\s*\{(.*?)\};"
        gentype_match = re.search(gentype_pattern, content, re.DOTALL)

        if gentype_match:
            gentype_data = gentype_match.group(1).strip().split("\n")
            for i, type_line in enumerate(gentype_data):
                if type_line.strip() and not type_line.strip().startswith("%"):
                    gen_type = type_line.strip().strip("';")
                    if i < len(generators):
                        generators[i]["type"] = gen_type

        # 解析genfuel数据（发电机燃料类型）
        genfuel_pattern = r"mpc\.genfuel\s*=\s*\{(.*?)\};"
        genfuel_match = re.search(genfuel_pattern, content, re.DOTALL)

        if genfuel_match:
            genfuel_data = genfuel_match.group(1).strip().split("\n")
            for i, fuel_line in enumerate(genfuel_data):
                if fuel_line.strip() and not fuel_line.strip().startswith("%"):
                    fuel_type = fuel_line.strip().strip("';")
                    if i < len(generators):
                        generators[i]["fuel"] = fuel_type

        # 解析branch数据
        branch_pattern = r"mpc\.branch\s*=\s*\[(.*?)\];"
        branch_match = re.search(branch_pattern, content, re.DOTALL)

        if branch_match:
            branch_data = branch_match.group(1).strip()
            for line in branch_data.split("\n"):
                line = line.strip()
                if line and not line.startswith("%"):
                    parts = line.split()
                    if len(parts) >= 6:  # 至少需要from_bus, to_bus, r, x, b, rateA
                        try:
                            from_bus = int(float(parts[0]))
                            to_bus = int(float(parts[1]))
                            r = float(parts[2])  # 电阻
                            x = float(parts[3])  # 电抗
                            b = float(parts[4])  # 导纳
                            rate_a = float(parts[5])  # 容量限制A

                            branches.append(
                                {
                                    "from": from_bus,
                                    "to": to_bus,
                                    "r": r,
                                    "x": x,
                                    "b": b,
                                    "capacity": rate_a,  # 线路容量
                                }
                            )
                        except (ValueError, IndexError) as e:
                            logger.warning(f"解析分支数据时出错: {line}, 错误: {e}")
        else:
            logger.error("在.m文件中未找到分支数据部分")

        # 构建图
        g = ig.Graph()

        # 添加节点
        for bus in buses:
            g.add_vertex(
                name=str(bus["id"]),
                label=bus["name"],
                latitude=bus["latitude"],
                longitude=bus["longitude"],
                base_kv=bus["base_kv"],
                substation_id=bus["substation_id"],
                is_electric_node=True,  # 标记为电力节点
                bus_type=bus["type"],  # 总线类型
                pd=bus["pd"],  # 有功负荷
                qd=bus["qd"],  # 无功负荷
                has_generator=False,  # 默认值，稍后更新
                total_gen_capacity=0.0,  # 默认值，稍后更新
            )

        # 添加发电机信息到对应的总线
        for gen in generators:
            bus_id = str(gen["bus_id"])
            try:
                v_index = g.vs.find(name=bus_id).index
                g.vs[v_index]["has_generator"] = True
                g.vs[v_index]["gen_pg"] = gen["pg"]
                g.vs[v_index]["gen_qg"] = gen["qg"]
                g.vs[v_index]["gen_pmax"] = gen["pmax"]
                g.vs[v_index]["gen_pmin"] = gen["pmin"]
                g.vs[v_index]["gen_type"] = gen["type"]
                g.vs[v_index]["gen_fuel"] = gen["fuel"]
                g.vs[v_index]["total_gen_capacity"] = gen["pmax"]
            except ValueError:
                logger.warning(f"无法找到与发电机相关的总线 ID: {bus_id}")

        # 添加边前确保所有节点都已添加
        bus_ids = {bus["id"] for bus in buses}
        for branch in branches:
            if branch["from"] in bus_ids and branch["to"] in bus_ids:
                g.add_edge(
                    str(branch["from"]),
                    str(branch["to"]),
                    r=branch["r"],
                    x=branch["x"],
                    b=branch["b"],
                    capacity=branch["capacity"],
                    is_electric_edge=True,
                )

        logger.info(f"电力系统图构建完成，节点数：{g.vcount()}，边数：{g.ecount()}")
        logger.info(f"包含 {sum(v['has_generator'] for v in g.vs)} 个发电机节点")
        return g

    def _extract_substation_coords(self, aux_path):
        """
        从 PowerWorld 的 .AUX 文件中提取变电站的经纬度坐标。

        Parameters
        ----------
        aux_path : Path
            .AUX 文件路径。

        Returns
        -------
        dict
            以变电站编号为键，(纬度,经度)元组为值的字典。
        """
        logger.info(f"从 {aux_path} 提取变电站坐标")
        substation_coords = {}
        in_substation_block = False
        try:
            with open(aux_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "Substation (" in line:
                        in_substation_block = True
                        continue
                    if in_substation_block and "}" in line:
                        in_substation_block = False
                        break
                    if in_substation_block and line.strip().startswith("{"):
                        # 跳过块的起始行
                        continue
                    if in_substation_block:
                        # 使用正则表达式提取字段
                        # 假设字段顺序固定为：Number, Name, IDExtra, Latitude, Longitude, DataMaintainerAssign, DataMaintainerInheritBlock
                        match = re.match(
                            r'(\d+)\s+"(.*?)"\s+"(.*?)"\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+"(.*?)"\s+"(.*?)"',
                            line.strip(),
                        )
                        if match:
                            try:
                                # 提取字段
                                substation_id = int(match.group(1))  # Number
                                name = match.group(2)  # Name  # noqa: F841
                                id_extra = match.group(3)  # IDExtra  # noqa: F841
                                latitude = float(match.group(4))  # Latitude
                                longitude = float(match.group(5))  # Longitude
                                data_maintainer_assign = match.group(  # noqa: F841
                                    6
                                )  # DataMaintainerAssign
                                data_maintainer_inherit_block = match.group(  # noqa: F841
                                    7
                                )  # DataMaintainerInheritBlock
                                # 存储变电站坐标
                                substation_coords[substation_id] = (latitude, longitude)
                            except (ValueError, IndexError):
                                continue
        except Exception as e:
            logger.error(f"提取变电站坐标时出错: {e}")

        logger.info(f"成功提取了 {len(substation_coords)} 个变电站的坐标")
        return substation_coords

    def _build_gas_graph(self):
        """
        构建天然气系统图。
        从 JSON 文件中提取节点和连接信息，并构建图。

        Returns
        -------
        igraph.Graph
            返回构建好的天然气系统图对象。
        """
        logger.info("正在构建天然气系统图")

        # 读取 JSON 数据
        with open(self.gas_file, "r") as f:
            gas_data = json.load(f)

        # 构建 igraph 对象
        g = ig.Graph()

        # 节点映射 (node number -> vertex index)
        node_indices = {}

        # 首先添加所有节点
        for node in gas_data.get("nodes", []):
            node_id = f"g_{node['number']}"
            g.add_vertex(
                name=node_id,  # 添加前缀防止与电力系统节点冲突
                label=node.get("name", ""),
                latitude=node.get("lat", 0.0),
                longitude=node.get("lon", 0.0),
                substation_id=node.get("sub", -1),  # 关联变电站ID
                gas_load_qf=node.get("qf", 0.0),
                gas_load_p=node.get("p", 0.0),
                qf_min=node.get("qf_min", 0.0),
                qf_max=node.get("qf_max", 0.0),
                is_gas_node=True,  # 标记节点类型
                node_type=node.get("name", "Unknown"),  # 节点类型
                is_slack=node.get("slack", False),  # 是否为松弛节点
            )
            node_indices[node["number"]] = g.vcount() - 1  # 记录节点索引

        # 添加天然气管道和其他连接
        for branch in gas_data.get("branches", []):
            from_node = f"g_{branch['n1']}"
            to_node = f"g_{branch['n2']}"

            try:
                # 添加边及其属性
                g.add_edge(
                    from_node,
                    to_node,
                    dev_type=branch.get("dev_type", ""),
                    q=branch.get("q", 0.0),  # 流量
                    length=branch.get("length", 0.0),  # 管道长度
                    diameter=branch.get("diameter", 0.0),  # 管道直径
                    friction_factor=branch.get("friction_factor", 0.0),  # 摩擦系数
                    k=branch.get("k", 0.0),  # k系数
                    uid=branch.get("uid", 0),  # 唯一标识符
                    is_gas_edge=True,  # 标记为天然气边
                )
            except Exception as e:
                logger.warning(f"添加天然气边时出错 ({from_node} -> {to_node}): {e}")

        # 添加一些基本的网络分析指标
        g.vs["degree"] = g.degree()

        # 处理可能的连通分量问题
        components = g.components()
        if len(components) > 1:
            logger.warning(f"天然气网络包含 {len(components)} 个不连通的子图")

        logger.info(f"天然气系统图构建完成，节点数：{g.vcount()}，边数：{g.ecount()}")
        return g

    def _merge_graphs(self, electric_graph, gas_graph):
        """
        合并电力系统图和天然气系统图。

        Parameters
        ----------
        electric_graph : igraph.Graph
            电力系统图。
        gas_graph : igraph.Graph
            天然气系统图。

        Returns
        -------
        igraph.Graph
            返回合并后的图。
        """
        logger.info("正在合并电力系统图和天然气系统图")

        # 创建一个新的空图
        merged_graph = ig.Graph()

        # 复制电力图的所有顶点属性
        for v in electric_graph.vs:
            attrs = v.attributes()
            # 从属性字典中移除 'name' 键，避免参数冲突
            name = attrs.pop("name", str(v.index))
            merged_graph.add_vertex(name=name, **attrs)

        # 复制电力图的所有边
        for e in electric_graph.es:
            source_name = electric_graph.vs[e.source]["name"]
            target_name = electric_graph.vs[e.target]["name"]
            edge_attrs = e.attributes()
            merged_graph.add_edge(source_name, target_name, **edge_attrs)

        # 复制天然气图的所有顶点
        for v in gas_graph.vs:
            attrs = v.attributes()
            # 从属性字典中移除 'name' 键，避免参数冲突
            name = attrs.pop("name", f"g_{v.index}")
            merged_graph.add_vertex(name=name, **attrs)

        # 复制天然气图的所有边
        for e in gas_graph.es:
            source_name = gas_graph.vs[e.source]["name"]
            target_name = gas_graph.vs[e.target]["name"]
            edge_attrs = e.attributes()
            merged_graph.add_edge(source_name, target_name, **edge_attrs)

        # 添加电力-天然气耦合连接
        # 将天然气节点与其关联的变电站连接
        for v in merged_graph.vs.select(is_gas_node_eq=True):
            # 使用安全的方式获取属性
            try:
                sub_id = v["substation_id"]
                # 如果 substation_id 是 None 或不存在，设置为默认值 -1
                if sub_id is None:
                    sub_id = -1
            except (KeyError, AttributeError):
                sub_id = -1

            if sub_id != -1:  # 如果有关联的变电站
                # 查找该变电站ID下的所有电力节点
                electric_nodes = merged_graph.vs.select(
                    is_electric_node_eq=True, substation_id_eq=sub_id
                )

                if electric_nodes:
                    # 与该变电站下的一个电力节点建立连接
                    merged_graph.add_edge(
                        v["name"], electric_nodes[0]["name"], is_coupling_edge=True
                    )
                    logger.debug(f"添加耦合连接: {v['name']} -> {electric_nodes[0]['name']}")

        logger.info(
            f"合并后的图构建完成，节点数：{merged_graph.vcount()}，边数：{merged_graph.ecount()}"
        )
        return merged_graph

    def _extract_basic_features(self, g):
        """
        提取图的基本特征。

        Parameters
        ----------
        g : igraph.Graph
            输入的图结构。

        Returns
        -------
        tuple
            返回节点特征矩阵和边特征矩阵。
        """
        logger.info("正在提取基本特征")

        # 节点特征：经纬度、节点类型编码等
        node_features_list = []

        for v in g.vs:
            features = [v.get("latitude", 0.0), v.get("longitude", 0.0)]

            # 添加节点类型的one-hot编码
            is_electric = int(v.get("is_electric_node", False))
            is_gas = int(v.get("is_gas_node", False))
            features.extend([is_electric, is_gas])

            # 对于电力节点，添加基准电压
            if is_electric:
                features.append(v.get("base_kv", 0.0))
            else:
                features.append(0.0)

            # 对于天然气节点，添加流量和压力特征
            if is_gas:
                features.append(v.get("gas_load_qf", 0.0))
                features.append(v.get("gas_load_p", 0.0))
            else:
                features.extend([0.0, 0.0])

            node_features_list.append(features)

        # 转换为numpy数组
        node_features = np.array(node_features_list, dtype=np.float32)

        # 边特征：边两端节点的经纬度差值、类型编码等
        edge_features_list = []

        for e in g.es:
            source = g.vs[e.source]
            target = g.vs[e.target]

            # 基本空间特征
            source_lat = source.get("latitude", 0.0)
            source_lon = source.get("longitude", 0.0)
            target_lat = target.get("latitude", 0.0)
            target_lon = target.get("longitude", 0.0)

            # 计算经纬度均值和差值
            mean_lat = (source_lat + target_lat) / 2
            mean_lon = (source_lon + target_lon) / 2
            diff_lat = abs(source_lat - target_lat)
            diff_lon = abs(source_lon - target_lon)

            # 欧式距离（简化计算，未考虑地球曲率）
            dist = ((source_lat - target_lat) ** 2 + (source_lon - target_lon) ** 2) ** 0.5

            # 边类型编码
            is_electric_source = int(source.get("is_electric_node", False))
            is_gas_source = int(source.get("is_gas_node", False))
            is_electric_target = int(target.get("is_electric_node", False))
            is_gas_target = int(target.get("is_gas_node", False))

            # 构建边特征向量
            edge_features = [
                mean_lat,
                mean_lon,
                diff_lat,
                diff_lon,
                dist,
                is_electric_source,
                is_gas_source,
                is_electric_target,
                is_gas_target,
            ]

            edge_features_list.append(edge_features)

        # 转换为numpy数组
        edge_features = np.array(edge_features_list, dtype=np.float32)

        logger.info(
            f"节点特征矩阵维度：{node_features.shape}，边特征矩阵维度：{edge_features.shape}"
        )
        return node_features, edge_features

    def _save_processed_data(self, electric_graph, gas_graph, g):
        """
        保存处理结果到指定的输出目录。
        Parameters
        ----------
        g : igraph.Graph
            图对象。
        node_features : numpy.ndarray
            节点特征矩阵。
        edge_features : numpy.ndarray
            边特征矩阵。
        """
        logger.info("正在保存处理结果")
        # 保存图结构
        electric_graph.write_pickle(self.output_dir / "electric_graph.pkl")
        gas_graph.write_pickle(self.output_dir / "gas_graph.pkl")
        g.write_pickle(self.output_dir / "merged_graph.pkl")
        # 保存节点特征和边特征
        # np.save(self.output_dir / "node_features.npy", node_features)
        # np.save(self.output_dir / "edge_features.npy", edge_features)
        logger.success("所有处理结果已保存到指定位置")


app = typer.Typer()


@app.command()
def main(
    electric_file: Annotated[
        Path,
        typer.Argument(help=".m 文件, 仅包含潮流数据的文本文件"),
    ] = RAW_DATA_DIR / "Texas7k_Gas/Texas7k.m",
    gas_file: Annotated[
        Path,
        typer.Argument(help="JSON 文件, 天然气系统数据文件路径"),
    ] = RAW_DATA_DIR / "Texas7k_Gas/Texas7k_Gas.json",
    aux_file: Annotated[
        Path,
        typer.Argument(help=".AUX 文件, 电力系统辅助数据文件路径"),
    ] = RAW_DATA_DIR / "Texas7k_Gas/Texas7k.AUX",
    output_dir: Annotated[Path, typer.Argument(help="处理结果保存目录路径")] = INTERIM_DATA_DIR
    / "Texas7k_Gas",
):
    """
    从原始数据文件生成电网和天然气网络
    的图特征，并将结果保存至指定目录。
    """
    log_file = LOGS_DIR / "texas_features.log"
    logger.add(log_file, mode="w")  # 表示覆盖保存
    logger.add("network_processing.log", mode="w")  # 表示覆盖保存
    logger.info("开始电网和天然气网络数据特征处理")

    # 初始化 NetworkProcessor 处理器
    processor = NetworkProcessor(
        electric_file=electric_file, gas_file=gas_file, aux_file=aux_file, output_dir=output_dir
    )

    # 执行数据处理
    electric_graph, gas_graph, merged_graph = processor.process()

    # 处理完成后记录日志
    logger.success("电网和天然气网络特征处理完成")
    logger.info(f"处理结果已保存至: {output_dir}")


if __name__ == "__main__":
    app()
