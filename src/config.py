from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


class ClimateParams:
    flood_max_lat = 55  # 洪水风险最大纬度
    flood_scale = 10  # 洪水风险衰减系数
    heat_min_lat = 40  # 热浪风险最小纬度
    heat_scale = 15  # 热浪风险增长系数
    drought_offset = 10  # 干旱经度偏移量
    drought_scale = 20  # 干旱风险缩放系数


class DynamicParams:
    k = 0.1  # 失效速率
    r = 0.05  # 恢复速率
    sigma = 50  # 地理距离衰减参数 (km)
    w1 = 0.7  # 一阶邻居权重
    w2 = 0.3  # 二阶邻居权重


class GNNParams:
    input_dim = 5
    hidden_dim = 16
    output_dim = 1
    lr = 0.01
    epochs = 100
