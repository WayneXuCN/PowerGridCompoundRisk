from pathlib import Path

# 项目根目录
PROJECT_DIR = Path(__file__).parent.parent

# 数据路径配置
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RESULT_DIR = DATA_DIR / "results"

# 确保目录存在
for dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, RESULT_DIR]:
    dir.mkdir(parents=True, exist_ok=True)