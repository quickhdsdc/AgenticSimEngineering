"""仓库内的标准路径（基于 __file__ 计算，与当前工作目录无关）。

重组项目结构后，脚本不该再依赖"从哪个目录运行"。所有产物统一落到
仓库根的 data/ 与 results/ 下，避免同名文件散落在各个子目录里。

    <repo>/                      REPO_ROOT
    ├── data/mbs/                DATA_DIR    仿真生成的数据集
    └── results/mbs/             RESULTS_DIR 验证与评测产物
"""
from pathlib import Path

# mbs_isaac/ -> projects/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "mbs"
RESULTS_DIR = REPO_ROOT / "results" / "mbs"


def data_path(name: str) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR / name)


def result_path(name: str) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / name)
