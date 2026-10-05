from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # 项目根（别写死绝对路径）
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

GRID_I, GRID_J = 300, 180          # 网格
GRID_SIZE = GRID_I * GRID_J        # 54000

ENERGY_THRESHOLD = 0.99            # POD能量阈值（可调）
RANDOM_STATE = 42                  # 随机种子（可复现）
TRAIN_CSV = DATA_DIR / "Train.csv"
PUBLIC_VALIDATION_CSV = DATA_DIR / "Public_Validation.csv"
TRAIN_DATA_DIR = DATA_DIR / "Train"          # 训练数据实际在 data\Train\dat_u* 下

def ensure_result_dirs():
    paths = {
        "arrays":  RESULTS_DIR / "arrays",
        "models":  RESULTS_DIR / "models",
        "figures": RESULTS_DIR / "figures",
        "reports": RESULTS_DIR / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
