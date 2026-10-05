"""数据读取模块：CSV清单 + Tecplot .dat 解析 + 快照矩阵组装"""
from pathlib import Path
import csv
import numpy as np
from .config import GRID_SIZE

def read_case_table(csv_path):
    """读 (V,H) 清单，过滤空行。返回形状 (N,2) 的数组。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            values = [item.strip() for item in row if item.strip()]  # ★过滤空行
            if not values:
                continue
            rows.append((float(values[0]), float(values[1])))
    return np.asarray(rows, dtype=float)

def case_to_dat_path(data_dir, v, h):
    """由 (V,H) 数值匹配 .dat 文件路径（别用字符串硬拼）"""
    folder = data_dir / f"dat_u{v:.3f}"
    if not folder.exists():
        raise FileNotFoundError(f"找不到航速文件夹: {folder}")
    target = round(h, 8)
    for candidate in folder.glob("suboff_h*.dat"):
        try:
            current = float(candidate.stem.split("h", 1)[1])
        except (IndexError, ValueError):
            continue
        if abs(current - target) < 1e-8:        # 数值匹配，容差
            return candidate
    raise FileNotFoundError(f"在 {folder} 下找不到 H={h:g} 的文件")

def read_dat(path):
    """读 Tecplot .dat：跳过9行头，返回 x,y,z 三个展平数组"""
    data = np.loadtxt(path, skiprows=9)        # ★skiprows=9 不是5
    assert data.shape == (GRID_SIZE, 3), f"{path} 形状异常: {data.shape}"
    return data[:, 0], data[:, 1], data[:, 2]

def load_dataset(data_dir, csv_path):
    """组装数据集 → (params, snapshots, grid_x, grid_y, file_paths)"""
    params = read_case_table(csv_path)
    snapshots = np.empty((GRID_SIZE, params.shape[0]), dtype=np.float64)
    paths = []
    grid_x = grid_y = None
    for i, (v, h) in enumerate(params):
        p = case_to_dat_path(data_dir, v, h)
        x, y, z = read_dat(p)
        snapshots[:, i] = z                    # 每列一个算例
        paths.append(p)
        if grid_x is None:                     # 所有算例同一网格，坐标取第一个
            grid_x, grid_y = x, y
    return params, snapshots, grid_x, grid_y, paths
