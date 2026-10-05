"""POD/SVD 降维模块：中心化 → SVD → 截断 → 投影/重构"""
from dataclasses import dataclass
import numpy as np

@dataclass
class SVDModel:
    mean: np.ndarray              # (54000, 1)
    modes: np.ndarray             # (54000, r)
    singular_values: np.ndarray   # (r,)
    retained_rank: int            # r
    cumulative_energy: np.ndarray # (r,)
    coefficients: np.ndarray      # (132, r) 训练集系数，fit 时算好

def fit_svd(z_train, threshold=0.99):
    """输入 (54000, N) 快照矩阵，输出截断后的 SVDModel"""
    mean = z_train.mean(axis=1, keepdims=True)
    zc = z_train - mean                                  # ★中心化（POD 用波动量）
    U, s, Vt = np.linalg.svd(zc, full_matrices=False)    # U:(54000,132) s:(132,) Vt:(132,132)
    energy = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(energy)
    r = int(np.searchsorted(cum, threshold)) + 1         # ★第一个累计能量≥0.99 的阶数
    coeffs = (s[:r, None] * Vt[:r, :]).T                 # (132, r) 训练集系数
    return SVDModel(
        mean=mean,
        modes=U[:, :r],
        singular_values=s[:r],
        retained_rank=r,
        cumulative_energy=cum[:r],
        coefficients=coeffs,
    )

def training_coefficients(model):
    """返回训练集 POD 系数 (N, r)"""
    return model.coefficients

def project_to_coefficients(z, model):
    """新场投影：z (54000, M) → 系数 (M, r)"""
    return ((z - model.mean).T @ model.modes)

def reconstruct_from_coefficients(coeffs, model):
    """系数 (M, r) → 重构场 (54000, M)"""
    return model.mean + model.modes @ coeffs.T
