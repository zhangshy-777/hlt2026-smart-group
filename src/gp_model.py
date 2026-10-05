"""GP 代理模型：V,H → POD 系数（每模态一个 GP）"""
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, RBF, WhiteKernel,
                                              DotProduct)
from sklearn.preprocessing import StandardScaler

def train_ensemble(x_train, a_train, seed=42, use_trend=True):
    """x_train:(132,2) [V,H]，a_train:(132,r) POD系数 → 返回 (scaler, models)
    ★先 StandardScaler 标准化：V(3~8) 和 H(0.5~3) 尺度差 10 倍，不标准化 RBF 会偏向 V
    ★use_trend=True 加 ConstantKernel*DotProduct 线性趋势项：V>8.23 外推区沿趋势延伸
      而不是回归均值（实验：Test2 反演航速 8.73%→5.34%、潜深 27.07%→17.06%）"""
    scaler = StandardScaler()
    xs = scaler.fit_transform(x_train)
    r = a_train.shape[1]
    models = []
    for j in range(r):
        kernel = (ConstantKernel(1.0)
                  * RBF(length_scale=[1.0, 1.0],
                        length_scale_bounds=(1e-2, 1e2))    # ★限制搜索范围，防撞 1e-5/1e5 边界陷阱
                  + WhiteKernel(noise_level=1e-3,
                                noise_level_bounds=(1e-6, 1e-1)))  # ★限制噪声上限，别把信号当噪声
        if use_trend:
            kernel = kernel + ConstantKernel(1.0) * DotProduct(sigma_0=1.0)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=5, random_state=seed)      # ★更多起点，摆脱局部最优
        gp.fit(xs, a_train[:, j])
        models.append(gp)
    return scaler, models

def predict_ensemble(scaler, models, x_new):
    """x_new:(M,2) → (a_pred:(M,r), a_std:(M,r)) 每个模态带不确定度"""
    xs = scaler.transform(x_new)
    a_pred = np.column_stack([m.predict(xs) for m in models])
    a_std  = np.column_stack([m.predict(xs, return_std=True)[1] for m in models])
    return a_pred, a_std

def save_ensemble(scaler, models, path):
    joblib.dump({"scaler": scaler, "models": models}, path)
