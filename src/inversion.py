"""任务二反演：同一低维空间内搜索 (V,H)，使 GP 预测系数逼近目标场系数
能量特征约束：log10(场能量) 对 H 强单调，与系数项互补，纠正潜深系统偏差"""
import numpy as np
from scipy.optimize import minimize
from .gp_model import predict_ensemble

def invert_field(z_target, svd, scaler, gp_models,
                 scaler_E=None, gp_E=None,
                 v_bounds=(3.0, 9.5), h_bounds=(0.3, 3.5),
                 n_grid=80, k=8, alpha=0.5, h_bias=0.0):
    """单个目标场 → (V_hat, H_hat)
    ★alpha=0.5：系数项与能量项归一化后等权；scaler_E=None 时退化为纯系数反演
    ★h_bias：潜深系统偏差校正（能量 GP 浅水端预测偏差所致，基于公开验证集统计
      ≈0.12：V 反演准确的 6 工况 H 普遍高估 0.09~0.25）
    ★只用前 k 个主导模态反演：高模态是残差，参与只会引入噪声"""
    zc = z_target[:, None] - svd.mean                              # (54000,1)，防1-D广播成矩阵
    a_target = (zc.T @ svd.modes).ravel()                          # (r,)
    logE_target = np.log10(max(float((z_target ** 2).sum()), 1e-12))

    # 1) 粗网格搜索
    vs = np.linspace(*v_bounds, n_grid)
    hs = np.linspace(*h_bounds, n_grid)
    V, H = np.meshgrid(vs, hs)
    grid = np.column_stack([V.ravel(), H.ravel()])                 # (6400, 2)
    a_grid, _ = predict_ensemble(scaler, gp_models, grid)          # (6400, r)
    c_coef = ((a_grid[:, :k] - a_target[:k]) ** 2).mean(axis=1)

    if scaler_E is not None:
        e_grid, _ = predict_ensemble(scaler_E, gp_E, grid)         # (6400, 1)
        c_ener = (e_grid[:, 0] - logE_target) ** 2
        cost = alpha * c_coef / c_coef.mean() + (1 - alpha) * c_ener / c_ener.mean()
    else:
        cost = c_coef
    x0 = grid[int(np.argmin(cost))]

    # 2) 局部优化精修（Powell 无梯度，稳健；从网格最优±邻域多点起步）
    def loss(x):
        a_pred, _ = predict_ensemble(scaler, gp_models, np.asarray([x]))
        c1 = float(((a_pred[0, :k] - a_target[:k]) ** 2).mean())
        if scaler_E is not None:
            e_pred, _ = predict_ensemble(scaler_E, gp_E, np.asarray([x]))
            c2 = float((e_pred[0, 0] - logE_target) ** 2)
            return alpha * c1 / c_coef.mean() + (1 - alpha) * c2 / c_ener.mean()
        return c1

    best, best_cost = x0.copy(), loss(x0)
    for dx, dh in [(0, 0), (0.3, 0), (-0.3, 0), (0, 0.15), (0, -0.15)]:
        x_start = np.clip(x0 + np.array([dx, dh]),
                          [v_bounds[0], h_bounds[0]], [v_bounds[1], h_bounds[1]])
        res = minimize(loss, x_start, method="Powell",
                       bounds=[v_bounds, h_bounds])   # ★(V 下,上)(H 下,上)，别用 zip 错配
        if res.fun < best_cost:
            best, best_cost = res.x, res.fun
    return float(best[0]), float(best[1]) - h_bias
