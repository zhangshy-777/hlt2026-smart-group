"""正式预测脚本（可复现）：
任务一：T1-01~09 (V,H) → 预测自由面波高 Z → results/predict/task1/T1-XX.dat（Tecplot，X Y Z 三列）
任务二：Predict/Part2/T2-01~09.dat → 反演 (V,H) → results/predict/task2_result.csv
与 main.py 同配置：energy=0.999、趋势核 GP、能量约束反演 alpha=0.5 h_bias=0.12
运行：python predict.py
"""
import argparse
import numpy as np
from pathlib import Path
from src.config import (DATA_DIR, TRAIN_DATA_DIR, TRAIN_CSV, RESULTS_DIR,
                        GRID_SIZE, ensure_result_dirs)
from src.svd_model import fit_svd, training_coefficients, reconstruct_from_coefficients
from src.gp_model import train_ensemble, predict_ensemble
from src.data_io import load_dataset, read_dat
from src.inversion import invert_field

# 赛题「需预测工况.xlsx」：T1-01 ~ T1-09（H=3.20 为外推工况，超出训练上限 3.0）
TASK1 = [
    (3.748, 1.70), (3.748, 2.80), (3.748, 3.20),
    (6.025, 1.70), (6.025, 2.80), (6.025, 3.20),
    (7.695, 1.70), (7.695, 2.80), (7.695, 3.20),
]

def write_dat(path, x, y, z, title, solution_time=0):
    """按组委会统一 Tecplot 格式写波高场（300×180，X Y Z 三列）"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'TITLE     = "{title}"\n')
        f.write('VARIABLES = "X"\n"Y"\n"Z"\n')
        f.write('ZONE T="Rectangular zone"\n')
        f.write(f' STRANDID=0, SOLUTIONTIME={solution_time}\n')
        f.write(' I=300, J=180, K=1, ZONETYPE=Ordered\n')
        f.write(' DATAPACKING=POINT\n')
        f.write(' DT=(SINGLE SINGLE SINGLE )\n')
        for i in range(GRID_SIZE):
            f.write(f'{x[i]:14.9E} {y[i]:14.9E} {z[i]:14.9E}\n')

def main():
    ap = argparse.ArgumentParser(description="海路通智能组 正式预测（任务一+任务二）")
    ap.add_argument("--energy", type=float, default=0.999, help="POD 能量阈值（答辩口径）")
    ap.add_argument("--alpha", type=float, default=0.5, help="反演系数/能量权重")
    ap.add_argument("--h-bias", type=float, default=0.12, help="潜深系统偏差校正")
    args = ap.parse_args()

    paths = ensure_result_dirs()
    pred_dir = RESULTS_DIR / "predict"
    (pred_dir / "task1").mkdir(parents=True, exist_ok=True)

    # ===== 训练（与 main.py 完全一致，保证可复现）=====
    print("加载训练集...")
    x_train, z_train, grid_x, grid_y, _ = load_dataset(TRAIN_DATA_DIR, TRAIN_CSV)
    print(f"  训练工况 {x_train.shape[0]} 个，快照矩阵 {z_train.shape}")

    svd = fit_svd(z_train, args.energy)
    a_train = training_coefficients(svd)
    print(f"  POD 保留模态 r={svd.retained_rank}，累计能量={svd.cumulative_energy[-1]*100:.4f}%")

    print("训练趋势核 GP（POD 系数 + 能量）...")
    scaler, gp_models = train_ensemble(x_train, a_train)                     # 系数 GP（趋势核）
    E_log = np.log10((z_train ** 2).sum(axis=0) + 1e-12)[:, None]
    scaler_E, gp_E = train_ensemble(x_train, E_log)                          # 能量 GP（趋势核）

    # ===== 任务一：T1-01~09 波高场预测 =====
    print("任务一：预测 9 个工况的自由面波高 Z ...")
    x_pred = np.asarray(TASK1, dtype=float)
    a_pred, _ = predict_ensemble(scaler, gp_models, x_pred)
    z_pred = reconstruct_from_coefficients(a_pred, svd)
    for i, (v, h) in enumerate(TASK1, 1):
        out = pred_dir / "task1" / f"T1-{i:02d}.dat"
        write_dat(out, grid_x, grid_y, z_pred[:, i - 1], f"T1-{i:02d}")
        print(f"  T1-{i:02d}  V={v:.3f} H={h:.2f} → {out.name}  "
              f"(Z∈[{z_pred[:, i-1].min():.3e}, {z_pred[:, i-1].max():.3e}])")

    # ===== 任务二：T2-01~09 反演 V,H =====
    print("任务二：反演 9 个匿名波高场的航速 V 与潜深 H ...")
    t2_dir = DATA_DIR / "Predict" / "Part2"
    results = []
    for f in sorted(t2_dir.glob("T2-*.dat")):
        _, _, z_t = read_dat(f)
        v_h, h_h = invert_field(z_t, svd, scaler, gp_models, scaler_E, gp_E,
                                alpha=args.alpha, h_bias=args.h_bias)
        results.append((f.stem, f.name, v_h, h_h))
        print(f"  {f.stem}: V_pred={v_h:.3f}  H_pred={h_h:.3f}")

    csv_path = pred_dir / "task2_result.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("匿名编号,V_pred,H_pred\n")
        for name, fname, v, h in results:
            f.write(f"{name},{v:.3f},{h:.3f}\n")
    print("已保存:", csv_path)

    print("\n完成。交付物：")
    print("  任务一:", pred_dir / "task1" / "T1-01.dat  ~  T1-09.dat")
    print("  任务二:", csv_path)

if __name__ == "__main__":
    main()
