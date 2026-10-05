import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                                  # 不弹窗，直接存图
import matplotlib.pyplot as plt
from src.config import DATA_DIR, TRAIN_DATA_DIR, TRAIN_CSV, GRID_I, GRID_J, ensure_result_dirs
from src.svd_model import fit_svd, training_coefficients, project_to_coefficients, reconstruct_from_coefficients
from src.data_io import load_dataset, read_dat

def parse_args():
    p = argparse.ArgumentParser(description="海路通智能组 主程序")
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="数据目录")
    p.add_argument("--energy", type=float, default=0.999, help="POD能量阈值（与答辩口径一致：20模态/99.90%）")
    p.add_argument("--model", choices=["gp", "svr", "mlp"], default="gp", help="代理模型")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()

def main():
    args = parse_args()
    paths = ensure_result_dirs()

    x_train, z_train, grid_x, grid_y, files = load_dataset(TRAIN_DATA_DIR, TRAIN_CSV)

    print("参数表:", x_train.shape, " 快照矩阵:", z_train.shape, " 文件数:", len(files))
    print("Z min={:.4e} max={:.4e} mean={:.4e}".format(z_train.min(), z_train.max(), z_train.mean()))

    # 画第一个算例，验证读对了（★reshape(180, 300)，别反）
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(z_train[:, 0].reshape(GRID_J, GRID_I),
                   origin="lower", aspect="auto", cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label="Z")
    ax.set_title(f"case1  V={x_train[0, 0]:.3f}  H={x_train[0, 1]:.2f}")
    ax.set_xlabel("X (300 pts)")
    ax.set_ylabel("Y (180 pts)")
    fig.savefig(paths["figures"] / "field_check.png", dpi=150, bbox_inches="tight")
    print("已保存:", paths["figures"] / "field_check.png")

    # ===== 第二步：POD/SVD 降维 =====
    svd = fit_svd(z_train, args.energy)
    a_train = training_coefficients(svd)
    print("保留模态数 r =", svd.retained_rank,
          " 累计能量 = {:.6f}".format(svd.cumulative_energy[-1]))

    z_recon = reconstruct_from_coefficients(a_train, svd)
    e_recon = np.linalg.norm(z_train - z_recon) / np.linalg.norm(z_train) * 100
    print("训练集整体重构误差 E_recon = {:.4f}%".format(e_recon))

    # 能量谱图
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, svd.retained_rank + 1), svd.singular_values, "o-")
    ax.set_xlabel("mode index")
    ax.set_ylabel("singular value (log)")
    ax.set_title("SVD spectrum  r={}".format(svd.retained_rank))
    fig.savefig(paths["figures"] / "svd_spectrum.png", dpi=150, bbox_inches="tight")

    # 重构对比图：第一个算例 true vs recon
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(z_train[:, 0].reshape(GRID_J, GRID_I), origin="lower", aspect="auto", cmap="RdBu_r")
    im2 = ax2.imshow(z_recon[:, 0].reshape(GRID_J, GRID_I), origin="lower", aspect="auto", cmap="RdBu_r")
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    ax1.set_title("Z_true  case1")
    ax2.set_title("Z_recon  case1")
    fig.savefig(paths["figures"] / "recon_check.png", dpi=150, bbox_inches="tight")

    # ===== 第三步：GP 代理模型（V,H → POD 系数 → 波场）=====
    from src.gp_model import train_ensemble, predict_ensemble, save_ensemble

    # 3.1 训练 20 个 GP（每模态一个），并保存
    scaler, gp_models = train_ensemble(x_train, a_train)
    save_ensemble(scaler, gp_models, paths["models"] / "gp_ensemble.joblib")
    print("GP 已保存:", paths["models"] / "gp_ensemble.joblib")

    # 3.2 训练集自检（GP 对训练点误差应很小）
    a_pred_train, _ = predict_ensemble(scaler, gp_models, x_train)
    z_pred_train = reconstruct_from_coefficients(a_pred_train, svd)
    e_gp_train = np.linalg.norm(z_train - z_pred_train) / np.linalg.norm(z_train) * 100
    print("训练集 GP 场误差 = {:.4f}%".format(e_gp_train))

    # 3.3 Test1 泛化验证：遍历目录解析 (V,H)，读真场
    test1_dir = DATA_DIR / "Test" / "Test1"
    vh_test, z_test, names = [], [], []
    for folder in sorted(test1_dir.glob("dat_u*")):
        v = float(folder.name[len("dat_u"):])              # "dat_u3.748" → 3.748
        for f in sorted(folder.glob("suboff_h*.dat")):
            h = float(f.stem.split("h", 1)[1])
            _, _, z = read_dat(f)
            vh_test.append([v, h]); z_test.append(z); names.append(f.stem)
    vh_test = np.asarray(vh_test, dtype=float)
    z_test = np.asarray(z_test, dtype=float).T          # (54000, 9)

    a_test_true = project_to_coefficients(z_test, svd)  # 真场投影 → 真实系数
    a_test_pred, _ = predict_ensemble(scaler, gp_models, vh_test)   # GP 预测系数
    z_test_pred = reconstruct_from_coefficients(a_test_pred, svd)   # 系数 → 波场

    errs = np.linalg.norm(z_test - z_test_pred, axis=0) / np.linalg.norm(z_test, axis=0) * 100
    print("Test1 场误差: 均值 {:.2f}%  最大 {:.2f}%".format(errs.mean(), errs.max()))
    for nm, (v, h), e in zip(names, vh_test, errs):
        print("  {:<16s} V={:.3f} H={:.2f}  err={:.2f}%".format(nm, v, h, e))

    # 3.4 画图：挑 V≈6.025 H≈0.80 的工况，真实 | GP预测 | 误差 三联
    idx = int(np.argmin(np.abs(vh_test[:, 0] - 6.025) + np.abs(vh_test[:, 1] - 0.80)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(z_test[:, idx].reshape(GRID_J, GRID_I),
                         origin="lower", aspect="auto", cmap="RdBu_r")
    im1 = axes[1].imshow(z_test_pred[:, idx].reshape(GRID_J, GRID_I),
                         origin="lower", aspect="auto", cmap="RdBu_r")
    im2 = axes[2].imshow(np.abs(z_test[:, idx] - z_test_pred[:, idx]).reshape(GRID_J, GRID_I),
                         origin="lower", aspect="auto", cmap="hot")
    axes[0].set_title("Z_true  V={:.3f} H={:.2f}".format(vh_test[idx, 0], vh_test[idx, 1]))
    axes[1].set_title("Z_GP")
    axes[2].set_title("|err|  {:.2f}%".format(errs[idx]))
    fig.colorbar(im0, ax=axes[0]); fig.colorbar(im1, ax=axes[1]); fig.colorbar(im2, ax=axes[2])
    fig.savefig(paths["figures"] / "gp_check.png", dpi=150, bbox_inches="tight")
    print("已保存:", paths["figures"] / "gp_check.png")

    # 能量特征 GP：log10(场能量)，对 H 强单调，用于反演约束
    E_train_log = np.log10((z_train ** 2).sum(axis=0) + 1e-12)[:, None]   # (132,1)
    scaler_E, gp_E = train_ensemble(x_train, E_train_log)

    # ===== 第四步：任务二反演（Z → V,H），Test2 验证 =====
    from src.inversion import invert_field

    test2_dir = DATA_DIR / "Test" / "Test2"
    vh_true, vh_pred = [], []
    for folder in sorted(test2_dir.glob("dat_u*")):
        v_t = float(folder.name[len("dat_u"):])                # "dat_u3.660" → 3.66
        for f in sorted(folder.glob("suboff_h*.dat")):
            h_t = float(f.stem.split("h", 1)[1])               # "suboff_h0.75" → 0.75
            _, _, z_t = read_dat(f)
            v_h, h_h = invert_field(z_t, svd, scaler, gp_models, scaler_E, gp_E,
                                    alpha=0.5, h_bias=0.12)
            vh_true.append([v_t, h_t]); vh_pred.append([v_h, h_h])
    vh_true = np.asarray(vh_true); vh_pred = np.asarray(vh_pred)
    rel_v = np.abs(vh_pred[:, 0] - vh_true[:, 0]) / vh_true[:, 0] * 100
    rel_h = np.abs(vh_pred[:, 1] - vh_true[:, 1]) / vh_true[:, 1] * 100
    print("Test2 反演: 航速平均相对误差 {:.2f}%  潜深平均相对误差 {:.2f}%"
          .format(rel_v.mean(), rel_h.mean()))
    for (v_t, h_t), (v_h, h_h), rv, rh in zip(vh_true, vh_pred, rel_v, rel_h):
        print("  真实 V={:.3f} H={:.2f}  反演 V={:.3f} H={:.3f}  "
              "航速err={:.2f}% 潜深err={:.2f}%".format(v_t, h_t, v_h, h_h, rv, rh))


if __name__ == "__main__":
    main()
