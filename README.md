# 沧海遗珠队 · 海路通杯（智能组）水下航行器兴波预测与反演

中山大学第二届"海路通"杯数智水池大赛（智能组）参赛代码。基于 **POD 降阶 + 高斯过程代理模型 + 能量约束反演** 实现两项任务：

- **任务一**：给定航速 V、潜深 H，预测自由面波高场 Z（`predict.py` → `results/predict/task1/T1-*.dat`）
- **任务二**：给定匿名波高场，反演航速 V、潜深 H（`predict.py` → `results/predict/task2_result.csv`）

## 环境

- Python 3.11（开发验证环境 3.11.16），建议 conda 独立环境：
  ```bash
  conda create -n hlt2026 python=3.11
  conda activate hlt2026
  pip install -r requirements.txt
  ```
- 依赖已按 `requirements.txt` 锁版本（numpy / pandas / scikit-learn / scipy / matplotlib / joblib / openpyxl 等）。

## 目录结构

```
├── main.py            # 主流程：数据读取 → POD → GP → Test1 正演验证 → Test2 反演验证
├── predict.py         # 正式预测：任务一 9 个波高场 + 任务二 9 个 V/H 反演
├── requirements.txt   # 锁定版本依赖
├── src/
│   ├── config.py      # 路径与超参数
│   ├── data_io.py     # Tecplot .dat 读取 / 快照矩阵组装
│   ├── svd_model.py   # POD/SVD 降阶与重构
│   ├── gp_model.py    # 高斯过程代理（RBF+WhiteKernel+线性趋势核）
│   └── inversion.py   # 任务二反演（粗网格搜索 + Powell 精修，系数+能量双目标）
└── results/           # 运行产物（图、模型、CSV）
```

## 运行

```bash
# 1) 训练 + 公开验证集评估（Test1 正演误差、Test2 反演误差）
python main.py

# 2) 正式预测（任务一 .dat + 任务二 CSV）
python predict.py
```

### 海光 DCU（可选）

在已安装 DTK 兼容版 PyTorch 的海光 DCU 容器中，可使用 DCU 加速 POD/SVD：

```bash
python main.py --backend dcu
python predict.py --backend dcu
```

普通 CPU 环境无需安装 PyTorch，默认 `--backend auto` 会自动回退到 NumPy/SciPy。GP 代理模型仍使用 scikit-learn，保证 CPU 与 DCU 容器均可复现。

关键参数（均有命令行默认值，可覆盖）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--energy` | 0.999 | POD 累计能量阈值（保留 20 模态 / 99.90%） |
| `--alpha` | 0.5 | 反演目标中系数项权重（1-alpha 为能量项权重） |
| `--h-bias` | 0.12 | 潜深系统偏差校正（能量 GP 浅水端偏差，基于公开验证集统计） |

## 方法概要

1. **数据**：132 个训练工况（12 V × 11 H），Tecplot .dat（300×180 网格，54000 点），`skiprows=9` 读取。
2. **POD/SVD 降维**：能量阈值 0.999 → 保留 20 个模态，累计能量 99.90%，训练集重构误差 2.82%。
3. **GP 代理**：每个 POD 模态一个 GP，输入标准化（V、H 尺度差 10 倍），核函数 `RBF + WhiteKernel + 线性趋势项`（趋势项显著改善 V 外推区反演精度）。
4. **任务一**：GP 预测系数 → POD 重构波场，恢复官方 Tecplot 格式。
5. **任务二反演**：目标场投影到 POD 系数，在 (V,H) 空间搜索使「GP 预测系数 + log10(场能量)」与目标场最接近；粗网格 80×80 定位 + Powell 局部精修；潜深系统偏差校正 0.12。

## 公开验证集精度（Test2 9 工况）

| 指标 | 误差 |
|---|---|
| 航速 V 平均相对误差 | 5.34% |
| 潜深 H 平均相对误差 | 8.70% |

分类口径：训练范围内/轻外推工况（6/9）航速 ≤3.7%、潜深 ≤6.5%；强外推工况（V=9.151，超出训练上限 8.23）航速 ~10%（GP 外推饱和，官方示例同样存在）。

## 数据

训练/测试数据来自赛题发布包，未包含在本仓库中。将赛题发布包的 `data` 目录（含 `Train`、`Test`、`Predict`、`Train.csv`、`Public_Validation.csv`）复制到项目根目录即可运行。
