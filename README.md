# RL-AMR

```bash
conda activate pinn
python -m train.train --config train/training.yaml

python -m evaluation.evaluate bumps
python -m evaluation.plot data/static_bump/degree.npz

python -m evaluation.evaluate advection
python -m evaluation.plot data/advection_IPN/degree.npz

python -m evaluation.evaluate transfer
python -m evaluation.plot data/static_bump/transfer.npz
```
```text
RL-AMR/
├── model/
│   └── ipn.py
├── train/
│   ├── train.py
│   ├── learning.py
│   └── training.yaml
├── evaluation/
│   ├── evaluate.py
│   ├── plot.py
│   └── evaluation.yaml
├── FEM/
│   ├── static_bump.py
│   ├── advection_bump.py
│   ├── common.py
│   └── workers.py
└── data/
```

训练参数在 `train/training.yaml`，评估参数在 `evaluation/evaluation.yaml`。
FEM 数值参数由环境的 `settings` 读取；模块顶部常量提供默认值，修改 settings 后需 reset。
观测采样尺寸固定在 `FEM/common.py`，模型共用该尺寸。

评估 NPZ 同时保存训练环境和各测试阶数/预算对应的实际测试环境。
绘图只读取已保存结果；训练曲线也由 `evaluation/plot.py` 绘制：

```bash
python -m evaluation.plot data/advection_IPN/history.csv
# 使用另一份评估配置
python -m evaluation.evaluate bumps --config /path/to/evaluation.yaml
```
