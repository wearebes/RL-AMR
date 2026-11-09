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

```bash
python -m evaluation.plot data/advection_IPN/history.csv
python -m evaluation.evaluate bumps --config /path/to/evaluation.yaml
```
