# RL-AMR

```powershell
wsl -d Ubuntu
```

```bash
conda activate pinn
python -m train
```

Settings are at the top of each file: training, optimizer and paths in `train.py`; network in `model.py`; FEM and sampling in `static_bump.py`.

Terminal: GPU and settings at startup; current-episode error, reward and loss every 100 episodes and at completion.

Outputs: `data/<run>/checkpoint.pt`, `history.csv`, `training.png`.
