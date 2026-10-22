# RL-AMR

```bash
conda activate pinn
python -m train --case bumps --num-workers 8
python -m train --case advection --num-workers 8
```

```bash
python -m evaluation_plot.plot --case bumps --checkpoint data/static_bump/checkpoint.pt --orders 2 3 5 --episodes 100 --workers 16 --no-show
python -m evaluation_plot.plot --case advection --orders 2 4 --episodes 100 --workers 16 --no-show
python evaluation_plot/plot_static-bump.py --checkpoint data/static_bump/checkpoint.pt
```
