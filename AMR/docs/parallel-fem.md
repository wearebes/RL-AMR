# FEM 并行训练

```bash
conda activate pinn
python -m train --num-workers 4
```

默认 16 个独立 FEM worker，可用 `--num-workers 8` 指定数量。`1` 表示单个独立 FEM 进程，`0` 表示主进程内串行执行。

每个 worker 同时持有一个网格，使用 spawn 启动；子进程的 OpenMP、OpenBLAS、MKL 线程数设为 1。不同 initial mesh 的轨迹分波并行推进，主进程统一进行 GPU 策略推理。网格、有限元空间和解向量留在各自进程中，只交换 NumPy 观测、动作和误差。

例如 batch=64、worker=4：每波推进 4 条轨迹，共 16 波；策略在整个 batch 内保持不变，累计满 64 回合后进行一次 Adam 更新，再开始下一批。最后不足一个 batch 时按实际回合数更新。每条轨迹仍依赖自己的前一步动作，提前结束的轨迹等待本波其余轨迹完成。

训练选择和循环见 `train.py`，FEM 设置见两个 case 文件，网络及学习规则见 `model.py`。worker 数独立于训练 batch；每回合的 seed 为 `seed + episode`。改变并行数会改变策略随机采样的调用顺序，因此不保证动作轨迹与串行运行完全相同。

每次优化器更新后原子替换 `checkpoint.pt`：`episodes` 表示已纳入模型更新的回合数，`target_episodes` 表示目标回合数，`num_workers` 表示实际并行数。中途失败时，history 可能包含尚未更新的下一批回合。最终曲线在训练完成时生成。

worker 异常或退出会停止训练并报错，不跳过失败回合；当前未实现自动重启或 checkpoint 自动恢复。

临时测试脚本、诊断记录和测试输出统一放在 `tem/`，不混入正式模块与 `data/`。
