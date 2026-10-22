# 训练崩溃排查记录（2026-10-22）

当前结论：尚未稳定复现原来的 `stack smashing detected`，不能宣布根因已找到或问题已修复。已完成的监测不支持“FEM 越积越多导致内存耗尽”的解释。

## 环境与范围

- WSL Ubuntu 24.04，`/home/yjc/miniconda3/envs/pinn/bin/python`，Python 3.12.0。
- PyTorch 2.14.0 / CUDA 13.0，NumPy 2.5.3，MFEM 4.10.0；GPU 为 RTX 4060 Laptop。
- 保持原模型、FEM、batch=64、seed=0、Adam、lr=1e-4。前几轮仅将诊断运行长度设为 512 回合，未改变探索衰减公式。
- 未修改 `train.py`、`static_bump.py`、`model.py`，未安装或升级环境依赖。诊断源代码快照在各运行目录。

## 证据

| 测试 | 结果 | 证据路径（仓库相对路径） |
|---|---|---|
| 原失败运行 | history 到 409 回合；无 checkpoint | `data/20261022_000928_060575/` |
| 第一次带监测训练 | 第 97 回合第 1 步，观测采样期间出现 Python ValueError；并非原来的栈保护 abort | `data/diagnostics/baseline_512.log` |
| 第二次相同训练 | 完成 512 回合；前 409 回合 final_error 和 policy_loss 与原运行逐行一致 | `data/diagnostics/baseline_repeat/` |
| Python 调试内存分配器 + 原生栈捕获 | 完成 512 回合，生成 checkpoint/history/图 | `data/diagnostics/native_debug/` |
| 单独重放第 97 回合首步 | 连续 100 次通过 | `data/diagnostics/replay_97.log` |
| 单独重放前 97 回合记录的全部动作 | 通过 | `data/diagnostics/replay_1_97.log` |
| 固定网格、固定点定位结果，重复标量采样 | 40,000,000 次通过 | `data/diagnostics/fixed_getvalue.log` |
| 固定网格，反复重建完整观测（含 FindPoints 和临时对象释放） | 3000 次通过 | `data/diagnostics/repeated_state.log` |
| 原始 `python -m train` 入口，仅附加错误栈捕获 | 运行中；设置 20 分钟诊断上限 | `data/diagnostics/original_entry.log` |

第二次训练每 5 秒采样的最大 RSS 为 1747.78 MiB；PyTorch CUDA allocator 的 allocated 最大 391.65 MiB，reserved 最大 400 MiB。预热后 RSS 基本稳定。这些是采样值，不是系统全程连续峰值；CUDA allocator 数值也不包含全部驱动和桌面显存。首次诊断异常前 RSS 为 1,772,496 kB、CUDA allocated 197,004,800 bytes。

## 尚未解释的异常

第一次诊断在 `static_bump.py` 的 `np.fromiter(... for i, e in enumerate(elements))` 期间报告 `ValueError: too many values to unpack (expected 2)`。其输入动作重放成功，后续完整训练也越过相同位置。它提供了一个需要重点观察的采样位置，但没有证据证明该 Python 异常就是原栈保护失败的根因，也不能仅凭栈位置将责任归于 MFEM。

原 WSL `core_pattern` 指向 `/wsl-capture-crash`，但该程序不存在，标准 core 目录中也未找到旧转储。因此原终端的 `core dumped` 字样没有提供可读取的旧 core 文件。

## 可复用的诊断入口

在仓库根目录、WSL pinn 环境运行：

```bash
python -X faulthandler -u diagnose_crash.py --output data/diagnostics/new_run --episodes 512
```

输出包括源代码快照、环境版本、每一步动作、FEM 阶段日志、5 秒内存采样，以及训练正常完成时的模型和曲线。输出目录必须尚不存在。

`native_crash_trace.c` 是仅供诊断的本地预加载库：栈保护失败时先输出原生 backtrace，再调用 abort；它不修复内存问题，也未写入正式训练入口。编译结果为 `data/diagnostics/native_crash_trace.so`。`probe_sampling.py` 用于动作重放，`probe_getvalue.py` 用于采样隔离测试。

上述 512 回合 checkpoint 均是诊断产物，不代表原计划的 20,000 回合训练已完成。
