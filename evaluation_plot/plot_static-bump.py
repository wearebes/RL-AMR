"""Test a static-trained IPN on advection bumps at budgets 10 and 100.

Run in the MFEM environment: python evaluation_plot/plot_static-bump.py
Samples stay in memory. By default the figure is shown without writing files;
--output optionally saves only the figure. Error bars are SEM across paired
initial conditions for one checkpoint, not across independent training runs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Each process owns its solver and policy; no MFEM objects cross process boundaries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from advection_bump import AdvectionBump
from evaluation_plot.plot import collect, indicator_action, load_policy


BUDGETS = (10, 100)
METHODS = ("IPN", "Random", "ZZ", "TrueError")
COLORS = ("#554fb0", "#777777", "#d78056", "#73c0d2")


def initialize_worker(checkpoint):
    global POLICY
    POLICY, _ = load_policy(checkpoint)


def evaluate(seed):
    """Paired methods and budgets share one IC; budget 10 is the 100-step prefix."""
    rng = np.random.default_rng(seed)
    final_errors = np.empty((len(BUDGETS), len(METHODS)))
    no_refine = np.empty(len(BUDGETS))
    for method in ("NoRefine",) + METHODS:
        source = AdvectionBump(observe=method == "IPN")
        source.settings["budget"] = max(BUDGETS)
        state = source.reset(seed)
        initial_error = state["error"]
        for step in range(1, max(BUDGETS) + 1):
            if method == "NoRefine":
                action = 0
            elif method == "IPN":
                with torch.inference_mode():
                    logits = POLICY(torch.from_numpy(state["observation"]),
                                    torch.from_numpy(state["valid_actions"]))
                    action = int(logits.argmax())
            elif method == "Random":
                action = int(rng.choice(np.flatnonzero(state["valid_actions"])))
            else:
                action = indicator_action(source, method)
            state, _ = source.step(action)
            if step in BUDGETS:
                index = BUDGETS.index(step)
                if method == "NoRefine":
                    no_refine[index] = state["error"]
                else:
                    final_errors[index, METHODS.index(method)] = state["error"]
    # Paper section 5.2: compare against no refinement at the SAME final time.
    reductions = (no_refine[:, None] - final_errors) / initial_error
    return reductions


def draw(results):
    import matplotlib.pyplot as plt

    means = results.mean(axis=0)
    sem = results.std(axis=0, ddof=1) / np.sqrt(len(results))
    with plt.rc_context({"font.size": 12}):
        figure, axis = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
        width = 0.18
        positions = np.arange(len(BUDGETS))
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            axis.bar(positions + (i - 1.5) * width, means[:, i],
                     width=width, color=color, label=method, yerr=sem[:, i],
                     error_kw={"elinewidth": 1, "capsize": 2})
        axis.set_xticks(positions, [str(budget) for budget in BUDGETS])
        axis.set_xlabel("Budget")
        axis.set_ylabel("Normalized error reduction")
        axis.axhline(0, color="0.4", linewidth=0.7)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                    ncol=2, frameon=False, columnspacing=1.4, handlelength=1.5)
    return figure, means, sem

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "data/static_bump/checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4,
                        help="independent test processes; 0 runs serially")
    parser.add_argument("--output", type=Path, help="optionally save only the figure")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    _, saved = load_policy(args.checkpoint)
    train_start, train_end = saved["seed"], saved["seed"] + saved["episodes"]
    if args.seed < train_end and args.seed + args.episodes > train_start:
        parser.error("test seeds overlap the checkpoint's training seeds")
    print(f"Static -> advection bumps | trained episodes={saved['episodes']} | "
          f"test episodes={args.episodes} | budgets={BUDGETS} | "
          f"seeds={args.seed}:{args.seed + args.episodes} | IPN=argmax", flush=True)
    if args.workers == 0:
        initialize_worker(args.checkpoint)
        results = collect(map(evaluate, range(args.seed, args.seed + args.episodes)), args.episodes)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn"),
                                 initializer=initialize_worker,
                                 initargs=(args.checkpoint,)) as pool:
            rows = pool.map(evaluate, range(args.seed, args.seed + args.episodes))
            results = collect(rows, args.episodes)
    figure, means, sem = draw(results)
    for index, budget in enumerate(BUDGETS):
        print(f"Budget {budget}")
        for method, mean, error in zip(METHODS, means[index], sem[index]):
            print(f"  {method:10s} {mean:.6f} +/- {error:.6f} (SEM)")
    if args.output is not None:
        figure.savefig(args.output, dpi=200)
        print(f"Figure: {args.output}", flush=True)
    import matplotlib.pyplot as plt
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
