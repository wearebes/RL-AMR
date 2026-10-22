"""Test static or advection bumps across polynomial degrees.

Run in the MFEM environment: python -m evaluation_plot.plot
The only output file is the figure. Error bars are SEM across paired test
episodes for one trained IPN, not uncertainty across independent training runs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path

# Each test process owns its MFEM objects and uses one numerical-library thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mfem.ser as mfem
import numpy as np
import torch

from model import IPN
from static_bump import StaticBump
from advection_bump import AdvectionBump


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("IPN", "Random", "ZZ", "TrueError")
COLORS = ("#554fb0", "#777777", "#d78056", "#73c0d2")
ORDERS = (2, 3, 5)


def load_policy(checkpoint):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["source"] not in ("static_bump", "advection_bump"):
        raise ValueError("Expected a static_bump or advection_bump checkpoint")
    policy = IPN().eval()
    policy.load_state_dict(saved["model"])
    return policy, saved


def indicator_action(source, method):
    """Mark the largest local indicator among refinable elements.

    TrueError uses local L2 errors. ZZ averages element gradient samples at
    shared nodal H1 DOFs (unit diffusion); it performs no patch least-squares fit.
    """
    valid = source.state["valid_actions"][1:]
    if not valid.any():
        return 0
    if method == "TrueError":
        scores = source.element_errors()
    else:  # ZZ: recover the gradient in a continuous H1 space.
        # ComputeElementFlux writes nodal samples, so recovery must use a nodal basis.
        recovery = mfem.H1_FECollection(source.settings["order"], 2, mfem.BasisType.GaussLobatto)
        flux_space = mfem.FiniteElementSpace(source.mesh, recovery, 2)
        integrator = mfem.DiffusionIntegrator()
        estimator = mfem.ZienkiewiczZhuEstimator(integrator, source.solution, flux_space)
        estimator.SetFluxAveraging(0)  # Arithmetic averaging, including attribute interfaces.
        scores = estimator.GetLocalErrors().GetDataArray().copy()
    return int(np.argmax(np.where(valid, scores, -np.inf))) + 1


def initialize_worker(checkpoint, orders=ORDERS, case="bumps"):
    global POLICY, TEST_ORDERS, TEST_CASE
    POLICY, _ = load_policy(checkpoint)
    TEST_ORDERS = orders
    TEST_CASE = case


def evaluate(seed):
    """Every degree and method sees the same initial function for this seed."""
    source = AdvectionBump() if TEST_CASE == "advection" else StaticBump()
    reductions = []
    for order in TEST_ORDERS:
        source.settings["order"] = order
        if TEST_CASE == "advection":
            baseline = AdvectionBump(observe=False)
            baseline.settings["order"] = order
            state = baseline.reset(seed)
            done = False
            while not done:
                state, done = baseline.step(0)
            reference = state["error"]
        rng = np.random.default_rng(seed)
        row = []
        for method in METHODS:
            state = source.reset(seed)
            initial = state["error"]
            done = False
            while not done:
                if method == "IPN":
                    with torch.inference_mode():
                        logits = POLICY(torch.from_numpy(state["observation"]),
                                        torch.from_numpy(state["valid_actions"]))
                        action = int(logits.argmax())
                elif method == "Random":
                    # The no-op is part of the same valid action set used by IPN.
                    action = int(rng.choice(np.flatnonzero(state["valid_actions"])))
                else:
                    action = indicator_action(source, method)
                state, done = source.step(action)
            reference_error = reference if TEST_CASE == "advection" else initial
            row.append((reference_error - state["error"]) / initial)
        reductions.append(row)
    return reductions


def draw(results, orders=ORDERS):
    import matplotlib.pyplot as plt

    means = results.mean(axis=0)
    sem = results.std(axis=0, ddof=1) / np.sqrt(len(results))
    with plt.rc_context({"font.size": 12}):
        figure, axis = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
        width = 0.18
        positions = np.arange(len(orders))
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            axis.bar(positions + (i - 1.5) * width, means[:, i], width=width, color=color,
                     label=method, yerr=sem[:, i],
                     error_kw={"elinewidth": 1, "capsize": 2})
        axis.set_xticks(positions, [f"p={order}" for order in orders])
        axis.set_ylabel("Normalized error reduction")
        # Reduction is a fraction and can be zero or negative: use a linear axis.
        low = min(0.0, float(np.min(means - sem)))
        high = max(0.05, float(np.max(means + sem)))
        span = high - low
        axis.set_ylim(low - (0.08 * span if low < 0 else 0), high + 0.12 * span)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
                    frameon=False, columnspacing=1.4, handlelength=1.5)
    return figure, means, sem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("bumps", "advection"), default="bumps")
    parser.add_argument("--checkpoint", type=Path,
                        help="defaults to the selected case's checkpoint")
    parser.add_argument("--orders", type=int, nargs="+",
                        help="default: 2 3 5 for bumps; 2 4 for advection")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4,
                        help="independent test processes; 0 runs serially")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    source_type = AdvectionBump if args.case == "advection" else StaticBump
    args.orders = args.orders or ((2, 4) if args.case == "advection" else ORDERS)
    checkpoint_dir = "advection_IPN" if args.case == "advection" else "static_bump"
    args.checkpoint = args.checkpoint or ROOT / "data" / checkpoint_dir / "checkpoint.pt"
    args.output = args.output or ROOT / "evaluation_plot" / f"{source_type.__module__}_degree.png"
    if args.episodes < 2:
        parser.error("at least two episodes are needed to estimate SEM")
    if any(order < 1 for order in args.orders):
        parser.error("orders must be positive")
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    _, saved = load_policy(args.checkpoint)
    train_start, train_end = saved["seed"], saved["seed"] + saved["episodes"]
    if args.seed < train_end and args.seed + args.episodes > train_start:
        parser.error("test seeds overlap the checkpoint's training seeds")
    print(f"{saved['source']} -> {source_type.__module__} | trained episodes={saved['episodes']} | "
          f"train p={saved['environment']['order']} | test p={args.orders} | "
          f"test episodes={args.episodes} | budget={source_type().settings['budget']} | "
          f"seeds={args.seed}:{args.seed + args.episodes} | IPN=argmax", flush=True)
    seeds = range(args.seed, args.seed + args.episodes)
    if args.workers == 0:
        initialize_worker(args.checkpoint, args.orders, args.case)
        rows = map(evaluate, seeds)
        results = collect(rows, args.episodes)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn"),
                                 initializer=initialize_worker,
                                 initargs=(args.checkpoint, args.orders, args.case)) as pool:
            results = collect(pool.map(evaluate, seeds), args.episodes)
    figure, means, sem = draw(results, args.orders)
    for index, order in enumerate(args.orders):
        print(f"p={order}")
        for method, mean, error in zip(METHODS, means[index], sem[index]):
            print(f"  {method:10s} {mean:.6f} +/- {error:.6f} (SEM)")
    figure.savefig(args.output, dpi=200)
    print(f"Figure: {args.output}", flush=True)
    import matplotlib.pyplot as plt
    if not args.no_show:
        plt.show()
    plt.close(figure)


def collect(rows, total):
    results = []
    for i, row in enumerate(rows, 1):
        results.append(row)
        if i % 10 == 0 or i == total:
            print(f"Test {i}/{total}", flush=True)
    return np.asarray(results)


if __name__ == "__main__":
    main()
