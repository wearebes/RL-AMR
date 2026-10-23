"""Run one evaluation from evaluation.yaml and save per-episode results."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path

# Each process owns its solver and uses one numerical-library thread.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import mfem.ser as mfem
import numpy as np
import torch
import yaml

from model.ipn import IPN
from FEM.static_bump import StaticBump
from FEM.advection_bump import AdvectionBump

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("IPN", "Random", "ZZ", "TrueError")


def load_policy(checkpoint):
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = IPN().eval()
    policy.load_state_dict(saved["model"])
    return policy, saved


def indicator_action(source, method):
    """Mark the largest local indicator among refinable elements.

    TrueError uses local L2 errors. ZZ uses an H1 Positive recovery space
    (unit diffusion), preserving the original comparison setup.
    """
    valid = source.state["valid_actions"][1:]
    if not valid.any():
        return 0
    if method == "TrueError":
        scores = source.element_errors()
    else:  # ZZ: recover the gradient in a continuous H1 space.
        recovery = mfem.H1_FECollection(source.settings["order"], 2, mfem.BasisType.Positive)
        flux_space = mfem.FiniteElementSpace(source.mesh, recovery, 2)
        integrator = mfem.DiffusionIntegrator()
        estimator = mfem.ZienkiewiczZhuEstimator(integrator, source.solution, flux_space)
        estimator.SetFluxAveraging(0)  # Arithmetic averaging, including attribute interfaces.
        scores = estimator.GetLocalErrors().GetDataArray().copy()
    return int(np.argmax(np.where(valid, scores, -np.inf))) + 1


def initialize_worker(config):
    global POLICY, CONFIG
    CONFIG = config
    POLICY, _ = load_policy(ROOT / config["checkpoint"])


def evaluate_degree(seed):
    """Every degree and method sees the same initial function for this seed."""
    source = AdvectionBump() if CONFIG["case"] == "advection" else StaticBump()
    reductions, environments = [], []
    for order in CONFIG["orders"]:
        source.settings["order"] = order
        environments.append(source.settings.copy())
        if CONFIG["case"] == "advection":
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
            reference_error = reference if CONFIG["case"] == "advection" else initial
            row.append((reference_error - state["error"]) / initial)
        reductions.append(row)
    return reductions, environments


def evaluate_transfer(seed):
    """Paired budgets share one IC and prefixes of the same trajectory."""
    budgets = CONFIG["budgets"]
    rng = np.random.default_rng(seed)
    final_errors = np.empty((len(budgets), len(METHODS)))
    no_refine = np.empty(len(budgets))
    for method in ("NoRefine",) + METHODS:
        source = AdvectionBump(observe=method == "IPN")
        source.settings["budget"] = max(budgets)
        state = source.reset(seed)
        initial_error = state["error"]
        for step in range(1, max(budgets) + 1):
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
            if step in budgets:
                index = budgets.index(step)
                if method == "NoRefine":
                    no_refine[index] = state["error"]
                else:
                    final_errors[index, METHODS.index(method)] = state["error"]
    # Paper section 5.2: compare against no refinement at the SAME final time.
    reductions = (no_refine[:, None] - final_errors) / initial_error
    environments = [{**source.settings, "budget": budget} for budget in budgets]
    return reductions, environments


def collect(rows, total):
    results = []
    for i, row in enumerate(rows, 1):
        results.append(row)
        if i % 10 == 0 or i == total:
            print(f"Test {i}/{total}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default="bumps")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("evaluation.yaml"))
    args = parser.parse_args()
    with args.config.open() as file:
        settings = yaml.safe_load(file)
    config = {**settings["defaults"], **settings["experiments"][args.experiment]}
    values = config["orders"] if config["mode"] == "degree" else config["budgets"]
    saved = torch.load(ROOT / config["checkpoint"], map_location="cpu", weights_only=False)
    print(f"{args.experiment}: {config}", flush=True)
    evaluate = evaluate_degree if config["mode"] == "degree" else evaluate_transfer
    seeds = range(config["seed"], config["seed"] + config["episodes"])
    if config["workers"] == 0:
        initialize_worker(config)
        rows = map(evaluate, seeds)
        results = collect(rows, config["episodes"])
    else:
        with ProcessPoolExecutor(max_workers=config["workers"],
                                 mp_context=mp.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(config,)) as pool:
            results = collect(pool.map(evaluate, seeds), config["episodes"])
    reductions, environments = zip(*results)
    output = ROOT / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, reductions=np.asarray(reductions),
                        test_environments=yaml.safe_dump(environments[0]), methods=METHODS,
                        zz_recovery_basis="Positive",
                        values=values, mode=config["mode"], seeds=np.asarray(list(seeds)),
                        config=yaml.safe_dump(config), trained_episodes=saved["episodes"],
                        training_seed=saved["seed"], training_source=saved["source"],
                        training_environment=yaml.safe_dump(saved["environment"]))
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
