"""Plot saved evaluation results; no FEM or policy execution."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#554fb0", "#777777", "#d78056", "#73c0d2")


def plot_history(path):
    """Plot final L2 error and total reward for each training episode."""
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].semilogy(data["episode"], data["final_error"], color="#2563eb", linewidth=1)
    axes[0].set(title="Error", ylabel="Final L2 error (log scale)")
    axes[1].plot(data["episode"], data["reward"], color="#2563eb", linewidth=1)
    axes[1].set(title="Reward", ylabel="Sum of step rewards")
    axes[1].axhline(0, color="0.5", linewidth=0.6)
    for axis in axes:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(Path(path).with_name("training.png"), dpi=180)
    plt.close(figure)


def plot(path):
    with np.load(path) as data:
        results = data["reductions"]
        methods, values = data["methods"], data["values"]
        transfer = str(data["mode"]) == "transfer"
    means = results.mean(axis=0)
    sem = results.std(axis=0, ddof=1) / np.sqrt(len(results))
    with plt.rc_context({"font.size": 12}):
        figure, axis = plt.subplots(figsize=(6.4, 4.6 if transfer else 4.0),
                                    layout="constrained")
        width = 0.18
        positions = np.arange(len(values))
        for i, (method, color) in enumerate(zip(methods, COLORS)):
            axis.bar(positions + (i - 1.5) * width, means[:, i], width=width,
                     color=color, label=method, yerr=sem[:, i],
                     error_kw={"elinewidth": 1, "capsize": 2})
        labels = [str(value) if transfer else f"p={value}" for value in values]
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Normalized error reduction")
        if transfer:
            axis.set_xlabel("Budget")
            axis.axhline(0, color="0.4", linewidth=0.7)
        else:
            low = min(0.0, float(np.min(means - sem)))
            high = max(0.05, float(np.max(means + sem)))
            span = high - low
            axis.set_ylim(low - (0.08 * span if low < 0 else 0), high + 0.12 * span)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                    ncol=2 if transfer else 4, frameon=False,
                    columnspacing=1.4, handlelength=1.5)
    output = path.with_suffix(".png")
    figure.savefig(output, dpi=200)
    plt.close(figure)
    print(f"Figure: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[1] / "data/static_bump/degree.npz")
    path = parser.parse_args().data
    if path.suffix == ".csv":
        plot_history(path)
    else:
        plot(path)
