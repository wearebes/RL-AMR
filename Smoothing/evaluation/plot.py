"""Plot saved FSI2 metrics without running or changing the simulation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# The order and visual identity never depend on which runs have finished.
METHODS = (
    ('similarity', 'Similarity', '#777777', '-'),
    ('laplacian', 'Laplacian', '#554fb0', '--'),
    ('laplacian_prcg', 'Laplacian + PR-CG', '#d78056', '-.'),
    ('laplacian_nn', 'Laplacian + NN', '#73c0d2', ':'),
)
# column: (input file, y label, fixed y limits if physically meaningful)
METRICS = {
    'min_q': ('mesh_smoothing.csv', r'$q_{\min}$', (0, 1)),
    'min_J': ('mesh_smoothing.csv', r'$J_{\min}$ (m$^2$)', None),
    'tip_dx': ('evaluation.csv', r'$d_{A,x}$ (m)', None),
    'tip_dy': ('evaluation.csv', r'$d_{A,y}$ (m)', None),
    'drag': ('evaluation.csv', r'$F_D$ (N/m)', None),
    'lift': ('evaluation.csv', r'$F_L$ (N/m)', None),
}
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 10, 'axes.linewidth': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.4, 'legend.frameon': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'pdf.fonttype': 42, 'savefig.dpi': 300,
}


def read_rows(path):
    with path.open(newline='') as file:
        return [{key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(file, skipinitialspace=True)]


def save(figure, output, name):
    for extension in ('pdf', 'png'):
        figure.savefig(output / f'{name}.{extension}')
    plt.close(figure)


def plot_series(root, output, metric):
    filename, label, limits = METRICS[metric]
    figure, axis = plt.subplots(figsize=(3.6, 2.9), layout='constrained')
    count = 0
    end_times = []
    for directory, name, color, style in METHODS:
        path = root / directory / filename
        if not path.exists():
            print(f'{metric}: missing {path}')
            continue
        rows = [row for row in read_rows(path)
                if row.get('record_type', 'step') == 'step']
        if not rows or metric not in rows[0]:
            print(f'{metric}: no column/data in {path}')
            continue
        time = np.array([float(row['time']) for row in rows])
        values = np.array([float(row[metric]) for row in rows])
        if not (np.isfinite(time).all() and np.isfinite(values).all()):
            raise ValueError(f'Non-finite data: {path}, {metric}')
        if np.any(np.diff(time) <= 0):
            raise ValueError(f'Time must increase: {path}')
        axis.plot(time, values, label=name, color=color, linestyle=style)
        settings = root / directory / 'run_settings.json'
        if settings.exists():
            end_times.append(json.loads(settings.read_text())['parameters']['problem_data']['end_time'])
        count += 1
    if not count:
        plt.close(figure)
        return
    axis.set(xlabel=r'$t$ (s)', ylabel=label)
    axis.set_xlim(left=0)
    if end_times:
        axis.set_xlim(0, max(end_times))
    if limits is not None:
        axis.set_ylim(*limits)
    if metric == 'min_J':
        axis.axhline(0, color='0.75', linewidth=0.6, zorder=0)
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.ticklabel_format(axis='both', style='sci', scilimits=(-3, 4), useOffset=False)
    axis.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                ncol=2, fontsize=8, handlelength=2.5, columnspacing=1)
    save(figure, output, metric)
    print(f'{metric}: plotted {count}/4 methods')


def plot_runtime(root, output):
    durations, final_times = [], []
    for directory, *_ in METHODS:
        path = root / directory / 'mesh_smoothing.csv'
        if not path.exists():
            print('runtime: waiting for four completed runs')
            return
        rows = read_rows(path)
        summary = [row for row in rows if row['record_type'] == 'summary']
        if not summary or summary[-1]['completed'].lower() != 'true':
            print(f'runtime: {directory} has no successful completion')
            return
        durations.append(float(summary[-1]['total_seconds']))
        final_times.append(float(summary[-1]['time']))
    if not np.allclose(final_times, final_times[0], rtol=0, atol=1e-9):
        raise ValueError('Runtime comparison requires the same final physical time.')
    if not np.isfinite(durations).all() or min(durations) <= 0:
        raise ValueError('Runtime must be positive and finite.')
    figure, axis = plt.subplots(figsize=(3.6, 2.9), layout='constrained')
    axis.bar(range(4), durations, color=[item[2] for item in METHODS], width=0.6)
    axis.set_xticks(range(4), ['Similarity', 'Laplacian', 'Laplacian\n+ PR-CG', 'Laplacian\n+ NN'], fontsize=8)
    axis.set_ylabel(r'$T_{\mathrm{wall}}$ (s)')
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(MaxNLocator(5))
    save(figure, output, 'runtime')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Directory containing the four method directories')
    parser.add_argument('--output', type=Path, help='Default: Smoothing/plot/fig')
    parser.add_argument('--metrics', nargs='+', choices=[*METRICS, 'runtime'],
                        default=[*METRICS, 'runtime'])
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parents[1] / 'plot' / 'fig'
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for metric in args.metrics:
            if metric == 'runtime':
                plot_runtime(args.root, output)
            else:
                plot_series(args.root, output, metric)


if __name__ == '__main__':
    main()
