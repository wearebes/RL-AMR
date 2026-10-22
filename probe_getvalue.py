"""Stress a fixed MFEM field's scalar sampling without training or refinement."""
import argparse
import faulthandler
import json
from pathlib import Path
import time
import numpy as np
from static_bump import StaticBump

faulthandler.enable()
parser = argparse.ArgumentParser()
parser.add_argument("--repeat", type=int, default=1000)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--state", action="store_true")
parser.add_argument("--actions")
parser.add_argument("--episode", type=int, default=410)
args = parser.parse_args()
if args.torch:
    import torch
source = StaticBump()
source.reset(args.episode - 1 if args.actions else 0)
if args.actions:
    for line in Path(args.actions).read_text().splitlines():
        record = json.loads(line)
        if record["episode"] == args.episode:
            source.step(record["action"])
    print(f"refined mesh elements={source.mesh.GetNE()} episode={args.episode}", flush=True)
x = np.linspace(0, 1, 200)
xx, yy = np.meshgrid(x, x)
points = np.column_stack((xx.ravel(), yy.ravel()))
found, elements, integration_points = source.mesh.FindPoints(points, warn=False)
assert found == len(points)
started = time.monotonic()
for repeat in range(args.repeat):
    if args.state:
        values = source._state()["observation"]
    else:
        values = np.fromiter((source.solution.GetValue(e, integration_points[i])
                             for i, e in enumerate(elements)), dtype=np.float64, count=len(points))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite sample")
    if (repeat + 1) % 25 == 0:
        operation = "state_rebuilds" if args.state else "sample_calls"
        count = repeat + 1 if args.state else (repeat + 1) * len(points)
        print(f"{operation}={count} seconds={time.monotonic() - started}", flush=True)
print("COMPLETE", flush=True)
