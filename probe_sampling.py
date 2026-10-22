"""Small CPU-only probe for the MFEM sampling boundary."""

import argparse
import faulthandler
import json
from pathlib import Path
import time

import numpy as np
from static_bump import StaticBump

faulthandler.enable()
parser = argparse.ArgumentParser()
parser.add_argument("--actions", required=True)
parser.add_argument("--first", type=int, default=1)
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()
records = [json.loads(line) for line in Path(args.actions).read_text().splitlines()]
grouped = {}
for record in records:
    grouped.setdefault(record["episode"], []).append(record["action"])
source = StaticBump()
started = time.monotonic()
for repeat in range(args.repeat):
    for episode, actions in grouped.items():
        if episode < args.first:
            continue
        print(f"reset repeat={repeat} episode={episode}", flush=True)
        source.reset(episode - 1)
        for step, action in enumerate(actions, 1):
            print(f"step episode={episode} step={step} action={action}", flush=True)
            state, _ = source.step(action)
        if not np.isfinite(state["observation"]).all():
            raise ValueError("Nonfinite observation")
print(f"COMPLETE seconds={time.monotonic() - started}", flush=True)
