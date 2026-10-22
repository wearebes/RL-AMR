"""Bounded crash reproduction; leaves the training protocol unchanged."""

import argparse
import faulthandler
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--mode", choices=("train", "fem"), default="train")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    faulthandler.enable(all_threads=True)
    import numpy as np
    import torch
    import train
    import static_bump

    for name in ("train.py", "static_bump.py", "model.py", "diagnose_crash.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    (output / "environment.json").write_text(json.dumps({
        "python": sys.version, "executable": sys.executable, "pid": os.getpid(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "mfem")},
        "cuda": torch.version.cuda, "args": vars(args),
    }, indent=2))
    started = time.monotonic()
    events = (output / "events.jsonl").open("w", buffering=1)
    actions = (output / "actions.jsonl").open("w", buffering=1)
    memory = (output / "memory.jsonl").open("w", buffering=1)
    position = {"episode": 0, "step": 0}
    def log(event, **extra):
        events.write(json.dumps({"seconds": time.monotonic() - started,
                                 **position, "event": event, **extra}) + "\n")
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
                          if ":" in line)
            memory.write(json.dumps({"seconds": time.monotonic() - started,
                **position, **{key: status.get(key, "").strip()
                for key in ("VmRSS", "VmHWM", "VmSize", "Threads")},
                "cuda_allocated": torch.cuda.memory_allocated(),
                "cuda_reserved": torch.cuda.memory_reserved()}) + "\n")
            stop.wait(5)
    class TracedBump(static_bump.StaticBump):
        def reset(self, seed=0, bumps=None):
            position.update(episode=seed + 1, step=0)
            log("reset.begin", seed=seed)
            result = super().reset(seed, bumps)
            log("reset.end", elements=self.mesh.GetNE())
            return result
        def step(self, action):
            position["step"] += 1
            actions.write(json.dumps({**position, "action": int(action)}) + "\n")
            log("step.begin", action=int(action), elements=self.mesh.GetNE())
            result = super().step(action)
            log("step.end", elements=self.mesh.GetNE())
            return result
        def _project(self):
            log("project.begin")
            result = super()._project()
            log("project.end")
            return result
        def _state(self):
            log("state.begin")
            result = super()._state()
            log("state.end")
            return result
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        if args.mode == "train":
            train.train(episodes=args.episodes, output=output / "training", source_type=TracedBump)
        else:
            source = TracedBump()
            rng = np.random.default_rng(0)
            for episode in range(args.episodes):
                state = source.reset(episode)
                for _ in range(source.settings["budget"]):
                    action = int(rng.choice(np.flatnonzero(state["valid_actions"])))
                    state, _ = source.step(action)
                if (episode + 1) % 10 == 0:
                    print(f"FEM episode={episode + 1} error={state['error']}", flush=True)
        log("complete")
    except BaseException as error:
        details = []
        for frame, line in traceback.walk_tb(error.__traceback__):
            if frame.f_code.co_filename.endswith("static_bump.py"):
                details.append({"function": frame.f_code.co_name, "line": line,
                    "locals": {key: {"type": type(value).__name__, "repr": repr(value)[:1500]}
                    for key, value in frame.f_locals.items()
                    if key in ("i", "e", "elements", "integration_points", ".0", "found", "count")}})
        (output / "exception.json").write_text(json.dumps({"error": repr(error), "frames": details}, indent=2))
        raise
    finally:
        stop.set()
        thread.join()
        for file in (events, actions, memory):
            file.close()


if __name__ == "__main__":
    main()
