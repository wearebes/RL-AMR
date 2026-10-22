"""Select a case and model; train all cases with one shared loop."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from advection_bump import AdvectionBump
from fem_workers import FEMWorkers
from model import IPN
from static_bump import StaticBump


CASES = {"bumps": StaticBump, "advection": AdvectionBump}
CASE = "advection"
MODEL_CLASS = IPN

EPISODES = 20_000
BATCH_SIZE = 128
SEED = 0
DEVICE = "cuda"
TORCH_CPU_THREADS = 2  # PyTorch threads, not FEM workers.
NUM_WORKERS = 16  # Independent FEM processes.
PRINT_EVERY = 100

OUTPUT_ROOT = Path(__file__).resolve().parent / "data"
HISTORY_FILE = "history.csv"
CHECKPOINT_FILE = "checkpoint.pt"
PLOT_FILE = "training.png"


def plot_history(path):
    """Plot final L2 error and total reward for each training episode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    figure.savefig(Path(path).with_name(PLOT_FILE), dpi=180)
    plt.close(figure)

def train(case=CASE, episodes=EPISODES, batch_size=BATCH_SIZE, seed=SEED, output=None,
          num_workers=NUM_WORKERS,
          model_type=MODEL_CLASS, device=DEVICE):
    # Setup before the training loop.
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_CPU_THREADS)
    policy = model_type().to(device)
    optimizer = policy.make_optimizer()
    learning = policy.training_settings()
    source_type = CASES[case]
    wave_size = num_workers or 1
    output = Path(output or OUTPUT_ROOT / f"{case}_{model_type.__name__}")
    output.mkdir(parents=True, exist_ok=False)
    print(f"case={case} model={model_type.__name__} device={device} episodes={episodes} "
          f"batch_size={batch_size} seed={seed} optimizer={learning['optimizer_name']} "
          f"lr={learning['learning_rate']} num_workers={num_workers}", flush=True)
    print(f"Output: {output}", flush=True)

    def save_checkpoint(completed):
        temporary = output / (CHECKPOINT_FILE + ".tmp")
        torch.save({
            "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "episodes": completed, "target_episodes": episodes,
            "seed": seed, "batch_size": batch_size, "num_workers": num_workers,
            **learning, "device": str(device), "model_class": model_type.__name__,
            "source": source_type.__module__, "environment": environment,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        }, temporary)
        temporary.replace(output / CHECKPOINT_FILE)

    with FEMWorkers(source_type, num_workers) as sources, \
            (output / HISTORY_FILE).open("w", newline="") as file:
        environment = sources.settings.copy()
        history = csv.writer(file)
        history.writerow(["episode", "initial_error", "final_error", "reward", "policy_loss"])
        for batch_start in range(0, episodes, batch_size):
            batch_end = min(batch_start + batch_size, episodes)
            losses = []
            # Finish this exact training batch before dispatching any next-policy episodes.
            for wave_start in range(batch_start, batch_end, wave_size):
                wave_end = min(wave_start + wave_size, batch_end)
                states = sources.reset([seed + episode for episode in range(wave_start, wave_end)])
                initial = {slot: state["error"] for slot, state in states.items()}
                previous = initial.copy()
                log_probabilities = {slot: [] for slot in states}
                rewards = {slot: [] for slot in states}
                active = list(states)
                while active:
                    actions, log_probs = {}, {}
                    for slot in active:
                        state = states[slot]
                        observation = torch.as_tensor(state["observation"], device=device)
                        valid = torch.as_tensor(state["valid_actions"], device=device)
                        actions[slot], log_probs[slot] = policy.select_action(
                            observation, valid, wave_start + slot)
                    results = sources.step(actions)
                    next_active = []
                    for slot in active:
                        state, done = results[slot]
                        states[slot] = state
                        rewards[slot].append(policy.true_reward(previous[slot], state["error"], initial[slot]))
                        previous[slot] = state["error"]
                        log_probabilities[slot].append(log_probs[slot])
                        if not done:
                            next_active.append(slot)
                    active = next_active

                for slot, state in states.items():
                    episode = wave_start + slot
                    episode_loss = policy.episode_loss(log_probabilities[slot], rewards[slot])
                    losses.append(episode_loss)
                    episode_reward = sum(rewards[slot])
                    history.writerow([episode + 1, initial[slot], state["error"],
                                      episode_reward, episode_loss.item()])
                    file.flush()
                    if (episode + 1) % PRINT_EVERY == 0 or episode + 1 == episodes:
                        print(f"episode={episode + 1}/{episodes} error={state['error']:.6e} "
                              f"reward={episode_reward:.6f} "
                              f"loss={episode_loss.item():.6f}", flush=True)

            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            save_checkpoint(batch_end)

    plot_history(output / HISTORY_FILE)
    print(f"Saved to {output}", flush=True)
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default=CASE)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="independent FEM processes (0: serial reference)")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--output")
    args = parser.parse_args()
    train(case=args.case, episodes=args.episodes,
          batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
          device=args.device, output=args.output)
