"""YAML configuration, shared FEM training loop and command-line entry."""

import argparse
import csv
from pathlib import Path

# Load the FEM runtime before torch to avoid local OpenMP initialization conflicts.
from FEM.advection_bump import AdvectionBump
from FEM.static_bump import StaticBump
from FEM.workers import FEMWorkers

import numpy as np
import torch
import yaml

from model.ipn import IPN
from train.learning import select_action, true_reward, episode_loss, training_settings

ROOT = Path(__file__).resolve().parents[1]
CASES = {"bumps": StaticBump, "advection": AdvectionBump}
HISTORY_FILE = "history.csv"
CHECKPOINT_FILE = "checkpoint.pt"


def collect_wave(policy, sources, seeds, first_episode, device, learning):
    """Finish one wave with the current policy; keep loss graphs only for this wave.

    Worker slots start at zero in each wave. first_episode is the global,
    zero-based episode index used by the exploration schedule.
    """
    states = sources.reset(seeds)
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
            actions[slot], log_probs[slot] = select_action(
                policy, observation, valid, first_episode + slot, learning["exploration"])
        results = sources.step(actions)
        next_active = []
        for slot in active:
            state, done = results[slot]
            states[slot] = state
            rewards[slot].append(true_reward(
                previous[slot], state["error"], initial[slot], learning["reward"]["scale"]))
            previous[slot] = state["error"]
            log_probabilities[slot].append(log_probs[slot])
            if not done:
                next_active.append(slot)
        active = next_active

    for slot, state in states.items():
        loss = episode_loss(log_probabilities[slot], rewards[slot], learning["gamma"])
        yield (first_episode + slot + 1, initial[slot], state["error"],
               sum(rewards[slot]), loss)


def save_checkpoint(output, policy, optimizer, metadata, completed):
    """Atomically save weights, optimizer state and the run's numerical settings."""
    temporary = output / (CHECKPOINT_FILE + ".tmp")
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                **metadata, "episodes": completed}, temporary)
    temporary.replace(output / CHECKPOINT_FILE)


def train(config):
    """Run the shared on-policy loop using the resolved YAML configuration."""
    run = config["training"]
    learning_config = config["learning"]
    case, episodes = run["case"], run["episodes"]
    batch_size, seed = run["batch_size"], run["seed"]
    num_workers, device = run["num_workers"], run["device"]
    torch.manual_seed(seed)
    torch.set_num_threads(run["torch_cpu_threads"])
    policy = IPN().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), **learning_config["adam"])
    learning = training_settings(learning_config)
    source_type = CASES[case]
    wave_size = num_workers or 1
    output = ROOT / (run["output"] or f"data/{case}_{type(policy).__name__}")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "config.yaml").open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    print(f"case={case} model={type(policy).__name__} device={device} episodes={episodes} "
          f"batch_size={batch_size} seed={seed} optimizer={learning['optimizer_name']} "
          f"lr={learning['learning_rate']} num_workers={num_workers}", flush=True)
    print(f"Output: {output}", flush=True)

    with FEMWorkers(source_type, num_workers) as sources, \
            (output / HISTORY_FILE).open("w", newline="") as file:
        metadata = {
            "target_episodes": episodes, "seed": seed, "batch_size": batch_size,
            "num_workers": num_workers, **learning, "device": str(device),
            "model_class": type(policy).__name__, "source": source_type.__module__,
            "environment": sources.settings.copy(), "config": config,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        }
        history = csv.writer(file)
        history.writerow(["episode", "initial_error", "final_error", "reward", "policy_loss"])
        for batch_start in range(0, episodes, batch_size):
            batch_end = min(batch_start + batch_size, episodes)
            optimizer.zero_grad()
            # Finish this exact training batch before dispatching any next-policy episodes.
            for wave_start in range(batch_start, batch_end, wave_size):
                wave_end = min(wave_start + wave_size, batch_end)
                seeds = [seed + episode for episode in range(wave_start, wave_end)]
                wave_loss = []
                for episode, initial, final, reward, episode_loss in collect_wave(
                        policy, sources, seeds, wave_start, device, learning_config):
                    wave_loss.append(episode_loss)
                    history.writerow([episode, initial, final, reward, episode_loss.item()])
                    file.flush()
                    if episode % run["print_every"] == 0 or episode == episodes:
                        print(f"episode={episode}/{episodes} error={final:.6e} "
                              f"reward={reward:.6f} "
                              f"loss={episode_loss.item():.6f}", flush=True)

                # Keep this batch's policy fixed; release each wave after backward.
                (torch.stack(wave_loss).sum() / (batch_end - batch_start)).backward()
                del wave_loss, episode_loss
            optimizer.step()
            save_checkpoint(output, policy, optimizer, metadata, batch_end)

    from evaluation.plot import plot_history
    plot_history(output / HISTORY_FILE)
    print(f"Saved to {output}", flush=True)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("training.yaml"))
    args = parser.parse_args()
    with args.config.open() as file:
        config = yaml.safe_load(file)
    train(config)


if __name__ == "__main__":
    main()
