"""On-policy training. Change the source import to use another AMR problem."""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from model import IPN
from static_bump import StaticBump


# Training
EPISODES = 20_000
BATCH_SIZE = 64
SEED = 0
GAMMA = 0.99
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 500
SOURCE_CLASS = StaticBump
MODEL_CLASS = IPN

# Optimizer
OPTIMIZER_CLASS = torch.optim.Adam
OPTIMIZER_KWARGS = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
                    "weight_decay": 0.0}

# Runtime and output
DEVICE = "cuda"
TORCH_CPU_THREADS = 2 # PyTorch threads, not FEM workers.
PRINT_EVERY = 100
OUTPUT_ROOT = Path(__file__).resolve().parent / "data"
HISTORY_FILE = "history.csv"
CHECKPOINT_FILE = "checkpoint.pt"
PLOT_FILE = "training.png"


def true_reward(previous_error, current_error, initial_error):
    return (previous_error - current_error) / initial_error



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

def train(episodes=EPISODES, batch_size=BATCH_SIZE, seed=SEED, output=None,
          source_type=SOURCE_CLASS):
    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_CPU_THREADS)
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required; activate your GPU-enabled pinn environment")
    policy = MODEL_CLASS().to(device)
    optimizer = OPTIMIZER_CLASS(policy.parameters(), **OPTIMIZER_KWARGS)
    source = source_type()
    output = Path(output) if output is not None else (
        OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"device={device} ({device_name}) episodes={episodes} "
          f"batch_size={batch_size} seed={seed} optimizer={OPTIMIZER_CLASS.__name__} "
          f"lr={OPTIMIZER_KWARGS['lr']}", flush=True)
    print(f"Output: {output}", flush=True)
    losses = []

    with (output / HISTORY_FILE).open("w", newline="") as file:
        history = csv.writer(file)
        history.writerow(["episode", "initial_error", "final_error", "reward", "policy_loss"])
        for episode in range(episodes):
            state = source.reset(seed + episode)
            initial_error = previous_error = state["error"]
            log_probabilities, rewards = [], []
            epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * min(
                episode / EPSILON_DECAY_EPISODES, 1.0)
            done = False
            while not done:
                observation = torch.as_tensor(state["observation"], device=device)
                valid = torch.as_tensor(state["valid_actions"], device=device)
                logits = policy(observation, valid)
                # Sample and differentiate the same exploration-mixture policy.
                probabilities = (1 - epsilon) * logits.softmax(dim=0)
                probabilities = probabilities + epsilon * valid / valid.sum()
                distribution = Categorical(probs=probabilities)
                action = distribution.sample()
                state, done = source.step(action.item())
                rewards.append(true_reward(previous_error, state["error"], initial_error))
                previous_error = state["error"]
                log_probabilities.append(distribution.log_prob(action))

            discounted, future = [], 0.0
            for reward in reversed(rewards):
                future = reward + GAMMA * future
                discounted.append(future)
            weights = torch.tensor([GAMMA ** t * value
                                    for t, value in enumerate(discounted[::-1])],
                                   dtype=torch.float32, device=device)
            episode_loss = -(torch.stack(log_probabilities) * weights).sum()
            losses.append(episode_loss)
            episode_reward = sum(rewards)
            history.writerow([episode + 1, initial_error, state["error"],
                              episode_reward, episode_loss.item()])
            file.flush()

            if len(losses) == batch_size or episode + 1 == episodes:
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.clear()

            if (episode + 1) % PRINT_EVERY == 0 or episode + 1 == episodes:
                print(f"episode={episode + 1}/{episodes} error={state['error']:.6e} "
                      f"reward={episode_reward:.6f} "
                      f"loss={episode_loss.item():.6f}", flush=True)

    torch.save({
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episodes": episodes,
        "seed": seed,
        "batch_size": batch_size,
        "gamma": GAMMA,
        "learning_rate": OPTIMIZER_KWARGS["lr"],
        "optimizer_name": OPTIMIZER_CLASS.__name__,
        "optimizer_settings": OPTIMIZER_KWARGS.copy(),
        "device": str(device),
        "exploration": {"start": EPSILON_START, "end": EPSILON_END,
                        "decay_episodes": EPSILON_DECAY_EPISODES},
        "source": source_type.__module__,
        "environment": source.settings,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
    }, output / CHECKPOINT_FILE)
    plot_history(output / HISTORY_FILE)
    print(f"Saved to {output}", flush=True)
    return policy


if __name__ == "__main__":
    train()
