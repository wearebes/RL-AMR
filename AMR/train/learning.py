"""On-policy learning rules; all tunable values come from train/training.yaml."""

import torch
from torch.distributions import Categorical


def true_reward(previous_error, current_error, initial_error, scale):
    return scale * (previous_error - current_error) / initial_error


def select_action(policy, observation, valid_actions, episode, exploration):
    logits = policy(observation, valid_actions)
    epsilon = exploration["start"] + (exploration["end"] - exploration["start"]) * min(
        episode / exploration["decay_episodes"], 1.0)
    probabilities = (1 - epsilon) * logits.softmax(dim=0)
    probabilities = probabilities + epsilon * valid_actions / valid_actions.sum()
    distribution = Categorical(probs=probabilities)
    action = distribution.sample()
    return action.item(), distribution.log_prob(action)


def episode_loss(log_probabilities, rewards, gamma):
    discounted, future = [], 0.0
    for reward in reversed(rewards):
        future = reward + gamma * future
        discounted.append(future)
    weights = torch.tensor([gamma ** t * value
                            for t, value in enumerate(discounted[::-1])],
                           dtype=torch.float32, device=log_probabilities[0].device)
    return -(torch.stack(log_probabilities) * weights).sum()


def training_settings(learning):
    """Retain existing checkpoint fields alongside the complete run config."""
    return {
        "gamma": learning["gamma"], "learning_rate": learning["adam"]["lr"],
        "optimizer_name": "Adam",
        "optimizer_settings": {**learning["adam"], "betas": tuple(learning["adam"]["betas"])},
        "exploration": learning["exploration"].copy(),
        "reward": learning["reward"].copy(),
    }
