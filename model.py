"""Independent Policy Network and shared on-policy learning rules."""

import torch
from torch import nn
from torch.distributions import Categorical


INPUT_CHANNELS = 2
OBSERVATION_SIZE = 24
CONV_CHANNELS = 6
KERNEL_SIZE = 5
STRIDE = 2
PADDING = 0
HIDDEN_SIZES = (128, 64)

OPTIMIZER_CLASS = torch.optim.Adam
OPTIMIZER_KWARGS = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
                    "weight_decay": 0.0}
GAMMA = 0.99
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 500


class IPN(nn.Module):

    def __init__(self):
        super().__init__()
        side = (OBSERVATION_SIZE + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
        features = CONV_CHANNELS * side * side
        self.network = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CONV_CHANNELS, kernel_size=KERNEL_SIZE,
                      stride=STRIDE, padding=PADDING),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(features, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[1], 1),
        )

    def forward(self, observation, valid_actions):
        logits = self.network(observation).squeeze(-1)
        return logits.masked_fill(~valid_actions, float("-inf"))

    def make_optimizer(self):
        return OPTIMIZER_CLASS(self.parameters(), **OPTIMIZER_KWARGS)

    @staticmethod
    def true_reward(previous_error, current_error, initial_error):
        return (previous_error - current_error) / initial_error

    def select_action(self, observation, valid_actions, episode):
        logits = self(observation, valid_actions)
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * min(
            episode / EPSILON_DECAY_EPISODES, 1.0)
        probabilities = (1 - epsilon) * logits.softmax(dim=0)
        probabilities = probabilities + epsilon * valid_actions / valid_actions.sum()
        distribution = Categorical(probs=probabilities)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def episode_loss(self, log_probabilities, rewards):
        discounted, future = [], 0.0
        for reward in reversed(rewards):
            future = reward + GAMMA * future
            discounted.append(future)
        weights = torch.tensor([GAMMA ** t * value
                                for t, value in enumerate(discounted[::-1])],
                               dtype=torch.float32, device=log_probabilities[0].device)
        return -(torch.stack(log_probabilities) * weights).sum()

    def training_settings(self):
        return {
            "gamma": GAMMA, "learning_rate": OPTIMIZER_KWARGS["lr"],
            "optimizer_name": OPTIMIZER_CLASS.__name__,
            "optimizer_settings": OPTIMIZER_KWARGS.copy(),
            "exploration": {"start": EPSILON_START, "end": EPSILON_END,
                            "decay_episodes": EPSILON_DECAY_EPISODES},
        }
