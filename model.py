"""Independent Policy Network  static AMR observations."""

from torch import nn


INPUT_CHANNELS = 2
OBSERVATION_SIZE = 24
CONV_CHANNELS = 6
KERNEL_SIZE = 5
STRIDE = 2
PADDING = 0
HIDDEN_SIZES = (128, 64)


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
