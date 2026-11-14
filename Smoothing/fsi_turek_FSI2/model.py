"""Dual MLP architecture shared with smoothing-dualMLP.ipynb checkpoints."""

import torch
from torch import nn

class VASONet(nn.Module):
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self):
        super().__init__()
        self.geo_mlp = nn.Sequential(
            nn.Linear(18, 320),nn.ReLU(),nn.Dropout(0.0064)
        )

        self.topo_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(352, 512), nn.ReLU(), nn.Dropout(0.0330),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.0330),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.0330),
            nn.Linear(512, 2)
        )
        self.apply(self.init_weights)

    def forward(self, coordinates, valence):
        x = coordinates.reshape(-1, 18)
        n = valence.reshape(-1, 1)

        hg = self.geo_mlp(x)
        ht = self.topo_mlp(n)

        h = torch.cat([hg, ht], dim=1)
        return self.decoder(h)
