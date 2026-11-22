from functools import partial

import torch
from torch import nn

class VASONet(nn.Module):
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, coordinate_width=18):
        super().__init__()
        self.coordinate_width = coordinate_width
        linear = partial(nn.Linear, dtype=torch.float64)
        self.geo_mlp = nn.Sequential(
            linear(coordinate_width, 320), nn.ReLU(), nn.Dropout(0.0064)
        )

        self.topo_mlp = nn.Sequential(
            linear(1, 32), nn.ReLU(),
            linear(32, 32), nn.ReLU(),
            linear(32, 32), nn.ReLU(),
            linear(32, 32), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            linear(352, 512), nn.ReLU(), nn.Dropout(0.0330),
            linear(512, 512), nn.ReLU(), nn.Dropout(0.0330),
            linear(512, 512), nn.ReLU(), nn.Dropout(0.0330),
            linear(512, 2)
        )
        self.apply(self.init_weights)

    def forward(self, coordinates, valence):
        x = coordinates.reshape(-1, self.coordinate_width)
        n = valence.reshape(-1, 1)

        hg = self.geo_mlp(x)
        ht = self.topo_mlp(n)

        h = torch.cat([hg, ht], dim=1)
        return self.decoder(h)
