from functools import partial
from typing import Any
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
        linear = partial(nn.Linear, dtype=torch.float32)
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

class quadrilateral(nn.Module):
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, input_points=36):
        super().__init__()
        self.coordinate_width = input_points
        linear=partial(nn.Linear, dtype=torch.float32)
        self.geo_mlp = nn.Sequential(
            linear(input_points, 320), nn.ReLU(), nn.Dropout(0.0064)
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
    
# CNN coordinates process, 2D and 3D mode
class CNNSmoothing(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        assert dim in (2, 3)
        self.features = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1024, kernel_size=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=(20, 1))
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )

    def forward(self, x):
        B, N, D = x.shape
        assert N == 20
        # (B, 20, dim) -> (B, dim, 20, 1)
        x = x.transpose(1, 2).unsqueeze(-1)
        # (B, dim, 20, 1) -> (B, 1024, 20, 1)
        x = self.features(x)
        # (B, 1024, 20, 1) -> (B, 1024)
        x = self.pool(x).flatten(1)
        # (B, 1024) -> (B, dim)
        return self.regressor(x)