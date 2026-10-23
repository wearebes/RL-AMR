"""Shared bump sampling and physical-space FEM observation construction."""

import numpy as np

ELEMENT_SAMPLES = 16
CONTEXT_SAMPLES = 4
WINDOW_SIZE = ELEMENT_SAMPLES + 2 * CONTEXT_SAMPLES


def sample_bumps(seed, count_range, center_range, width_range, bumps=None):
    rng = np.random.default_rng(seed)
    if bumps is None:
        count = rng.integers(count_range[0], count_range[1] + 1)
        bumps = np.column_stack((rng.uniform(*center_range, (count, 2)),
                                 rng.uniform(*width_range, count)))
    return np.array(bumps, dtype=np.float64, copy=True)


def observation_points(lower, upper):
    """Pixel centers with a context border; callers apply their boundary rule."""
    positions = (np.arange(WINDOW_SIZE) - CONTEXT_SAMPLES + 0.5) / ELEMENT_SAMPLES
    xx, yy = np.meshgrid(positions, positions, indexing="xy")
    local = np.stack((xx, yy), axis=-1)
    return lower[:, None, None, :] + (upper - lower)[:, None, None, :] * local


def locate_samples(mesh, points):
    unique, inverse = np.unique(points.reshape(-1, 2), axis=0, return_inverse=True)
    _, elements, integration_points = mesh.FindPoints(unique, warn=False)
    return elements, integration_points, inverse


def sample_observation(solution, depths, elements, integration_points, inverse):
    """Channels: numerical solution and neighbor depth; action 0 has a zero image."""
    count = len(depths)
    values = np.fromiter((solution.GetValue(int(e), integration_points[i])
                          for i, e in enumerate(elements)), dtype=np.float64,
                         count=len(elements))
    observation = np.zeros((count + 1, 2, WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
    observation[1:, 0] = values[inverse].reshape(count, WINDOW_SIZE, WINDOW_SIZE)
    sampled_depth = depths[np.asarray(elements)[inverse]]
    observation[1:, 1] = sampled_depth.reshape(count, WINDOW_SIZE, WINDOW_SIZE)
    return observation
