"""Static bumps: MFEM simulation and observations, independent of the policy."""

from importlib.metadata import version
import mfem.ser as mfem
import numpy as np


# Mesh and approximation
NX = 8
NY = 8
ORDER = 2
DOMAIN_SIZE = 1.0
REFINEMENT_BUDGET = 10
MAX_DEPTH = 3

# Function distribution
BUMP_COUNT = (1, 6)
CENTER_RANGE = (0.2, 0.9)
WIDTH_RANGE = (0.05, 0.2)

# Observation sampling
ELEMENT_SAMPLES = 16
CONTEXT_SAMPLES = 4
WINDOW_SIZE = ELEMENT_SAMPLES + 2 * CONTEXT_SAMPLES


class BumpFunction(mfem.PyCoefficient):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters

    def EvalValue(self, x):
        cx, cy, width = self.parameters.T
        return float(np.exp(-((x[0] - cx) ** 2 + (x[1] - cy) ** 2) / width).sum())


class StaticBump:
    """Sources expose reset(seed) and step(action), returning state and done.

    State contains observation [N+1, 2, 24, 24], valid_actions [N+1],
    and the global true L2 error. Action 0 does nothing; action i+1
    refines current MFEM element i. Only observation enters the network.
    """

    def __init__(self):
        self.settings = {
            "nx": NX, "ny": NY, "order": ORDER,
            "budget": REFINEMENT_BUDGET, "max_depth": MAX_DEPTH,
            "domain_size": DOMAIN_SIZE,
            "bump_count": BUMP_COUNT, "center_range": CENTER_RANGE,
            "width_range": WIDTH_RANGE,
            "element_samples": ELEMENT_SAMPLES, "context_samples": CONTEXT_SAMPLES,
            "observation_boundary": "replicate",
            "observation_sampling": "pixel centers",
            "mfem_version": version("mfem"),
        }
        self.mesh = self.fec = self.fes = self.solution = None

    def reset(self, seed=0, bumps=None):
        rng = np.random.default_rng(seed)
        if bumps is None:
            count = rng.integers(BUMP_COUNT[0], BUMP_COUNT[1] + 1)
            bumps = np.column_stack((rng.uniform(*CENTER_RANGE, (count, 2)),
                                     rng.uniform(*WIDTH_RANGE, count)))
        self.bumps = np.array(bumps, dtype=np.float64, copy=True)
        self.exact = BumpFunction(self.bumps)
        # Release dependent MFEM objects before replacing their mesh.
        self.solution = self.fes = self.fec = self.mesh = None
        self.mesh = mfem.Mesh.MakeCartesian2D(self.settings["nx"], self.settings["ny"],
                                              mfem.Element.QUADRILATERAL, True,
                                              DOMAIN_SIZE, DOMAIN_SIZE)
        self.mesh.EnsureNCMesh()
        self.fec = mfem.H1_FECollection(self.settings["order"], 2, mfem.BasisType.Positive)
        self.fes = mfem.FiniteElementSpace(self.mesh, self.fec)
        self.solution = mfem.GridFunction(self.fes)
        self.steps = 0
        self._project()
        self.state = self._state()
        if self.state["error"] <= 0:
            raise ValueError("Initial L2 error must be positive for normalized reward")
        return self.state

    def _project(self):
        self.solution.ProjectCoefficient(self.exact)
        self.solution.RestrictConforming()

    def step(self, action):
        if self.steps >= self.settings["budget"]:
            raise RuntimeError("Episode finished; call reset before stepping again")
        if not 0 <= action < len(self.state["valid_actions"]) or not self.state["valid_actions"][action]:
            raise ValueError("Invalid action for the current mesh")
        if action != 0:
            self.mesh.GeneralRefinement(mfem.intArray([action - 1]), 1, 0)
            self.fes.Update()
            self.solution.Update()
            self._project()
            self.state = self._state()
        self.steps += 1
        return self.state, self.steps == self.settings["budget"]

    def _state(self):
        count = self.mesh.GetNE()
        vertices = np.asarray(self.mesh.GetVertexArray())
        corners = np.array([vertices[self.mesh.GetElementVertices(i)] for i in range(count)])
        lower, upper = corners.min(axis=1), corners.max(axis=1)
        depths = np.array([self.mesh.ncmesh.GetElementDepth(i) for i in range(count)])

        # Sample the FEM solution in physical space, including actual neighbors.
        positions = (np.arange(WINDOW_SIZE) - CONTEXT_SAMPLES + 0.5) / ELEMENT_SAMPLES
        xx, yy = np.meshgrid(positions, positions, indexing="xy")
        local = np.stack((xx, yy), axis=-1)
        points = lower[:, None, None, :] + (upper - lower)[:, None, None, :] * local
        points = np.clip(points, 0.0, DOMAIN_SIZE).reshape(-1, 2)
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        found, elements, integration_points = self.mesh.FindPoints(unique, warn=False)
        if found != len(unique):
            raise RuntimeError("MFEM could not locate observation sample points")
        values = np.fromiter((self.solution.GetValue(e, integration_points[i])
                              for i, e in enumerate(elements)), dtype=np.float64,
                             count=len(unique))
        observation = np.zeros((count + 1, 2, WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
        observation[1:, 0] = values[inverse].reshape(count, WINDOW_SIZE, WINDOW_SIZE)
        observation[1:, 1] = depths[np.asarray(elements)[inverse]].reshape(count, WINDOW_SIZE, WINDOW_SIZE)
        valid = np.concatenate(([True], depths < self.settings["max_depth"]))
        error = self.solution.ComputeL2Error(self.exact)
        return {"observation": observation, "valid_actions": valid, "error": error}
