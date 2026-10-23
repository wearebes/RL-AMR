"""Static bumps: MFEM simulation and observations, independent of the policy."""

from importlib.metadata import version
import mfem.ser as mfem
import numpy as np

from FEM.common import (ELEMENT_SAMPLES, CONTEXT_SAMPLES, WINDOW_SIZE,
                        sample_bumps, observation_points, locate_samples, sample_observation)


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
        self.bumps = sample_bumps(seed, self.settings["bump_count"], self.settings["center_range"],
                                  self.settings["width_range"], bumps)
        self.exact = BumpFunction(self.bumps)
        # Release dependent MFEM objects before replacing their mesh.
        self.solution = self.fes = self.fec = self.mesh = None
        self.mesh = mfem.Mesh.MakeCartesian2D(self.settings["nx"], self.settings["ny"],
                                              mfem.Element.QUADRILATERAL, True,
                                              self.settings["domain_size"], self.settings["domain_size"])
        self.mesh.EnsureNCMesh()
        self.fec = mfem.H1_FECollection(self.settings["order"], 2, mfem.BasisType.Positive)
        self.fes = mfem.FiniteElementSpace(self.mesh, self.fec)
        self.solution = mfem.GridFunction(self.fes)
        self.steps = 0
        self._project()
        self.state = self._state()
        return self.state

    def _project(self):
        self.solution.ProjectCoefficient(self.exact)
        self.solution.RestrictConforming()

    def step(self, action):
        if action:
            self.mesh.GeneralRefinement(mfem.intArray([action - 1]), 1, 0)
            self.fes.Update()
            self.solution.Update()
            self._project()
            self.state = self._state()
        self.steps += 1
        return self.state, self.steps == self.settings["budget"]

    def element_errors(self):
        errors = mfem.Vector(self.mesh.GetNE())
        self.solution.ComputeElementL2Errors(self.exact, errors)
        return errors.GetDataArray().copy()

    def _state(self):
        count = self.mesh.GetNE()
        vertices = np.asarray(self.mesh.GetVertexArray())
        corners = np.array([vertices[self.mesh.GetElementVertices(i)] for i in range(count)])
        lower, upper = corners.min(axis=1), corners.max(axis=1)
        depths = np.array([self.mesh.ncmesh.GetElementDepth(i) for i in range(count)])

        # Sample the FEM solution in physical space, including actual neighbors.
        points = np.clip(observation_points(lower, upper), 0.0, self.settings["domain_size"])
        elements, integration_points, inverse = locate_samples(self.mesh, points)
        observation = sample_observation(self.solution, depths, elements, integration_points, inverse)
        valid = np.concatenate(([True], depths < self.settings["max_depth"]))
        error = self.solution.ComputeL2Error(self.exact)
        return {"observation": observation, "valid_actions": valid, "error": error}
