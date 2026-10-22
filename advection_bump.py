"""Periodic advection of Gaussian bumps, compatible with FEMWorkers.

Use ``FEMWorkers(AdvectionBump, num_workers=4)`` for independent CPU episodes,
or ``train(case="advection")`` with the shared training function.
The policy sees only the numerical solution and mesh depth. ``state['error']``
compares that solution with the transported analytic initial condition.
"""

from importlib.metadata import version

import mfem.ser as mfem
import numpy as np


NX = NY = 8
ORDER = 2
DOMAIN_SIZE = 1.0
VELOCITY = (1.0, 0.0)
REFINEMENT_BUDGET = 20
MAX_DEPTH = 2
ACTION_TIME = 0.1  # Ten policy steps per traversal of the unit domain.
CFL = 0.1
ERROR_QUADRATURE_ORDER = 20

BUMP_COUNT = (1, 4)
CENTER_RANGE = (0.3, 0.7)
WIDTH_RANGE = (0.005, 0.05)

ELEMENT_SAMPLES = 16
CONTEXT_SAMPLES = 4
WINDOW_SIZE = ELEMENT_SAMPLES + 2 * CONTEXT_SAMPLES


class BumpFunction(mfem.PyCoefficient):
    """Exact transport of the periodic extension of the paper's Gaussian sum.

    The Gaussian is defined on [0, 1)^2 and extended by wrapping coordinates.
    Nonzero Gaussian tails can give a jump at the periodic seam; this convention
    retains the stated initial function rather than adding Gaussian images.
    """

    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.time = 0.0

    def values(self, points):
        points = (np.asarray(points) - self.time * np.asarray(VELOCITY)) % DOMAIN_SIZE
        delta = points[..., None, :] - self.parameters[:, :2]
        return np.exp(-(delta ** 2).sum(axis=-1) / self.parameters[:, 2]).sum(axis=-1)

    def EvalValue(self, x):
        return float(self.values(x))


class AdvectionOperator(mfem.PyTimeDependentOperator):
    """DG semidiscrete system: M du/dt = K u, with upwind fluxes."""

    def __init__(self, space):
        super().__init__(space.GetVSize())
        self.velocity_vector = mfem.Vector(list(VELOCITY))
        self.velocity = mfem.VectorConstantCoefficient(self.velocity_vector)
        self.mass = mfem.BilinearForm(space)
        self.mass.AddDomainIntegrator(mfem.MassIntegrator())
        self.mass.Assemble()
        self.mass.Finalize()
        self.transport = mfem.BilinearForm(space)
        self.transport.AddDomainIntegrator(mfem.ConvectionIntegrator(self.velocity, -1.0))
        self.transport.AddInteriorFaceIntegrator(
            mfem.TransposeIntegrator(mfem.DGTraceIntegrator(self.velocity, 1.0, -0.5)))
        self.transport.Assemble(0)
        self.transport.Finalize(0)
        # Periodic faces are interior faces: no physical boundary forcing.
        self.rhs = mfem.Vector(space.GetVSize())
        self.preconditioner = mfem.DSmoother()
        self.solver = mfem.CGSolver()
        self.solver.SetPreconditioner(self.preconditioner)
        self.solver.SetOperator(self.mass.SpMat())
        self.solver.iterative_mode = False
        self.solver.SetRelTol(1e-12)
        self.solver.SetAbsTol(0.0)
        self.solver.SetMaxIter(100)
        self.solver.SetPrintLevel(0)

    def Mult(self, x, y):
        self.transport.SpMat().Mult(x, self.rhs)
        self.solver.Mult(self.rhs, y)
        if not self.solver.GetConverged():
            raise RuntimeError("Advection mass solve did not converge")


class AdvectionBump:
    """Same settings/reset/step contract as StaticBump, usable in spawn workers.

    Action 0 advances time without refinement. Action i+1 refines element i,
    transfers the existing numerical solution, then advances by ACTION_TIME.
    reset() alone projects the analytic function; step() never reprojects it.
    observe=False skips image sampling and returns observation=None for baselines.
    """

    def __init__(self, observe=True):
        self.observe = observe  # Baseline policies can omit unused image observations.
        self.settings = {
            "nx": NX, "ny": NY, "order": ORDER, "domain_size": DOMAIN_SIZE,
            "budget": REFINEMENT_BUDGET, "max_depth": MAX_DEPTH,
            "velocity": VELOCITY, "action_time": ACTION_TIME, "cfl": CFL,
            "bump_count": BUMP_COUNT, "center_range": CENTER_RANGE,
            "width_range": WIDTH_RANGE, "basis": "discontinuous Bernstein",
            "time_integrator": "RK4", "boundary": "periodic",
            "error_quadrature_order": ERROR_QUADRATURE_ORDER,
            "exact_solution": "wrapped initial Gaussian sum",
            "element_samples": ELEMENT_SAMPLES, "context_samples": CONTEXT_SAMPLES,
            "observation_boundary": "periodic", "observation_sampling": "pixel centers",
            "mfem_version": version("mfem"),
        }
        self.mesh = self.fec = self.fes = self.solution = None
        self.operator = self.ode = None
        nodes, weights = np.polynomial.legendre.leggauss((ERROR_QUADRATURE_ORDER + 2) // 2)
        points = (nodes + 1.0) / 2.0
        xx, yy = np.meshgrid(points, points, indexing="ij")
        self.quadrature_points = np.stack((xx.ravel(), yy.ravel()), axis=-1)
        self.gauss_weights = np.outer(weights, weights).ravel() / 4.0

    def reset(self, seed=0, bumps=None):
        rng = np.random.default_rng(seed)
        if bumps is None:
            count = rng.integers(BUMP_COUNT[0], BUMP_COUNT[1] + 1)
            bumps = np.column_stack((rng.uniform(*CENTER_RANGE, (count, 2)),
                                     rng.uniform(*WIDTH_RANGE, count)))
        self.bumps = np.array(bumps, dtype=np.float64, copy=True)
        self.exact = BumpFunction(self.bumps)
        # Release dependents first: MFEM objects own native references to the mesh.
        self.ode = self.operator = self.solution = self.fes = self.fec = self.mesh = None
        base = mfem.Mesh.MakeCartesian2D(NX, NY, mfem.Element.QUADRILATERAL,
                                         True, DOMAIN_SIZE, DOMAIN_SIZE)
        shifts = mfem.vector_Vector([mfem.Vector([DOMAIN_SIZE, 0.0]),
                                     mfem.Vector([0.0, DOMAIN_SIZE])])
        self.mesh = mfem.Mesh.MakePeriodic(base, base.CreatePeriodicVertexMapping(shifts))
        self.mesh.EnsureNCMesh()
        self.fec = mfem.L2_FECollection(self.settings["order"], 2, mfem.BasisType.Positive)
        self.fes = mfem.FiniteElementSpace(self.mesh, self.fec)
        self.solution = mfem.GridFunction(self.fes)
        self.solution.ProjectCoefficient(self.exact)
        self.time = 0.0
        self.steps = 0
        self._update_mesh()
        self.state = self._state()
        return self.state

    def _update_mesh(self):
        """Rebuild the PDE operator and observation coordinates after refinement."""
        self.ode = self.operator = None
        self.operator = AdvectionOperator(self.fes)
        self.ode = mfem.RK4Solver()
        self.ode.Init(self.operator)
        count = self.mesh.GetNE()
        # Periodic meshes identify vertices; use physical element transformations.
        ip = mfem.IntegrationPoint()
        lower, upper = np.empty((count, 2)), np.empty((count, 2))
        for i in range(count):
            transform = self.mesh.GetElementTransformation(i)
            ip.Set2(0.0, 0.0)
            lower[i] = transform.Transform(ip)
            ip.Set2(1.0, 1.0)
            upper[i] = transform.Transform(ip)
        self.lower, self.upper = lower, upper
        self.depths = np.array([self.mesh.ncmesh.GetElementDepth(i) for i in range(count)])
        # Conservative DG step based on the smallest current cell and polynomial order.
        rate = (np.abs(VELOCITY) / (upper - lower)).sum(axis=1).max()
        self.dt = CFL / ((2 * self.settings["order"] + 1) * rate)
        if not self.observe:
            return
        positions = (np.arange(WINDOW_SIZE) - CONTEXT_SAMPLES + 0.5) / ELEMENT_SAMPLES
        xx, yy = np.meshgrid(positions, positions, indexing="xy")
        local = np.stack((xx, yy), axis=-1)
        points = lower[:, None, None, :] + (upper - lower)[:, None, None, :] * local
        points = points.reshape(-1, 2) % DOMAIN_SIZE
        unique, self.sample_inverse = np.unique(points, axis=0, return_inverse=True)
        found, self.sample_elements, self.sample_points = self.mesh.FindPoints(unique, warn=False)
        if found != len(unique):
            raise RuntimeError("MFEM could not locate periodic observation sample points")

    def step(self, action):
        if (not 0 <= action < len(self.state["valid_actions"]) or
                not self.state["valid_actions"][action]):
            raise ValueError("Invalid action for the current mesh")
        if action:
            self.ode = self.operator = None
            self.mesh.GeneralRefinement(mfem.intArray([int(action) - 1]), 1, 0)
            self.fes.Update()
            self.solution.Update()  # Transfer u_h, never replace it with the true solution.
            self._update_mesh()
        target = (self.steps + 1) * ACTION_TIME
        substeps = int(np.ceil((target - self.time) / self.dt))
        dt = (target - self.time) / substeps
        for _ in range(substeps):
            self.time, _ = self.ode.Step(self.solution, self.time, dt)
        self.time = target
        self.exact.time = self.time
        self.steps += 1
        self.state = self._state()
        return self.state, self.steps == self.settings["budget"]

    def element_errors(self):
        """Integrate (u_h - u_true)^2, splitting cells at the transported seams."""
        seam = (self.time * np.asarray(VELOCITY)) % DOMAIN_SIZE
        errors = np.zeros(self.mesh.GetNE())
        values = mfem.Vector()
        for element, (lower, upper) in enumerate(zip(self.lower, self.upper)):
            cuts = [[lower[d], upper[d]] for d in range(2)]
            for d in range(2):
                if lower[d] < seam[d] < upper[d]:
                    cuts[d].insert(1, seam[d])
            for x0, x1 in zip(cuts[0][:-1], cuts[0][1:]):
                for y0, y1 in zip(cuts[1][:-1], cuts[1][1:]):
                    size = np.array([x1 - x0, y1 - y0])
                    points = np.array([x0, y0]) + size * self.quadrature_points
                    reference = (points - lower) / (upper - lower)
                    rule = mfem.IntegrationRule(len(points))
                    for i, point in enumerate(reference):
                        rule.IntPoint(i).Set2(float(point[0]), float(point[1]))
                    self.solution.GetValues(element, rule, values)
                    difference = values.GetDataArray() - self.exact.values(points)
                    errors[element] += np.dot(self.gauss_weights, difference ** 2) * size.prod()
        return np.sqrt(errors)

    def _state(self):
        count = self.mesh.GetNE()
        observation = None
        if self.observe:
            values = np.fromiter((self.solution.GetValue(int(e), self.sample_points[i])
                                  for i, e in enumerate(self.sample_elements)),
                                 dtype=np.float64, count=len(self.sample_elements))
            observation = np.zeros((count + 1, 2, WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
            observation[1:, 0] = values[self.sample_inverse].reshape(count, WINDOW_SIZE, WINDOW_SIZE)
            sampled_depth = self.depths[np.asarray(self.sample_elements)[self.sample_inverse]]
            observation[1:, 1] = sampled_depth.reshape(count, WINDOW_SIZE, WINDOW_SIZE)
        error = float(np.linalg.norm(self.element_errors()))
        if not np.isfinite(error) or (observation is not None and not np.isfinite(observation).all()):
            raise RuntimeError("Advection produced a non-finite solution or error")
        return {"observation": observation,
                "valid_actions": np.concatenate(([True], self.depths < MAX_DEPTH)),
                "error": float(error), "time": self.time}
