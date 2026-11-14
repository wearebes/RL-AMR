# Mesquite PR-CG for the original Kratos FSI2 case

All added code lives here. The author's `kratos_fsi2/Examples/.../fsi_turek_FSI2/source`
entry, physics, mesh, materials, time step, coupling and BIN output remain unchanged.
Mesquite 2.3.0 and the installed Kratos package are not patched.

## Four-case smoke test

Run in the WSL `pinn` environment:

```bash
bash Smoothing/fsi_turek_FSI2/job.sh
```

Default end time is 0.05 s (25 steps at the unchanged 0.002 s time step).
The cases run sequentially under `Smoothing/data/smoke_005/`:

| Directory | Mesh motion | Smoothing |
|---|---|---|
| similarity | structural_similarity | disabled |
| laplacian | laplacian | disabled |
| laplacian_prcg | laplacian | PR-CG |
| laplacian_nn | laplacian | VASONet (CPU float64) |

All other physical and coupling settings are shared. Existing output directories
are never overwritten. A failed case is reported; the remaining cases are still
attempted and the script returns nonzero if any case fails.

Optional arguments specify end time and a new output root:

```bash
bash Smoothing/fsi_turek_FSI2/job.sh 0.5 /path/to/new/results
```

`run_fsi2.py` also accepts `--mesh-motion`, `--method`, `--end-time`, and
`--output` for a single case. It records the effective parameters and method in
`run_settings.json`. Configuration files and author sources are not rewritten.
The central ProjectParameters.json retains its 0.5 s end time; job.sh overrides
it in memory. Mesh/material paths resolve to the author's original files.
OpenMP/BLAS settings are inherited without overrides. `PYTHON` selects the Python executable.
The existing native bridge is used without rebuilding.

## Integration

`model.py` defines the dual MLP used by the saved checkpoint.
`mesh_smoothing.py` contains the smoothing methods, native bindings and Kratos
mesh integration. Unexpected Python errors stop the run; optimizer-reported
rejections are recorded and discarded.

`run_fsi2.py` extends the original `FsiAnalysis` and reuses the author's entry factory.
`mesh_smoothing.py` maps actual node IDs and wraps the existing mesh solver's
`MoveMesh`. Structural Similarity resets coordinates before solving, so the input
mesh is refreshed from `X0 + MESH_DISPLACEMENT[0]`.

The C++ bridge calls native `IdealWeightMeanRatio` / `QualityAssessor` for reported
quality and candidate selection. Optimization now uses the regularized mean ratio
and LogSumExp from Gouveia et al., Eqs. 9–10:

```
S = [b-a, (2*(c-a)-(b-a))/sqrt(3)]
d = det(S), r = sqrt(d*d + 4*delta*delta), h = (d+r)/2
q_hat = 2*h / ||S||_F^2
F = log(sum(exp(-alpha*q_hat))) / alpha
```

`RegularizedMeanRatio` and `SoftMinObjective` provide analytic gradients to the
native `ConjugateGradient`. LogSumExp is shifted by the minimum q_hat for stable
evaluation; the negative-determinant branch of h is rationalized. No new optimizer
or line search is introduced. Native one-ring patches and termination remain.
`regularization_delta=2.5e-5` and `softmin_alpha=50` are visible settings from the
paper, not values calibrated for FSI2. S uses physical XY coordinates: no hidden
patch normalization or scale-dependent change of delta is made. Delta has the
same scale as det(S); q_hat is not clamped to [0,1] and may exceed 1. Keep this
scale dependence in mind before authorizing a run.

Candidate selection is exactly:

```
element q < quality_threshold
  -> union of its vertex Node IDs
  -> ascending Node ID, excluding boundary/interface/constrained nodes
  -> first 100 free candidates
  -> native one-ring patch optimization, at most 10 inner iterations per patch
```

Node references are sorted once at initialization, and element connectivity and
displacement writeback use that same ID-to-vertex map. The solver's node container
is not reordered. Repeated element incidences set one candidate flag, so each
node is selected at most once per call. No worst-neighbor score or quality-priority
sort is used. Boundaries and unselected nodes stay fixed. Budget 100, inner limit
10, outer limit 1 and the existing threshold 0.50 are in the configuration JSON;
there is no separate Python copy of default parameter values.
The quality threshold is a candidate-selection condition, not a convergence
guarantee. Soft-min improvement does not ensure that ordinary min_q increases.

Only accepted current-step displacement is written back. Original `MoveMesh` runs
once and Kratos updates coordinates and Bossak mesh velocity/acceleration; history
slots are untouched. Failed native work is discarded. An invalid original mesh
is an error. This implementation supports the fixed-topology 2D OpenMP case, not MPI.
The bridge is called synchronously outside Kratos parallel regions. This native
PR-CG patch traversal is serial; increasing OMP threads speeds only parallel work.

## Outputs

Each experiment contains original fluid/structure `.post.bin`, a GiD `.post.lst`,
`solver.log` and a single additional `mesh_smoothing.csv`. Both configuration JSON
files remain in the code directory. No Python/C++ files, configuration copies,
shared library, copied mesh or `__pycache__` belong in result directories.

CSV `record_type` is `configuration`, `step`, `rejection` or `summary`:

- `step,time,min_J,min_q,mean_q`: entire final fluid mesh after original output.
- `prcg_calls,prcg_seconds`: all real native optimization queue calls in that
  physical step, including failures. Selection and measurement are excluded.
- `smoothing_seconds`: extraction, native checks/optimization, writeback and final
  metric measurement; excludes original Structural Similarity and MoveMesh.
- `rejected_calls,reason`: rejected work and immediate event reasons, without
  additional terminal printing.
- `total_seconds,completed`: final run wall time (including BIN and CSV I/O) and
  whether the configured end time was reached. This flag does not certify convergence.

`min_J` is twice native `MsqMeshEntity::compute_signed_area()` for linear XY
triangles: signed geometric determinant in m², not a normalized area ratio.
Positive orientation is defined by `PlanarDomain::XY`. Its magnitude depends on
element size. The native mean ratio supplies shape quality; all-fixed triangles
are included. CSV is recorded every physical step; original BIN remains every 0.01 s.

The new objective and compact result structure require bridge ABI 5. Unused
before/after quality-return fields have been removed; the result contains status,
call/selected/moved counts and optimization seconds. Coordinates remain double
buffers and current displacement is written through the original Kratos API.
The ABI-5 bridge has been rebuilt in WSL and loaded successfully using
`/home/yjc/miniconda3/envs/pinn/bin/python`. Only library loading/version metadata
were checked: no optimization, numerical tests or FSI experiment was run for this
revision. Earlier verification applies to the old objective, not the regularized
LogSumExp revision. No full-run or accuracy claim is made.

## Smart Laplacian and VASONet

VASONet requires PyTorch; model_path is relative to this code directory.
Smart Laplacian remains available through the single-case `--method smart_laplacian`
option but is not one of the four comparison cases.

Candidate selection shares the PR-CG threshold, ascending node-ID order and
100-node budget; boundary/interface/constrained nodes remain fixed.
Smart Laplacian proposes the neighbor centroid. VASONet uses the exact notebook
architecture and saved `Smoothing/MLP.pt`, CCW topological rings, bounding-box
normalization and zero padding. Unsupported valences outside 3–9 are skipped.
Inference runs on CPU in float64, including normalized inputs and model operations.
The saved checkpoint is unchanged; its loaded parameters are represented in float64.
Both accept a proposal only if every incident triangle stays positive and the
ordinary local minimum mean ratio strictly improves. VASONet therefore includes
an explicit acceptance guard beyond the notebook's raw prediction.

One proposal per selected node per outer sweep; PR-CG's inner iteration limit
does not apply to these two methods. CSV uses `optimizer_calls/seconds` for
whole smoothing calls, including selection and validity checks, so these timings
are not directly equivalent to the original native PR-CG queue timing.
Model loading is outside per-call timing. OpenMP settings are inherited.


WSL pinn: checkpoint loading and float64 CPU forward pass verified. Full FSI validation is pending.
