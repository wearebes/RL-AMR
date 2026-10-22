# Findings

- Project-wide simplification: worker command dispatch uses reset/step directly, redundant command/seed bookkeeping and routine argument guards removed. Native-object release order, solver convergence, point-location checks and bounded worker shutdown remain.
- Evaluation shares load_policy/indicator_action/collect in the existing plot.py; no extra helper module. Both case classes expose element_errors(), and advection's seam integration is no longer duplicated in the transfer plot.
- Default output names are bumps_IPN/advection_IPN. The training API now selects case by name: train(case="advection"). Setup runs once before the training loop.
- Saved pre-simplification sources in tem/simplify/before. Full static/advection trajectories match observations/masks exactly and errors within rtol=1e-13; both ZZ/TrueError actions match. Shared error arrays match the former transfer implementation exactly. Policy action/loss/gradients are unchanged; serial/two-process results and Python-error/native-exit reporting pass.

- Architecture inspection (2026-10-22): StaticBump and AdvectionBump already share the FEMWorkers contract. Case selection needs only a class mapping; the training loop needs no case-specific branches.
- Keep IPN.forward and state_dict unchanged so existing evaluation scripts and checkpoints remain usable.
- Final architecture keeps only model.py and train.py for policy/training responsibilities; no separate trainer module. Both cases use the same loop without case branches.
- Verified exact action/log-probability equivalence at episodes 0/100/500/700 and exact loss/gradient equivalence against the former inline formulas. Adam settings and old checkpoint loading pass.
- Final CPU integration: static 2 episodes / 2 workers; advection 3 episodes / batch size 2 / 2 workers (including the partial final batch). Both save reloadable checkpoints, correct reward totals, CSV and plots. These checks do not establish trained policy quality or CUDA execution.

- Current static_bump.py projects an analytic Gaussian sum into an MFEM H1 space and calls ComputeL2Error(exact).
- train.py uses StaticBump, online episode seeds, and normalized error-reduction rewards; no validation entry point exists.
- The existing IPN chooses a refinement action; it does not predict a solution or its error.
- Current static distribution differs from the supplied advection table. Do not change static settings to match it.
- The checkout contains pre-existing modified and deleted files; preserve them.

- User requests one advection_bump.py, compatible with FEMWorkers settings/reset(seed)/step(action), no dataset partitioning.
- Paper: DG second-order Bernstein, periodic unit square, velocity (1,0), initial 8x8 mesh, max depth 2, 20 RL steps, physical time 0.1 per RL step.
- Initial condition is projected once. Refinement transfers existing numerical solution before PDE evolution; true solution is used only for scoring.
- MFEM official PyMFEM example 9 provides standard weak DG advection volume and upwind face forms.

- Existing WSL pinn has working MFEM. Verified periodic mesh creation, DG refinement transfer, and physical element transforms.
- Implemented initial single-file DG/RK4 environment. First smoke test passes reset, refine, no-refine, and time advance.
- MFEM Vector constructor requires list rather than tuple in this runtime; corrected and retained coefficient vector lifetime.
- Explicit periodic convention: wrap the stated Gaussian sum on the fundamental square; nonzero tails can create seam jumps (paper does not specify a smoothing convention).

- Final implementation uses upwind DG advection and RK4 with a conservative cell-size/order CFL bound. These are explicit implementation choices; the paper does not fully specify its time integrator.
- Broad bumps exposed 2-4% error-quadrature bias when integrating across the moving periodic seam. Fixed by splitting each physical cell at the transported seams before Gauss integration.
- Seam-aware order-20 and order-40 integration agree within 1.4e-17 in the checked broad-bump cases.
- Validation passed: mass drift 1.04e-17; centroid displacement 0.099976 for target 0.1; errors on 8x8 and 16x16 grids 0.01995917 and 0.00796771; two workers match serial observations/masks/errors exactly.
- CPU training integration test: two episodes with two workers, checkpoint saved and reloaded into IPN, followed by two complete inference episodes. This is an integration smoke test, not a policy-performance result.
