# Advection bumps implementation

1. Confirm MFEM runtime, paper contract, and worker API. Status: complete.
2. Add a single readable advection_bump.py exposing settings/reset/step. Status: complete.
3. Validate transport, error, refinement transfer, and serial/multiprocess equivalence. Status: complete.

Scope: no dataset splits; preserve existing static source and training settings.
Tests and temporary scripts go in tem/advection/.

## Shared training architecture (2026-10-22)

1. Inspect both case interfaces and existing training semantics. Status: complete.
2. Move fixed policy rules into model.py; keep selection and the single shared loop together in train.py (user clarified: no trainer.py). Status: complete.
3. Verify unchanged policy math and smoke-train both cases through the shared entry. Status: complete.

Preserve all existing optimizer, reward, exploration, discount, batch and FEM settings.

## Simplify all project code (2026-10-22)

1. Review all seven project Python files and save the current sources under tem/simplify/before. Status: complete.
2. Simplify worker plumbing, remove routine defensive checks, share duplicated evaluation and error integration code; use human-readable output names. Status: complete.
3. Compare numerical states/indicators with saved sources, smoke-train both cases and verify worker failures and evaluation. Status: complete.

## Errors
- The short transfer-plot smoke initially used spawn workers, which reimported the source's 10/100 budgets rather than inheriting the parent's temporary 1/2 budgets. Stopped only that smoke process and its two children; reran the short smoke serially. Formal source budgets unchanged.
- A text replacement matched an already-updated parents[1] path in the transfer plot and produced invalid syntax; corrected the path immediately and recompiled all project files. Preserved its current grouped plot layout.
- Architecture checks: sandbox WSL inventory returned E_ACCESSDENIED; escalated access found the existing /home/yjc/miniconda3/envs/pinn/bin/python runtime.
- Windows pinn Python has no MFEM. Inspect existing WSL runtime.
- WSL sandbox read was denied earlier; approved read-only WSL inspection works.


- Numerical check found order-12 L2 quadrature differs from order-24 by 5.5e-6 relatively on a narrow bump. Increased error quadrature to order 20.
- Explicit IntegrationRuleArray is not accepted by ComputeL2Error in this binding; a Python list of rules is supported.
