# Progress

2026-10-22: Reviewed all seven project Python files for simplicity. Removed routine input guards, extra worker bookkeeping and duplicate evaluation code. Static and advection now expose element_errors(); the transfer plot reuses the existing plot.py helpers. Output names use bumps_IPN/advection_IPN. User preference saved as an explicitly requested memory update note.

2026-10-22: Full pre/post trajectory and indicator comparisons pass for both cases, including exact observation/mask agreement and error agreement within floating-point roundoff. Policy sampling/loss/gradient checks, serial/two-process comparisons and injected worker failures pass. Final two-worker smoke training completes for bumps (2 episodes) and advection (3 episodes, batch size 2), producing named directories with checkpoint/history/plots. Static evaluation CLI with two spawned workers passes. Transfer evaluation uses a short serial prefix for smoke only; source budgets remain 10/100.

2026-10-22: Architecture cleanup: fixed optimizer/reward/exploration/discount rules now live in model.py. Per user clarification, selection, shared training loop, FEM scheduling and saving all remain in train.py; removed the newly introduced trainer.py. Existing IPN forward and state_dict names are unchanged. Added --case static/advection and explicit short-run CLI options.

2026-10-22: Verification complete. Both cases pass short CPU training through the final CLI with two FEM workers; advection also covers the partial final batch. Checked action sampling, loss and gradients exactly against original formulas, Adam settings, original checkpoint loading, saved source metadata, CSV reward telescoping and plot creation. Test helper and outputs: tem/architecture/. Formal training was not started.

2026-10-22: Inspected static source, training, policy, worker interfaces. Reading the supplied paper to establish the advection benchmark contract before adding code.

2026-10-22: Implementing standalone advection environment under clarified scope. Existing worker protocol supports new source classes without modification.

Implementation smoke: seed=0 initial error 0.0365177706878587, after refine t=0.1 error 0.03718346683801268, 67 elements; no-refine also advances time. Preparing independent numerical and multiprocessing validation.

2026-10-22: Finished advection_bump.py. Verified 20-step episodes, maximum-depth masking, repeated resets, transfer of an injected constant, periodic transport/mass conservation, quadrature convergence, grid convergence, serial reference and two spawn workers. Corrected seam-integration bias. Training/checkpoint/inference smoke passed; formal training not started. All temporary tests and outputs are under tem/advection/.
