# Findings

- Original run history ends at episode 409; no checkpoint exists in that run.
- train.py saves a checkpoint only after all episodes.
- StaticBump uses serial MFEM, repeated reset/refinement/point sampling; batch accumulates policy graphs.
- Existing source files are user work (untracked); preserve unrelated changes.
- WSL pinn: Python 3.12.0, torch 2.14.0 (CUDA 13.0), numpy 2.5.3, mfem 4.10.0.
- Baseline uses batch 64, seed 0, CUDA, Adam 1e-4, unchanged model and FEM parameters; bounded to 512 episodes to cross original failure near 409.
- Added separate diagnose_crash.py; traces phases/actions, /proc RSS and CUDA allocator every 5 seconds; snapshots sources in each diagnostic directory.
- Early baseline RSS about 1.5 GiB, WSL available memory about 13 GiB. No early exhaustion evidence. gdb/lldb are not installed; core_pattern routes to /wsl-capture-crash.
- First diagnostic failed at episode 97 step 1 in _state / np.fromiter / GetValue with ValueError: too many values to unpack (expected 2). This differs from original abort; relation not yet established. First 88 episode outputs matched original exactly when checked before failure.
- Just before failure RSS 1,772,496 kB and CUDA allocated 197,004,800 bytes, no resource exhaustion observed.
- FEM-only replay of episode 97 step 1 repeated 100 times passed (15.2 seconds). Second full-training attempt passed episode 100, so the first diagnostic exception is not deterministic for that seed/action.
- FEM-only full recorded action sequence from episodes 1-97 also passed (109.9 seconds). Second training run crossed episode 300; RSS ~1.7 GiB stable after warmup, CUDA reserved ~400 MiB.
- Second training run matched original history for every one of the 409 available episodes (final_error and policy_loss), then crossed episode 434 without failure.
- Fixed-grid CPU-only GetValue/fromiter stress passed 40,000,000 sample evaluations (97.3 seconds), no reset/refinement or training involved.
- baseline_repeat completed all 512 episodes in 618.3 seconds and saved checkpoint/history/plot. No source fix applied; original abort remains unreproduced.
- WSL core_pattern points at /wsl-capture-crash, but that executable is absent, and standard crash directories contain no core files.
- Planning files were moved externally into docs/ during investigation; continued at their new location. SHA256 of current train.py/static_bump.py matches baseline snapshots.
- Repeated full _state reconstruction on a fixed mesh passed 3000 iterations, including repeated FindPoints and integration-point allocation/release (about 220 seconds). Its old log's 'calls' label is repeat*40000, not a valid measured scalar-call count for this mode; report only state iterations.
