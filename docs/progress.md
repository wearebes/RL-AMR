# Progress

- 2026-10-22: inspected source and original run; user authorized live reproduction and monitoring.
- WSL normal sandbox access failed; requested authorized execution.
- Authorized WSL execution succeeded; baseline_512 running with faulthandler and memory/action traces.
- baseline_512 stopped at episode 97 step 1 with a different Python exception in observation sampling; recorded actions allow FEM-only replay.
- Episode 97 replayed 100 times successfully. Repeating the full first 97 episodes CPU-only; second training run has crossed 200 and matches original printed metrics.
- Built a local diagnostic-only stack-check interceptor (native_crash_trace.so); another bounded run uses it with PYTHONMALLOC=debug. No environment packages installed or training sources changed.
- baseline_repeat completed 512 episodes. Started repeated _state allocation/release stress plus the original unwrapped train entrypoint under native-stack capture (20-minute ceiling), to minimize instrumentation effects.
- Baseline sampled maxima: RSS 1747.78 MiB, CUDA allocated 391.65 MiB, CUDA reserved 400 MiB. Native-debug run crossed episode 300; _state rebuild stress crossed 2000 iterations.
