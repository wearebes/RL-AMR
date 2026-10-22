# Training native crash investigation

Goal: reproduce and localize the stack-smashing crash in WSL pinn without changing the training protocol.

1. Inspect environment and existing artifacts — complete.
2. Reproduce with fault traces and memory monitoring — in progress.
3. Isolate the native operation and apply a narrow fix if supported — pending.
4. Validate and report evidence and limitations — pending.

## Errors
- Normal sandbox WSL invocation returned Wsl/E_ACCESSDENIED; retry through authorized execution.
- psutil is absent; monitor /proc directly without installing dependencies.
- Planning files moved externally to docs/; a patch to the previous root path failed, then resolved the new paths without recreating root files.
