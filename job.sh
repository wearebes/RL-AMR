#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
# Run locally with bash, or submit to Slurm with sbatch; four cases run sequentially.
set -euo pipefail
# Experiment settings: edit here before submission.
# Slurm: edit --cpus-per-task above. Locally, inherit OMP_NUM_THREADS/runtime defaults.
threads="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-}}"
conda_env="${CONDA_ENV:-fsi}"
conda_exe="${CONDA_EXE:-conda}"
end_time="${1:-20}"
repo_root="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
output_root="${2:-$repo_root/Smoothing/data/fsi2_20s}"
case_dir="$repo_root/Smoothing/fsi_turek_FSI2"
if [[ -n "$threads" && ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Thread count must be a positive integer.\n' >&2
    exit 1
fi
if [[ -n "$threads" ]]; then
    export OMP_NUM_THREADS="$threads"
fi
export OMP_DYNAMIC=FALSE
# Explicit PYTHON bypasses Conda (e.g. a cluster module or an absolute env path).
if [[ -n "${PYTHON:-}" ]]; then
    python_cmd=("$PYTHON")
else
    if ! command -v "$conda_exe" >/dev/null 2>&1; then
        printf 'Conda not found. Set CONDA_EXE to its executable, or PYTHON to your environment Python.\n' >&2
        exit 1
    fi
    python_cmd=("$conda_exe" run --no-capture-output -n "$conda_env" python)
fi
names=(similarity laplacian laplacian_prcg laplacian_nn)
motions=(structural_similarity laplacian laplacian laplacian)
methods=(o o prcg vasonet)
"${python_cmd[@]}" -c 'import sys, numpy, torch, KratosMultiphysics as K; print("Python:", sys.executable); print("Kratos threads:", K.ParallelUtilities.GetNumThreads())'
for name in "${names[@]}"; do
    if [[ -e "$output_root/$name" ]]; then
        printf 'Output already exists: %s\n' "$output_root/$name" >&2
        exit 1
    fi
done
failed=0
for i in "${!names[@]}"; do
    run_dir="$output_root/${names[$i]}"
    mkdir -p "$run_dir"
    printf 'Starting %s (end_time=%s)\n' "${names[$i]}" "$end_time"
    if "${python_cmd[@]}" -u "$case_dir/run_fsi2.py" \
        --mesh-motion "${motions[$i]}" --method "${methods[$i]}" \
        --end-time "$end_time" --output "$run_dir" > "$run_dir/solver.log" 2>&1; then
        printf 'Finished %s\n' "${names[$i]}"
    else
        printf 'Failed %s; see %s/solver.log\n' "${names[$i]}" "$run_dir" >&2
        failed=1
    fi
done
exit "$failed"
