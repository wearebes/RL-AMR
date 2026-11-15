#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
# Four formal FSI2 cases, sequentially sharing one explicit CPU allocation.
set -euo pipefail
# Experiment settings: edit here before submission.
# Slurm: edit --cpus-per-task above; plain server: edit the fallback 128 below.
threads="${SLURM_CPUS_PER_TASK:-128}"
end_time="${1:-20}"
python_bin="${PYTHON:-python}"
repo_root="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
output_root="${2:-$repo_root/Smoothing/data/fsi2_20s}"
case_dir="$repo_root/Smoothing/fsi_turek_FSI2"
if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Thread count must be a positive integer.\n' >&2
    exit 1
fi
export OMP_NUM_THREADS="$threads"
export OMP_DYNAMIC=FALSE
names=(similarity laplacian laplacian_prcg laplacian_nn)
motions=(structural_similarity laplacian laplacian laplacian)
methods=(o o prcg vasonet)
"$python_bin" -c 'import numpy, torch, KratosMultiphysics'
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
    if "$python_bin" -u "$case_dir/run_fsi2.py" \
        --mesh-motion "${motions[$i]}" --method "${methods[$i]}" \
        --end-time "$end_time" --output "$run_dir" > "$run_dir/solver.log" 2>&1; then
        printf 'Finished %s\n' "${names[$i]}"
    else
        printf 'Failed %s; see %s/solver.log\n' "${names[$i]}" "$run_dir" >&2
        failed=1
    fi
done
exit "$failed"
