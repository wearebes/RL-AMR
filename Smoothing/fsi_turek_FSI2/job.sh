#!/usr/bin/env bash
# Four controlled FSI2 cases; inherit the current OpenMP environment.
set -euo pipefail
case_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$case_dir" rev-parse --show-toplevel)"
python_bin="${PYTHON:-python}"
end_time="${1:-0.05}"
output_root="${2:-$repo_root/Smoothing/data/smoke_005}"
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
