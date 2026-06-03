#!/usr/bin/env bash
# run_stress.sh — runtime stress/endurance for hrx-on-hip, on the HRX backend.
#
# Swaps HRX in, runs the @stress tests (no native A/B — these assert stability
# invariants: no crash / NaN / leak, results stay correct under load), then
# restores native ROCm. Scale the workloads with HRX_STRESS_SCALE (default 1).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${HRX_VENV:-/home/zjgar/code/fpp/hrx-venv}"
USE_HRX="${USE_HRX:-/home/zjgar/code/fpp/use_hrx.sh}"
USE_ROCM="${USE_ROCM:-/home/zjgar/code/fpp/use_rocm.sh}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HRX_GPU_DRIVER=amdgpu AMD_SERIALIZE_KERNEL=1
unset LD_PRELOAD || true
cd "$HERE"

echo "============ stress on HRX (HRX_STRESS_SCALE=${HRX_STRESS_SCALE:-1}) ============"
"$USE_HRX" | sed 's/^/  /'
pytest -m stress ${PYTEST_ARGS:--q -ra}
rc=$?

echo "==================== restore native ROCm ============================"
"$USE_ROCM" | sed 's/^/  /'

if [ "$rc" -eq 0 ]; then
  echo "RESULT: PASS — hrx-on-hip stable under stress."
else
  echo "RESULT: FAIL — stress run failed (pytest rc=$rc)."
fi
exit "$rc"
