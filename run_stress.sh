#!/usr/bin/env bash
# run_stress.sh — runtime stress/endurance for hrx-on-hip, on the HRX backend.
#
# Swaps HRX in, runs the @stress tests (no native A/B — these assert stability
# invariants: no crash / NaN / leak, results stay correct under load), then
# restores native ROCm. Scale the workloads with HRX_STRESS_SCALE (default 1).
#
# Inputs (env): same as run_ab.sh — HRX_VENV, HRX_BUILD_DIR (or HRX_BINDING),
#   USE_HRX / USE_ROCM, HRX_ALLOW_NO_GPU, PYTEST_ARGS, HRX_STRESS_SCALE.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${HRX_VENV:-${VIRTUAL_ENV:-}}"
[ -n "$VENV" ] || { echo "FAIL: no venv — set HRX_VENV=/path/to/venv (or activate one)." >&2; exit 4; }
USE_HRX="${USE_HRX:-$HERE/scripts/use_hrx.sh}"
USE_ROCM="${USE_ROCM:-$HERE/scripts/use_rocm.sh}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HRX_GPU_DRIVER=amdgpu AMD_SERIALIZE_KERNEL=1
export HRX_BUILD_DIR="${HRX_BUILD_DIR:-}"
unset LD_PRELOAD || true
cd "$HERE"

CORE="$(python -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))' 2>/dev/null || true)"
export HRX_CORE_LIB="${HRX_CORE_LIB:-$CORE}"

ensure_native() {
  if [ -e "$HRX_CORE_LIB/libamdhip64.so.7.orig" ]; then
    "$USE_ROCM" | sed 's/^/  /'
  else
    echo "  (native ROCm already live — no .orig backup yet)"
  fi
}

# GPU presence guard — stress on no GPU verifies nothing.
ensure_native
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FAIL: no GPU visible to torch — stress verifies nothing."
  [ "${HRX_ALLOW_NO_GPU:-0}" = "1" ] && { echo "  HRX_ALLOW_NO_GPU=1 → exiting 0 (nothing verified)."; exit 0; }
  exit 5
fi

echo "============ stress on HRX (HRX_STRESS_SCALE=${HRX_STRESS_SCALE:-1}) ============"
"$USE_HRX" | sed 's/^/  /'
pytest -m stress ${PYTEST_ARGS:--q -ra}
rc=$?

echo "==================== restore native ROCm ============================"
ensure_native

if [ "$rc" -eq 0 ]; then
  echo "RESULT: PASS — hrx-on-hip stable under stress."
else
  echo "RESULT: FAIL — stress run failed (pytest rc=$rc)."
fi
exit "$rc"
