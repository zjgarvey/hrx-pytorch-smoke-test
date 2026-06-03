#!/usr/bin/env bash
# run_ab.sh — "does pytorch work with hrx-on-hip?" via a native-ROCm A/B.
#
#   1. (optional) build the HRX HIP binding.
#   2. RECORD : swap to native ROCm, run the suite (saves golden tensors and
#               confirms the native baseline itself is sane).
#   3. COMPARE: swap to HRX, run the same suite (asserts every result matches the
#               native golden and that nothing crashes).
#   4. restore native ROCm.
#
# Exit 0 iff the native baseline passed AND HRX matched it. Override paths with
# the env vars below (defaults target this machine's layout) for CI.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${HRX_VENV:-/home/zjgar/code/fpp/hrx-venv}"
USE_HRX="${USE_HRX:-/home/zjgar/code/fpp/use_hrx.sh}"
USE_ROCM="${USE_ROCM:-/home/zjgar/code/fpp/use_rocm.sh}"
HRX_BUILD_DIR="${HRX_BUILD_DIR:-/home/zjgar/code/fpp/hrx/build/hrx}"
GOLDEN_DIR="${HRX_AB_GOLDEN_DIR:-$HERE/goldens}"
PYTEST_ARGS="${PYTEST_ARGS:--q -ra}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HRX_GPU_DRIVER=amdgpu AMD_SERIALIZE_KERNEL=1
export HRX_AB_GOLDEN_DIR="$GOLDEN_DIR"
unset LD_PRELOAD || true
cd "$HERE"
rm -rf "$GOLDEN_DIR"

if [ "${HRX_SKIP_BUILD:-0}" != "1" ] && [ -f "$HRX_BUILD_DIR/build.ninja" ]; then
  echo "==================== building HRX binding ==========================="
  cmake --build "$HRX_BUILD_DIR" --target libhrx_src_binding_hip_amdhip64 \
    -j"$(nproc)" || { echo "FAIL: HRX build failed"; exit 3; }
fi

echo "==================== [1/2] RECORD on native ROCm ===================="
"$USE_ROCM" | sed 's/^/  /'
if ! HRX_AB_MODE=record pytest $PYTEST_ARGS; then
  echo "FAIL: native ROCm baseline did not pass — the test/env is broken, not HRX."
  "$USE_ROCM" >/dev/null 2>&1 || true
  exit 2
fi

echo "==================== [2/2] COMPARE on HRX ==========================="
"$USE_HRX" | sed 's/^/  /'
HRX_AB_MODE=compare pytest $PYTEST_ARGS
rc=$?

echo "==================== restore native ROCm ============================"
"$USE_ROCM" | sed 's/^/  /'

if [ "$rc" -eq 0 ]; then
  echo "RESULT: PASS — pytorch works with hrx-on-hip (matches native ROCm)."
else
  echo "RESULT: FAIL — HRX diverged from native or crashed (pytest rc=$rc)."
fi
exit "$rc"
