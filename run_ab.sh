#!/usr/bin/env bash
# run_ab.sh — "does pytorch work with hrx-on-hip?" via a native-ROCm A/B.
#
#   0. ensure native ROCm is live and a GPU is actually visible.
#   1. RECORD : on native ROCm, run the suite (saves golden tensors and confirms
#               the native baseline itself is sane).
#   2. COMPARE: swap to HRX, run the same suite (asserts every result matches the
#               native golden and that nothing crashes).
#   3. restore native ROCm.
#
# Exit 0 iff the native baseline passed AND HRX matched it. This script only runs
# the A/B; provisioning (venv, nightly ROCm/PyTorch, building HRX) is done by
# build_and_test.sh. For a one-shot from-scratch run, use that instead.
#
# Inputs (env):
#   HRX_VENV        venv to activate (default: the already-active venv, else error)
#   HRX_BUILD_DIR   HRX build dir — used to locate the binding for the HRX pass
#                   (or set HRX_BINDING directly to the libamdhip64.so.7* file)
#   HRX_ALLOW_NO_GPU=1   treat "no GPU visible" as a (verifies-nothing) pass
#   USE_HRX / USE_ROCM   backend-swap scripts (default: this repo's scripts/)
#   HRX_SKIP_BUILD=1     skip the optional incremental rebuild
#   PYTEST_ARGS, HRX_AB_GOLDEN_DIR
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${HRX_VENV:-${VIRTUAL_ENV:-}}"
[ -n "$VENV" ] || { echo "FAIL: no venv — set HRX_VENV=/path/to/venv (or activate one)." >&2; exit 4; }
USE_HRX="${USE_HRX:-$HERE/scripts/use_hrx.sh}"
USE_ROCM="${USE_ROCM:-$HERE/scripts/use_rocm.sh}"
GOLDEN_DIR="${HRX_AB_GOLDEN_DIR:-$HERE/goldens}"
PYTEST_ARGS="${PYTEST_ARGS:--q -ra}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HRX_GPU_DRIVER=amdgpu AMD_SERIALIZE_KERNEL=1
export HRX_AB_GOLDEN_DIR="$GOLDEN_DIR"
export HRX_BUILD_DIR="${HRX_BUILD_DIR:-}"   # consumed by use_hrx.sh to find the binding
unset LD_PRELOAD || true
cd "$HERE"
rm -rf "$GOLDEN_DIR"

CORE="$(python -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))' 2>/dev/null || true)"
export HRX_CORE_LIB="${HRX_CORE_LIB:-$CORE}"

ensure_native() {
  if [ -e "$HRX_CORE_LIB/libamdhip64.so.7.orig" ]; then
    "$USE_ROCM" | sed 's/^/  /'
  else
    echo "  (native ROCm already live — no .orig backup yet)"
  fi
}

# Gate the suite's validity: make sure the lib torch will load is actually the
# backend we think it is. Without this, a failed swap would run COMPARE on native
# and "pass" (native-vs-native) — a false green. Skipped if the binding path is
# unknown (manual runs without HRX_BINDING).
assert_backend() {  # $1 = hrx|native
  [ -n "${HRX_BINDING:-}" ] || return 0
  local live bind
  live="$(readlink -f "$HRX_CORE_LIB/libamdhip64.so.7" 2>/dev/null || true)"
  bind="$(readlink -f "$HRX_BINDING" 2>/dev/null || true)"
  if [ "$1" = hrx ] && [ "$live" != "$bind" ]; then
    echo "FAIL: expected HRX live, but libamdhip64.so.7 -> $live"; return 1
  fi
  if [ "$1" = native ] && [ "$live" = "$bind" ]; then
    echo "FAIL: expected native live, but the HRX binding is still swapped in"; return 1
  fi
  return 0
}

# Optional incremental rebuild (only if a build dir with ninja is supplied).
if [ "${HRX_SKIP_BUILD:-0}" != "1" ] && [ -n "${HRX_BUILD_DIR:-}" ] && [ -f "$HRX_BUILD_DIR/build.ninja" ]; then
  echo "==================== building HRX binding ==========================="
  cmake --build "$HRX_BUILD_DIR" --target libhrx_src_binding_hip_amdhip64 \
    -j"$(nproc)" || { echo "FAIL: HRX build failed"; exit 3; }
fi

echo "============== [0/2] ensure native ROCm + GPU present =============="
ensure_native
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "FAIL: no GPU visible to torch (torch.cuda.is_available()=False)."
  echo "      The A/B verifies nothing without a GPU — pass the device into the"
  echo "      container (--device=/dev/kfd --device=/dev/dri ...)."
  if [ "${HRX_ALLOW_NO_GPU:-0}" = "1" ]; then
    echo "      HRX_ALLOW_NO_GPU=1 → exiting 0 (NOTHING was actually verified)."
    exit 0
  fi
  exit 5
fi
assert_backend native || { ensure_native; exit 6; }

echo "==================== [1/2] RECORD on native ROCm ===================="
if ! HRX_AB_MODE=record pytest $PYTEST_ARGS; then
  echo "FAIL: native ROCm baseline did not pass — the test/env is broken, not HRX."
  ensure_native >/dev/null 2>&1 || true
  exit 2
fi

echo "==================== [2/2] COMPARE on HRX ==========================="
if ! "$USE_HRX" | sed 's/^/  /'; then
  echo "FAIL: could not swap in the HRX backend."; ensure_native; exit 6
fi
assert_backend hrx || { ensure_native; exit 6; }
HRX_AB_MODE=compare pytest $PYTEST_ARGS
rc=$?

echo "==================== restore native ROCm ============================"
ensure_native

if [ "$rc" -eq 0 ]; then
  echo "RESULT: PASS — pytorch works with hrx-on-hip (matches native ROCm)."
else
  echo "RESULT: FAIL — HRX diverged from native or crashed (pytest rc=$rc)."
fi
exit "$rc"
