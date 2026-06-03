#!/usr/bin/env bash
# use_hrx.sh — make the HRX HIP binding the SOLE libamdhip64.so.7 the venv's
# PyTorch loads. Single backend: no LD_PRELOAD, no passthrough, no two-library
# confound. Reversible: use_rocm.sh restores stock ROCm.
#
# Why a file-swap (not LD_PRELOAD): two libamdhip64 in one process split HIP
# runtime state across libraries (handles from lib A are opaque to lib B).
# Swapping the single file torch actually loads keeps exactly one backend live.
#
# Paths are DISCOVERED, never hardcoded:
#   HRX_CORE_LIB  the venv's _rocm_sdk_core/lib dir (auto-detected from the active
#                 venv if unset).
#   HRX_BINDING   the built binding artifact (a real libamdhip64.so.7* file). If
#                 unset, derived from HRX_BUILD_DIR/libhrx/src/binding/hip/.
set -euo pipefail

CORE="${HRX_CORE_LIB:-$(python -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))' 2>/dev/null || true)}"
[ -n "$CORE" ] && [ -d "$CORE" ] || {
  echo "ERROR: can't locate _rocm_sdk_core/lib — set HRX_CORE_LIB or activate the venv." >&2; exit 1; }

HRX="${HRX_BINDING:-}"
if [ -z "$HRX" ] && [ -n "${HRX_BUILD_DIR:-}" ]; then
  cand="$HRX_BUILD_DIR/libhrx/src/binding/hip/libamdhip64.so.7"
  [ -e "$cand" ] && HRX="$(readlink -f "$cand")"
fi
[ -n "$HRX" ] && [ -e "$HRX" ] || {
  echo "ERROR: HRX binding not found — set HRX_BINDING=/path/to/libamdhip64.so.7* or HRX_BUILD_DIR." >&2; exit 1; }

LIVE="$CORE/libamdhip64.so.7"
ORIG="$CORE/libamdhip64.so.7.orig"

# One-time backup of the stock ROCm lib (only if not already saved).
if [ ! -e "$ORIG" ]; then
  if [ -L "$LIVE" ]; then
    echo "ERROR: $LIVE is already a symlink but no .orig backup exists — refusing to clobber." >&2
    exit 1
  fi
  cp -a "$LIVE" "$ORIG"
  echo "backed up stock ROCm libamdhip64.so.7 -> $(basename "$ORIG")"
fi

ln -sfn "$HRX" "$LIVE"
echo "== HRX backend ACTIVE =="
ls -la "$LIVE"
if command -v readelf >/dev/null; then
  printf 'SONAME: '; readelf -d "$LIVE" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p'
fi
