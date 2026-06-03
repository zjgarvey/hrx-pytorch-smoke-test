#!/usr/bin/env bash
# use_rocm.sh — restore the stock ROCm libamdhip64.so.7 as the SOLE library the
# venv's PyTorch loads (the native baseline). Reverses use_hrx.sh. Exactly one
# backend is live at a time — no LD_PRELOAD, no passthrough.
#
# Paths are DISCOVERED, never hardcoded:
#   HRX_CORE_LIB  the venv's _rocm_sdk_core/lib dir. If unset, auto-detected from
#                 the active venv via `import _rocm_sdk_core` (so this also works
#                 standalone after `source <venv>/bin/activate`).
set -euo pipefail

CORE="${HRX_CORE_LIB:-$(python -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))' 2>/dev/null || true)}"
[ -n "$CORE" ] && [ -d "$CORE" ] || {
  echo "ERROR: can't locate _rocm_sdk_core/lib — set HRX_CORE_LIB or activate the venv." >&2; exit 1; }

LIVE="$CORE/libamdhip64.so.7"
ORIG="$CORE/libamdhip64.so.7.orig"
[ -e "$ORIG" ] || { echo "ERROR: no backup at $ORIG — nothing to restore (run use_hrx.sh first)." >&2; exit 1; }

ln -sfn "libamdhip64.so.7.orig" "$LIVE"
echo "== native ROCm backend ACTIVE =="
ls -la "$LIVE"
if command -v readelf >/dev/null; then
  printf 'SONAME: '; readelf -d "$LIVE" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p'
fi
