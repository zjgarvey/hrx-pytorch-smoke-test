#!/usr/bin/env bash
# install_system_deps.sh — install the system build toolchain for building the
# HRX HIP binding on a Debian/Ubuntu host (or in a fresh ubuntu container).
#
# Installs: clang/lld/llvm-ar (LLVM ${LLVM_VERSION:-23} from apt.llvm.org),
# cmake, ninja, ccache, git, python3.12 + venv, libzstd-dev, binutils, curl.
# Idempotent; needs root (uses sudo if not already root).
#
# ROCm and PyTorch are deliberately NOT installed here — those arrive as nightly
# venv wheels (TheRock), wired up by build_and_test.sh. Nothing here touches
# /opt/rocm.
set -euo pipefail
LLVM_VERSION="${LLVM_VERSION:-23}"

SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"
command -v apt-get >/dev/null || {
  echo "ERROR: this helper supports apt (Debian/Ubuntu) only. Install the toolchain" >&2
  echo "       (clang-${LLVM_VERSION}, lld, cmake, ninja, ccache, libzstd-dev, python3.12-venv) by hand." >&2
  exit 1; }

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg lsb-release software-properties-common \
  git build-essential binutils \
  python3.12 python3.12-venv python3.12-dev python3-pip \
  cmake ninja-build ccache libzstd-dev libnuma1

# LLVM/clang toolchain from apt.llvm.org (official llvm.sh installer).
if [ ! -x "/usr/lib/llvm-${LLVM_VERSION}/bin/clang" ] && ! command -v "clang-${LLVM_VERSION}" >/dev/null; then
  curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
  $SUDO bash /tmp/llvm.sh "${LLVM_VERSION}"
  rm -f /tmp/llvm.sh
fi

# Expose unversioned clang/clang++/lld/llvm-ar/... on PATH, pinned to this LLVM.
LLVM_BIN="/usr/lib/llvm-${LLVM_VERSION}/bin"
[ -d "$LLVM_BIN" ] || { echo "ERROR: $LLVM_BIN missing after LLVM install." >&2; exit 1; }
for t in clang clang++ clang-cpp lld ld.lld llvm-ar llvm-ranlib clang-scan-deps; do
  [ -x "$LLVM_BIN/$t" ] && $SUDO ln -sf "$LLVM_BIN/$t" "/usr/local/bin/$t"
done

echo "== toolchain =="
for t in clang clang++ lld llvm-ar cmake ninja ccache git python3.12; do
  printf '%-14s ' "$t"; command -v "$t" || echo MISSING
done
clang --version | head -1
