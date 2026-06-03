#!/usr/bin/env bash
# build_and_test.sh — one command from a bare machine to a hrx-on-HIP A/B result.
#
# Steps (each skippable):
#   1. [--install-system-deps] apt-install the build toolchain (clang/lld/cmake/...)
#   2. create a venv and install nightly ROCm + PyTorch + the test deps
#   3. clone HRX at a given ref and build the libamdhip64 binding
#         (or reuse a prebuilt tree with --hrx-build-dir)
#   4. run the native-ROCm A/B smoke suite (record on native, compare on HRX)
#
# Nothing here points at /opt/rocm or any /home/... path: ROCm is the venv's
# TheRock SDK (located via `rocm-sdk path`), and every work dir is configurable.
#
# Examples:
#   ./build_and_test.sh --install-system-deps           # bare host/container, full run
#   ./build_and_test.sh --hrx-ref my/branch             # test a different HRX branch
#   ./build_and_test.sh --hrx-build-dir /path/build/hrx # reuse an existing build
#   ./build_and_test.sh --skip-venv --skip-build        # just re-run the A/B
#
# A GPU must be visible (pass --device=/dev/kfd --device=/dev/dri into the
# container). Without one the A/B verifies nothing and fails loudly unless
# --allow-no-gpu is given.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- defaults (overridable by flag or env) ---------------------------------
HRX_REPO="${HRX_REPO:-https://github.com/ROCm/hrx-system.git}"
HRX_REF="${HRX_REF:-users/zjgarvey/feature/nightly_rocm_plus_pytorch_support}"
INDEX_URL="${HRX_INDEX_URL:-https://rocm.nightlies.amd.com/whl-multi-arch/}"
DEVICE="${ROCM_DEVICE:-device-gfx942}"
CHIP="${HRX_CHIP:-gfx942}"
PYTHON="${PYTHON:-python3.12}"
WORKDIR="${HRX_WORKDIR:-$(dirname "$REPO_ROOT")}"   # venv/src as siblings of the repo
JOBS="${JOBS:-$(nproc)}"

VENV="${HRX_VENV:-}"
HRX_SRC="${HRX_SRC:-}"
HRX_BUILD_DIR="${HRX_BUILD_DIR:-}"
INSTALL_SYS=0; SKIP_VENV=0; SKIP_BUILD=0; RUN_STRESS=0; ALLOW_NO_GPU=0

usage(){ sed -n '2,31p' "$0"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hrx-repo) HRX_REPO="$2"; shift 2;;
    --hrx-ref) HRX_REF="$2"; shift 2;;
    --hrx-src-dir) HRX_SRC="$2"; shift 2;;
    --hrx-build-dir) HRX_BUILD_DIR="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --chip) CHIP="$2"; shift 2;;
    --index-url) INDEX_URL="$2"; shift 2;;
    --workdir) WORKDIR="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --install-system-deps) INSTALL_SYS=1; shift;;
    --skip-venv) SKIP_VENV=1; shift;;
    --skip-build) SKIP_BUILD=1; shift;;
    --stress) RUN_STRESS=1; shift;;
    --allow-no-gpu) ALLOW_NO_GPU=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

VENV="${VENV:-$WORKDIR/hrx-venv}"
HRX_SRC="${HRX_SRC:-$WORKDIR/hrx}"
mkdir -p "$WORKDIR"
log(){ printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }
activate(){ set +u; # shellcheck disable=SC1091
  source "$VENV/bin/activate"; set -u; }

# ---- 1. system deps --------------------------------------------------------
if [ "$INSTALL_SYS" -eq 1 ]; then
  log "[1/4] install system build deps"
  "$REPO_ROOT/scripts/install_system_deps.sh"
else
  echo "[1/4] skipping system-deps install (pass --install-system-deps on a bare host)"
fi

# ---- 2. venv + nightly ROCm/PyTorch + test deps ----------------------------
if [ "$SKIP_VENV" -eq 1 ] && [ -d "$VENV" ]; then
  log "[2/4] reuse existing venv: $VENV"
  activate
else
  log "[2/4] create venv + nightly ROCm/PyTorch + test deps  (index: $INDEX_URL)"
  rm -rf "$VENV"
  "$PYTHON" -m venv "$VENV"
  activate
  python -m pip install --no-cache-dir --upgrade pip
  pip install --no-cache-dir --index-url "$INDEX_URL" "rocm[libraries,devel,${DEVICE}]"
  pip install --no-cache-dir --index-url "$INDEX_URL" "torch[${DEVICE}]" "torchvision[${DEVICE}]" torchaudio
  pip install --no-cache-dir -r "$REPO_ROOT/requirements-test.txt"
fi
command -v rocm-sdk >/dev/null || { echo "FAIL: rocm-sdk not in venv — wheel install failed." >&2; exit 1; }
echo ">> ROCM_HOME: $(rocm-sdk path --root)"
python -c "import torch; print('>> torch', torch.__version__, '| hip', getattr(torch.version,'hip',None), '| cuda?', torch.cuda.is_available())"

# ---- 3. build (or reuse) the HRX binding -----------------------------------
configure_and_build_hrx(){
  local src="$1" build="$2" rocm_cmake
  rocm_cmake="$(rocm-sdk path --cmake)"
  [ -d "$rocm_cmake" ] || { echo "FAIL: rocm-sdk cmake dir missing: $rocm_cmake" >&2; exit 1; }
  # Prefer HRX's own configure script if the fetched ref ships one (keeps build
  # options versioned with HRX); else use the embedded invocation below.
  if [ -x "$src/build_tools/scripts/configure_hrx.sh" ]; then
    echo ">> using HRX repo's build_tools/scripts/configure_hrx.sh"
    ROCM_HOME="$(rocm-sdk path --root)" JOBS="$JOBS" \
      "$src/build_tools/scripts/configure_hrx.sh" --build-dir "$build" --chip "$CHIP"
    return
  fi
  echo ">> embedded cmake configure"
  local CC CXX AR RANLIB
  CC="$(command -v clang-23 || command -v clang)"
  CXX="$(command -v clang++-23 || command -v clang++)"
  AR="$(command -v llvm-ar-23 || command -v llvm-ar || command -v ar)"
  RANLIB="$(command -v llvm-ranlib-23 || command -v llvm-ranlib || command -v ranlib)"
  local launcher=()
  if command -v ccache >/dev/null; then
    launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
  fi
  cmake -S "$src" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_PREFIX_PATH="$rocm_cmake" \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_ASM_COMPILER="$CC" \
    -DCMAKE_AR="$AR" -DCMAKE_RANLIB="$RANLIB" \
    "${launcher[@]}" \
    -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
    -DCMAKE_MODULE_LINKER_FLAGS="-fuse-ld=lld" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DIREE_HAL_DRIVER_AMDGPU=ON \
    -DLIBHRX_BUILD=ON -DLIBHRX_BUILD_CUDA_BINDING=OFF \
    -DLIBHRX_BUILD_CTS=OFF -DLIBHRX_BUILD_PASSTHROUGH=OFF \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=prebuilt \
    -DIREE_ROCM_TEST_TARGET_CHIP="$CHIP"
  cmake --build "$build" --target libhrx_src_binding_hip_amdhip64 -j "$JOBS"
}

if [ -n "$HRX_BUILD_DIR" ]; then
  log "[3/4] using prebuilt HRX build: $HRX_BUILD_DIR"
elif [ "$SKIP_BUILD" -eq 1 ] && [ -f "$HRX_SRC/build/hrx/build.ninja" ]; then
  HRX_BUILD_DIR="$HRX_SRC/build/hrx"
  log "[3/4] reuse existing HRX build: $HRX_BUILD_DIR"
else
  log "[3/4] clone HRX ($HRX_REF) + build the binding into $HRX_SRC/build/hrx"
  if [ -d "$HRX_SRC/.git" ]; then
    git -C "$HRX_SRC" fetch origin "$HRX_REF"
    git -C "$HRX_SRC" checkout -q FETCH_HEAD
  elif git clone --depth 1 --branch "$HRX_REF" "$HRX_REPO" "$HRX_SRC" 2>/dev/null; then
    : # fast path: ref is a branch/tag
  else
    git clone "$HRX_REPO" "$HRX_SRC"          # fallback: full clone (supports a raw SHA)
    git -C "$HRX_SRC" checkout -q "$HRX_REF"
  fi
  echo ">> HRX at $(git -C "$HRX_SRC" rev-parse --short HEAD)"
  HRX_BUILD_DIR="$HRX_SRC/build/hrx"
  configure_and_build_hrx "$HRX_SRC" "$HRX_BUILD_DIR"
fi

BINDING="$(readlink -f "$HRX_BUILD_DIR/libhrx/src/binding/hip/libamdhip64.so.7" 2>/dev/null || true)"
[ -n "$BINDING" ] && [ -e "$BINDING" ] || {
  echo "FAIL: built HRX binding not found under $HRX_BUILD_DIR/libhrx/src/binding/hip/" >&2; exit 1; }
echo ">> HRX binding: $BINDING"

# ---- 4. run the A/B smoke suite --------------------------------------------
log "[4/4] native-ROCm A/B smoke suite"
export HRX_VENV="$VENV" HRX_BUILD_DIR="$HRX_BUILD_DIR" HRX_BINDING="$BINDING" HRX_SKIP_BUILD=1
[ "$ALLOW_NO_GPU" -eq 1 ] && export HRX_ALLOW_NO_GPU=1
rc=0
"$REPO_ROOT/run_ab.sh" || rc=$?
if [ "$RUN_STRESS" -eq 1 ] && [ "$rc" -eq 0 ]; then
  log "stress/endurance suite"
  "$REPO_ROOT/run_stress.sh" || rc=$?
fi
exit "$rc"
