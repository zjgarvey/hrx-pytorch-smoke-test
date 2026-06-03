# hrx-pytorch-smoke-test

A presubmit smoke suite that answers **"does PyTorch work with hrx-on-HIP?"** by
running a pytest suite on the **HRX** HIP binding and asserting it matches a
**native ROCm** baseline on the same GPU.

It is **self-contained**: one script fetches a HRX branch, provisions a nightly
ROCm + PyTorch venv, builds the HRX binding, and runs the A/B — no preexisting
local setup required. The acceptance bar is *"clone this repo into a clean Ubuntu
box with a GPU, run one script, get a PASS."*

## Why native A/B (not a CPU oracle)

The HRX `libamdhip64.so.7` and stock ROCm cannot be loaded in the same process —
the venv loads exactly one, chosen by a symlink (`scripts/use_hrx.sh` /
`scripts/use_rocm.sh`). So the suite runs twice:

1. **record** on native ROCm — every `ab.check(...)` saves its result as a golden;
2. **compare** on HRX — every `ab.check(...)` asserts its result matches the
   native golden (and nothing crashes).

Any HRX-vs-native divergence (or crash) is a failure. Inputs and model init are
generated deterministically on the CPU, so both passes share identical
weights+inputs and only the GPU backend differs.

## Quick start — one command

On a Debian/Ubuntu host (or container) with an AMD GPU visible:

```bash
./build_and_test.sh --install-system-deps
```

That will, in order: install the build toolchain, create a venv and install the
**nightly** ROCm + PyTorch wheels (TheRock), clone and build the HRX binding, and
run the A/B. It exits non-zero on any divergence, crash, or — deliberately — if
no GPU is visible (an all-skipped run verifies nothing; pass `--allow-no-gpu` to
override). Drop `--install-system-deps` if the toolchain (clang-23, lld, cmake,
ninja, ccache, libzstd-dev, python3.12-venv) is already present.

Useful flags (all also settable via env; see the script header):

| flag | meaning | default |
|------|---------|---------|
| `--hrx-ref REF` | HRX branch/tag/sha to build | the nightly feature branch |
| `--hrx-repo URL` | HRX git remote | `github.com/ROCm/hrx-system` |
| `--hrx-build-dir DIR` | reuse an existing HRX build (skip clone+build) | — |
| `--venv DIR` | venv location | `<repo-parent>/hrx-venv` |
| `--device SPEC` | TheRock device package | `device-gfx942` |
| `--chip CHIP` | GPU arch for the build | `gfx942` |
| `--workdir DIR` | where venv/HRX source live | repo parent |
| `--skip-venv` / `--skip-build` | reuse the existing venv / build | off |
| `--stress` | also run the endurance suite after the A/B | off |
| `--allow-no-gpu` | treat "no GPU" as a (skipped) pass | off |

```bash
./build_and_test.sh --hrx-ref my/experimental/branch   # test another HRX branch
./build_and_test.sh --hrx-build-dir /path/build/hrx     # reuse a prebuilt binding
./build_and_test.sh --skip-venv --skip-build            # just re-run the A/B
```

## Run it in a clean container

`scripts/docker_smoke.sh` is the end-to-end acceptance check: it builds a
toolchain image (`Dockerfile`, just clang-23/cmake/ninja/...), then in a **fresh**
container with the GPU passed through it clones this repo and runs
`build_and_test.sh`.

```bash
git add -A && git commit -m wip      # docker_smoke clones the committed tree
scripts/docker_smoke.sh              # build image + full build+A/B with GPU passthrough
scripts/docker_smoke.sh --stress     # extra flags are forwarded to build_and_test.sh
HRX_REF=my/branch scripts/docker_smoke.sh
```

Equivalent by hand, against a bare `ubuntu:24.04`:

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  ubuntu:24.04 bash
# inside the container:
apt-get update && apt-get install -y git
git clone <this-repo-url> /work/test && cd /work/test
./build_and_test.sh --install-system-deps
```

## What's covered

- device enumeration & properties (`tests/test_device.py`)
- elementwise ops, matmul/bmm, reductions, softmax, layernorm, conv2d across
  fp32/fp16/bf16; on-device RNG determinism; exact index ops (`tests/test_ops.py`)
- H2D/D2H/D2D copies, fills, non-contiguous, allocator churn (`tests/test_memory.py`)
- MLP / CNN / ResNet-18 training: loss decreases **and** matches native
  (`tests/test_train.py`)

## Single-backend / manual runs

`build_and_test.sh` orchestrates everything, but the pieces compose. With a venv
active and the HRX binding built, swap backends and run pytest directly:

```bash
source <venv>/bin/activate
export HRX_BUILD_DIR=<hrx>/build/hrx          # so use_hrx.sh can find the binding
scripts/use_rocm.sh && HRX_AB_MODE=record  pytest   # save goldens on native
scripts/use_hrx.sh  && HRX_AB_MODE=compare pytest   # assert vs goldens on HRX
scripts/use_rocm.sh                                  # restore native
```

`run_ab.sh` wraps exactly that (record → compare → restore, with a GPU guard);
`build_and_test.sh` calls it after provisioning. The swap scripts discover the
venv's `_rocm_sdk_core/lib` automatically; point them at a binding via
`HRX_BUILD_DIR` or `HRX_BINDING`.

## Stress / endurance (opt-in)

A separate, longer job (`@pytest.mark.stress`, excluded from the presubmit) that
asserts hrx-on-HIP stays stable and correct under load — no crash / NaN / leak:
soak training, allocator churn, multi-stream concurrency, sustained kernel
dispatch, repeated large matmul, and a mixed-op soak (`tests/test_stress.py`).
Runs on HRX only (stability invariants, not a native A/B):

```bash
./run_stress.sh                       # default load (or: build_and_test.sh --stress)
HRX_STRESS_SCALE=4 ./run_stress.sh    # 4x longer soak for a nightly job
```

Leak detection warms up first (torch allocates ~160 MB of one-time persistent
CUDA state — identical on native and HRX, not freed by `empty_cache()`), then
measures per-iteration growth, so one-time state is not mistaken for a leak.

## Out of scope (known gap)

DDP/FSDP: RCCL topology discovery needs a real PCIe BDF, which HRX does not yet
report (it returns a synthetic device-ordinal identity). Tracked separately in
the HRX bring-up notes.
