# hrx-pytorch-smoke-test

A presubmit smoke suite that answers **"does PyTorch work with hrx-on-HIP?"** by
running a pytest suite on the **HRX** HIP binding and asserting it matches a
**native ROCm** baseline on the same GPU.

## Why native A/B (not a CPU oracle)

The HRX `libamdhip64.so.7` and stock ROCm cannot be loaded in the same process —
the venv picks one via a symlink (`use_hrx.sh` / `use_rocm.sh`). So the suite
runs twice:

1. **record** on native ROCm — every `ab.check(...)` saves its result as a golden;
2. **compare** on HRX — every `ab.check(...)` asserts its result matches the
   native golden (and nothing crashes).

Any HRX-vs-native divergence (or crash) is a failure. Inputs and model init are
generated deterministically on the CPU, so both passes share identical
weights+inputs and only the GPU backend differs.

## Run it

Prereqs: a built HRX binding (`libhrx_src_binding_hip_amdhip64`), the
`use_hrx.sh` / `use_rocm.sh` backend-swap scripts, and the nightly pytorch+rocm
venv.

```bash
pip install -r requirements-test.txt    # pytest (torch comes from the venv)
./run_ab.sh
```

`run_ab.sh` (re)builds the HRX binding, records goldens on native, compares on
HRX, restores native, and exits non-zero on any divergence/crash. Override paths
via `HRX_VENV`, `USE_HRX`, `USE_ROCM`, `HRX_BUILD_DIR`, `HRX_SKIP_BUILD=1`
(see the script header).

### Single-backend / manual runs

```bash
# against whichever backend is currently swapped in:
HRX_AB_MODE=record  pytest    # save goldens   (run while native is active)
HRX_AB_MODE=compare pytest    # assert vs goldens (run while HRX is active)
```

## What's covered

- device enumeration & properties (`tests/test_device.py`)
- elementwise ops, matmul/bmm, reductions, softmax, layernorm, conv2d across
  fp32/fp16/bf16; on-device RNG determinism; exact index ops (`tests/test_ops.py`)
- H2D/D2H/D2D copies, fills, non-contiguous, allocator churn (`tests/test_memory.py`)
- MLP / CNN / ResNet-18 training: loss decreases **and** matches native
  (`tests/test_train.py`)

## Out of scope (known gap)

DDP/FSDP: RCCL topology discovery needs a real PCIe BDF, which HRX does not yet
report (it returns a synthetic device-ordinal identity). Tracked separately in
the HRX bring-up notes.
