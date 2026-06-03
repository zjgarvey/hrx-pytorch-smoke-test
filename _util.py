"""Deterministic helpers shared by the suite.

Inputs are generated on the CPU with a fixed Generator and then moved to the
GPU, so the values are bit-identical regardless of the active backend — the A/B
comparison then isolates the operator/model under test rather than the RNG path.
(On-device RNG is exercised by its own test.)
"""

import torch

# HRX routes torch's own kernels to the same GPU as native, so results should be
# very close to native. Tolerances tolerate run-to-run nondeterminism (atomics,
# BLAS algorithm selection) while still catching real divergence.
TOL = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.float64: dict(atol=1e-6, rtol=1e-6),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=3e-2, rtol=3e-2),
}

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _ids(dtype):
    return str(dtype).split(".")[-1]


def cpu_seeded(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)


def dev(*shape, seed=0, dtype=torch.float32, device="cuda"):
    """Deterministic device tensor: generated on CPU (seeded), moved to device."""
    return cpu_seeded(*shape, seed=seed, dtype=dtype).to(device)


def dev_randint(high, *shape, seed=0, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, high, shape, generator=g).to(device)


def tol(dtype):
    return TOL.get(dtype, TOL[torch.float32])
