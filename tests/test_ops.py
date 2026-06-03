"""Operator correctness: HRX result must match the native-ROCm golden."""

import pytest
import torch

from _util import DTYPES, _ids, dev, tol

cuda = pytest.mark.usefixtures("torch_cuda")

_ELEMENTWISE = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "sub": lambda a, b: a - b,
    "relu": lambda a, b: torch.relu(a),
    "gelu": lambda a, b: torch.nn.functional.gelu(a),
    "sigmoid": lambda a, b: torch.sigmoid(a),
    "tanh": lambda a, b: torch.tanh(a),
    "exp": lambda a, b: torch.exp(a.clamp(-5, 5)),
    "rsqrt": lambda a, b: torch.rsqrt(a.abs() + 1e-3),
}


@cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_ids)
@pytest.mark.parametrize("op", list(_ELEMENTWISE))
def test_elementwise(ab, dtype, op):
    a = dev(1024, seed=1, dtype=dtype)
    b = dev(1024, seed=2, dtype=dtype)
    out = _ELEMENTWISE[op](a, b)
    torch.cuda.synchronize()
    ab.check(out, name=op, **tol(dtype))


@cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_ids)
def test_matmul(ab, dtype):
    x = dev(256, 256, seed=3, dtype=dtype)
    out = x @ x
    torch.cuda.synchronize()
    ab.check(out, **tol(dtype))


@cuda
def test_bmm(ab):
    a = dev(8, 64, 32, seed=4)
    b = dev(8, 32, 16, seed=5)
    ab.check(torch.bmm(a, b), atol=1e-3, rtol=1e-3)


@cuda
@pytest.mark.parametrize("red", ["sum", "mean", "amax", "amin"])
def test_reductions(ab, red):
    x = dev(512, 256, seed=6)
    ab.check(getattr(torch, red)(x, dim=1), name=red, atol=1e-3, rtol=1e-3)


@cuda
def test_argmax_indices_exact(ab):
    x = dev(256, 512, seed=11)
    # Index outputs must match native exactly.
    ab.check(torch.argmax(x, dim=1).to(torch.int32), atol=0, rtol=0)


@cuda
def test_softmax(ab):
    x = dev(128, 1000, seed=7)
    ab.check(torch.softmax(x, dim=1), atol=1e-5, rtol=1e-4)


@cuda
def test_layernorm(ab):
    x = dev(32, 256, seed=10)
    ab.check(torch.nn.functional.layer_norm(x, (256,)), atol=1e-4, rtol=1e-4)


@cuda
def test_conv2d(ab):
    x = dev(4, 8, 32, 32, seed=8)
    w = dev(16, 8, 3, 3, seed=9)
    out = torch.nn.functional.conv2d(x, w, padding=1)
    torch.cuda.synchronize()
    ab.check(out, atol=1e-3, rtol=1e-3)


@cuda
def test_ondevice_rng_is_deterministic(ab):
    # CUDA philox RNG is deterministic; HRX runs the same kernel as native, so
    # seeded on-device randn must be bit-identical to the native golden.
    g = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn(4096, device="cuda", generator=g)
    torch.cuda.synchronize()
    ab.check(x, atol=0, rtol=0)
