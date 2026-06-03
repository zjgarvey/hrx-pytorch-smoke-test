"""Runtime stress / endurance — does hrx-on-hip stay correct and stable under load?

Opt-in (`@pytest.mark.stress`, excluded from the fast presubmit; run via
run_stress.sh). These run on whichever backend is active (intended: HRX) and
assert *stability* invariants rather than an exact native A/B:

  - no crash / hang over sustained, concurrent, or heavy use,
  - no NaN/Inf,
  - no per-iteration torch-tensor memory leak,
  - results stay correct under load.

Leak methodology: torch allocates ~160 MB of one-time persistent CUDA state
(cublas/MIOpen workspaces, etc.) on first use — identical on native and HRX, and
NOT freed by empty_cache(). That is not a leak. So `_leak_growth` warms up first
to allocate that persistent state, then measures live-tensor *growth* across the
measured window: a real per-iteration leak shows up as growth; one-time state
does not. Scale every workload with HRX_STRESS_SCALE (default 1).
"""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from _util import dev, dev_randint

cuda = pytest.mark.usefixtures("torch_cuda")
stress = pytest.mark.stress

SCALE = float(os.environ.get("HRX_STRESS_SCALE", "1"))
# Per-iteration leak is detected as live-tensor growth across the measured window
# (after warmup). A few MB of slack absorbs allocator fragmentation.
LEAK_BUDGET = 8 * 1024 * 1024


def _n(base):
    return max(1, int(base * SCALE))


def _leak_growth(step, *, warmup, measured):
    """Run `warmup` steps to allocate one-time persistent state, then return the
    growth in live torch memory across `measured` more steps."""
    for i in range(warmup):
        step(i)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    for i in range(measured):
        step(warmup + i)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() - base


@stress
@cuda
def test_training_endurance():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 10),
    ).to("cuda")
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    x = dev(128, 256, seed=1)
    y = dev_randint(10, 128, seed=2)
    losses = {}

    def step(i):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        v = float(loss.detach())
        assert v == v and abs(v) < 1e6, f"loss non-finite at step {i}: {v}"
        losses[i] = v

    total = _n(300)
    warmup = max(5, total // 10)
    grew = _leak_growth(step, warmup=warmup, measured=total - warmup)
    assert losses[total - 1] < losses[0], (
        f"loss did not decrease ({losses[0]} -> {losses[total - 1]})"
    )
    assert grew <= LEAK_BUDGET, (
        f"training leaked {grew} bytes over {total - warmup} measured steps"
    )


@stress
@cuda
def test_allocator_churn():
    g = torch.Generator().manual_seed(0)

    def step(_i):
        n = int(torch.randint(1, 4_000_000, (1,), generator=g).item())
        t = torch.empty(n, device="cuda")
        t.fill_(1.0)
        del t

    total = _n(3000)
    warmup = max(20, total // 10)
    grew = _leak_growth(step, warmup=warmup, measured=total - warmup)
    assert grew <= LEAK_BUDGET, (
        f"allocator leaked {grew} bytes over {total - warmup} measured cycles"
    )


@stress
@cuda
def test_multistream_concurrency():
    x = dev(512, 512, seed=3)
    ref = (x @ x).cpu()
    streams = [torch.cuda.Stream() for _ in range(8)]
    outs = []
    for _ in range(_n(4)):
        for s in streams:
            with torch.cuda.stream(s):
                outs.append(x @ x)
    torch.cuda.synchronize()
    for o in outs:
        torch.testing.assert_close(o.cpu(), ref, atol=1e-3, rtol=1e-3)


@stress
@cuda
def test_sustained_dispatch():
    a = dev(65536, seed=4)
    acc = torch.zeros_like(a)
    iters = _n(20000)
    for _ in range(iters):
        acc.add_(a)  # many small in-place kernel launches, no allocation
    torch.cuda.synchronize()
    assert torch.isfinite(acc).all()
    torch.testing.assert_close(acc, a * float(iters), atol=1.0, rtol=1e-3)


@stress
@cuda
def test_large_matmul_repeated():
    N = 2048
    x = dev(N, N, seed=5)
    out = None
    for _ in range(_n(30)):
        out = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), "large matmul produced non-finite values"
    torch.testing.assert_close(out[0:1].cpu(), (x[0:1] @ x).cpu(), atol=5e-1, rtol=1e-2)


@stress
@cuda
def test_mixed_op_soak():
    conv_w = dev(16, 8, 3, 3, seed=40)
    img = dev(8, 8, 32, 32, seed=41)
    mat = dev(512, 512, seed=42)
    vals = {}

    def step(i):
        a = torch.relu(mat @ mat)
        b = torch.softmax(a, dim=1).sum()
        c = F.conv2d(img, conv_w, padding=1).mean()
        v = float((b + c).detach())
        assert v == v and abs(v) < 1e8, f"mixed soak non-finite at iter {i}: {v}"
        vals[i] = v

    total = _n(200)
    warmup = max(5, total // 10)
    grew = _leak_growth(step, warmup=warmup, measured=total - warmup)
    assert grew <= LEAK_BUDGET, f"mixed soak leaked {grew} bytes"
