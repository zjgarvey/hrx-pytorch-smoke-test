"""Memory transfers / allocator — copies must be exact; allocs must not crash."""

import pytest
import torch

from _util import dev

cuda = pytest.mark.usefixtures("torch_cuda")


@cuda
def test_h2d_d2h_roundtrip(ab):
    cpu = torch.arange(1000, dtype=torch.float32)
    d = cpu.to("cuda")
    back = d.cpu()
    assert torch.equal(back, cpu), "H2D->D2H roundtrip changed the data"
    ab.check(d, atol=0, rtol=0)


@cuda
def test_d2d_copy(ab):
    a = dev(4096, seed=20)
    b = torch.empty_like(a)
    b.copy_(a)
    torch.cuda.synchronize()
    ab.check(b, atol=0, rtol=0)


@cuda
def test_fill_zeros_ones_full(ab):
    z = torch.zeros(1024, device="cuda")
    o = torch.ones(1024, device="cuda")
    f = torch.full((1024,), 3.5, device="cuda")
    torch.cuda.synchronize()
    ab.check(torch.cat([z, o, f]), atol=0, rtol=0)


@cuda
def test_noncontiguous_then_contiguous(ab):
    x = dev(64, 128, seed=21)
    t = x.t().contiguous()  # transpose (strided) then materialize
    ab.check(t, atol=0, rtol=0)


@cuda
def test_large_alloc_free_no_crash(torch_cuda):
    torch = torch_cuda
    # ~256 MiB churn to exercise the allocator; must not crash or OOM.
    bufs = [torch.empty(64 * 1024 * 1024 // 4, device="cuda") for _ in range(4)]
    for b in bufs:
        b.fill_(1.0)
    torch.cuda.synchronize()
    del bufs
    torch.cuda.empty_cache()
