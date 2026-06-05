"""Multi-GPU + async correctness: every visible ordinal must be a real, isolated
device, and ResNet-18 must train fully async (no per-step syncs) on any of them.

The harness keeps AMD_SERIALIZE_KERNEL UNSET (see run_ab.sh): a serialized run
syncs after every launch and the async claim verifies nothing.

Regression background (the bug class these tests pin down): a runtime whose
device entries all alias ordinal 0 passes single-GPU smoke tests perfectly, but
per-device state (allocation pools, event timing, peer lookups) collapses onto
device 0. The first symptoms in the wild were hipErrorNotFound from hipBLASLt
on non-default ordinals, then GPU memory faults the moment two ordinals were
touched by the same process. The repro is a one-liner per device — kernels on
ordinal 0 then ordinal 1; the second device faults. Run cheap, fail loud.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from _util import dev, dev_randint

cuda = pytest.mark.usefixtures("torch_cuda")


@pytest.fixture(autouse=True)
def _restore_current_device():
    """These tests hop ordinals; never leak that onto later tests."""
    before = torch.cuda.current_device() if torch.cuda.is_available() else 0
    yield
    if torch.cuda.is_available():
        torch.cuda.set_device(before)


def _ordinals():
    """Ordinals worth testing without sweeping all 8 GPUs: first, second, last."""
    n = torch.cuda.device_count()
    return sorted({0, min(1, n - 1), n - 1})


def _multi_gpu_or_skip():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 visible GPUs")


@cuda
def test_async_config_and_device_count(ab):
    """Two preconditions for the rest of this module: (1) AMD_SERIALIZE_KERNEL
    is not set (async means deep queues actually form), and (2) the backend
    enumerates the SAME devices as native — a backend that drops to one device
    would otherwise quietly skip-green every multi-GPU test below.
    """
    if os.environ.get("HRX_AB_SERIALIZE", "0") == "1":
        pytest.skip("HRX_AB_SERIALIZE=1: serialized debug run, async asserts off")
    assert os.environ.get("AMD_SERIALIZE_KERNEL", "0") in ("", "0"), \
        "AMD_SERIALIZE_KERNEL is set: this run serializes; async tests verify nothing"
    ab.check(torch.tensor([torch.cuda.device_count()]),
             name="device_count", atol=0, rtol=0)


@cuda
def test_cross_device_sequential_kernels(ab):
    """Kernels on ordinal 0 then on each later ordinal, one process — the second
    device faulted when device entries aliased ordinal 0 (memory placed on the
    wrong GPU)."""
    _multi_gpu_or_skip()
    sums = []
    for d in _ordinals():
        torch.cuda.set_device(d)
        a = dev(2048, 2048, seed=60, device=f"cuda:{d}")
        for _ in range(20):
            a = a.relu() - 0.01
        torch.cuda.synchronize(d)
        sums.append(a.sum().cpu())
    # Identical seeded input + identical math => identical result per device.
    ab.check(torch.stack(sums), name="per_device_sums", atol=2e-2, rtol=2e-2)


@cuda
def test_event_elapsed_positive_per_device():
    """hipEventElapsedTime must be > 0 across real GPU work on every ordinal
    (zero elapsed broke MIOpen's tuner: 'Invalid elapsed time' + bad solvers).
    Single-GPU machines still cover ordinal 0."""
    for d in _ordinals():
        torch.cuda.set_device(d)
        a = dev(2048, 2048, seed=61, device=f"cuda:{d}")
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(10):
            a = a.relu()
        end.record()
        torch.cuda.synchronize(d)
        elapsed = start.elapsed_time(end)
        assert elapsed > 0.0, f"ord{d}: elapsed_time={elapsed} (must be > 0)"


def _train_resnet18_async(device_index, *, warmup=3, steps=10):
    """Train with NO per-step sync — kernels queue/overlap; one final sync.
    This is the async path a serialized smoke never exercises."""
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    torch.cuda.set_device(device_index)
    d = f"cuda:{device_index}"
    torch.manual_seed(0)
    model = resnet18(weights=None).to(d).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = dev(4, 3, 224, 224, seed=34, device=d)
    y = dev_randint(1000, 4, seed=35, device=d)

    losses = []
    for _ in range(warmup + steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    torch.cuda.synchronize(device_index)
    return torch.stack(losses)[warmup:]


@cuda
def test_resnet18_train_async(ab):
    losses = _train_resnet18_async(0)
    assert torch.isfinite(losses).all(), f"NaN/Inf: {losses.tolist()}"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses.tolist()}"
    ab.check(losses, name="loss_curve", atol=5e-2, rtol=5e-2)


@cuda
def test_resnet18_train_async_nondefault_ordinal(ab):
    _multi_gpu_or_skip()
    losses = _train_resnet18_async(torch.cuda.device_count() - 1)
    assert torch.isfinite(losses).all(), f"NaN/Inf: {losses.tolist()}"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses.tolist()}"
    ab.check(losses, name="loss_curve", atol=5e-2, rtol=5e-2)


@cuda
def test_devices_physically_distinct():
    """Each ordinal must be a distinct physical device. Reported identity and
    free-memory counters can both be synthesized per-ordinal while every byte
    actually lands on device 0, so assert physical coexistence instead: >50%
    of one card's VRAM on EACH of two ordinals cannot fit on one aliased card.
    Aliased placement either OOMs/faults or the buffers overlap and the
    pattern check fails."""
    _multi_gpu_or_skip()
    d0, d1 = 0, torch.cuda.device_count() - 1
    frac_bytes = int(torch.cuda.get_device_properties(0).total_memory * 0.55)
    n = frac_bytes // 4  # fp32 elements
    try:
        a = torch.full((n,), 1.0, dtype=torch.float32, device=f"cuda:{d0}")
        b = torch.full((n,), 2.0, dtype=torch.float32, device=f"cuda:{d1}")
        torch.cuda.synchronize(d0)
        torch.cuda.synchronize(d1)
        # Spot-check both ends of each buffer; overlap clobbers one of them.
        for t, want, d in ((a, 1.0, d0), (b, 2.0, d1)):
            head = float(t[:1024].sum())
            tail = float(t[-1024:].sum())
            assert head == 1024 * want and tail == 1024 * want, (
                f"ord{d}: buffer corrupt (head={head} tail={tail}); buffers "
                f"on 'different devices' overlap — devices alias"
            )
        del a, b
    finally:
        for d in (d0, d1):
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
