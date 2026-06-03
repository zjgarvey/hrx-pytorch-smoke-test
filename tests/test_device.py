"""Device enumeration / properties — invariants that must hold under both backends."""

import pytest

cuda = pytest.mark.usefixtures("torch_cuda")


@cuda
def test_cuda_available(torch_cuda):
    assert torch_cuda.cuda.is_available()


@cuda
def test_device_count_positive(torch_cuda):
    assert torch_cuda.cuda.device_count() >= 1


@cuda
def test_device_name_is_amd(torch_cuda):
    name = torch_cuda.cuda.get_device_name(0)
    assert any(k in name for k in ("AMD", "Instinct", "MI", "gfx")), (
        f"unexpected device name: {name!r}"
    )


@cuda
def test_device_properties_sane(torch_cuda):
    p = torch_cuda.cuda.get_device_properties(0)
    assert p.total_memory > (1 << 30)  # > 1 GiB
    assert p.multi_processor_count > 0


# NOTE: HRX currently reports a synthetic PCI identity (device ordinal, with
# domain/device = 0) rather than the real BDF. That is harmless for compute but
# blocks RCCL topology discovery (and therefore DDP/FSDP). It is tracked
# separately and intentionally not asserted here.
