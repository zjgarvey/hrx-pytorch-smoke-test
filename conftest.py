"""Pytest harness for "does pytorch work with hrx-on-hip?" — native ROCm A/B.

The HRX HIP binding and stock ROCm cannot coexist in one process (the
`libamdhip64.so.7` the venv loads is process-global and chosen by a symlink), so
correctness is checked by running the SAME suite twice and comparing:

  1. RECORD pass  (`HRX_AB_MODE=record`, backend = native ROCm via use_rocm.sh):
     every `ab.check(...)` saves its result tensor as a golden.
  2. COMPARE pass (`HRX_AB_MODE=compare`, backend = HRX via use_hrx.sh):
     every `ab.check(...)` loads the native golden and asserts torch.allclose.

So any HRX-vs-native divergence (or crash) is a test failure. `run_ab.sh`
orchestrates the two passes and the backend swap. A single `pytest` invocation
also works (compare mode by default; needs goldens to exist).
"""

import os
import pathlib
import sys

import pytest

# Make the repo root importable so tests can `import _util`.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

AB_MODE = os.environ.get("HRX_AB_MODE", "compare").lower()
GOLDEN_DIR = pathlib.Path(
    os.environ.get("HRX_AB_GOLDEN_DIR", str(pathlib.Path(__file__).parent / "goldens"))
)


def active_backend():
    """Best-effort detection of which libamdhip64 is live, from /proc/self/maps.

    HRX maps its own `libhrx.so`; stock ROCm does not.
    """
    try:
        maps = pathlib.Path("/proc/self/maps").read_text()
    except OSError:
        return "unknown"
    if "libhrx.so" in maps:
        return "hrx"
    if "libamdhip64" in maps:
        return "native"
    return "unknown"


def pytest_report_header(config):
    return [
        f"HRX A/B mode: {AB_MODE}   golden_dir: {GOLDEN_DIR}",
        f"live libamdhip64 backend (from /proc/self/maps): {active_backend()}",
    ]


@pytest.fixture(scope="session")
def torch_cuda():
    """Session torch handle; skips the whole suite if no GPU is visible."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False — no GPU backend")
    return torch


class _ABComparator:
    """Records goldens on the native pass; asserts equality on the HRX pass."""

    def __init__(self, request):
        # Stable, filesystem-safe key prefix per test.
        self.prefix = (
            request.node.nodeid.replace("/", "_").replace("::", "__").replace(" ", "_")
        )
        self._auto = 0

    def _path(self, name):
        if name is None:
            self._auto += 1
            name = f"v{self._auto}"
        return GOLDEN_DIR / f"{self.prefix}__{name}.pt"

    def check(self, tensor, name=None, *, atol=1e-4, rtol=1e-4):
        """Record (native) or compare (HRX) a result tensor against the golden."""
        import torch

        got = tensor.detach().to("cpu")
        path = self._path(name)
        if AB_MODE == "record":
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(got, path)
            return got
        if not path.exists():
            pytest.fail(
                f"no native golden at {path.name}; run the record pass on native "
                f"ROCm first (run_ab.sh does both)"
            )
        golden = torch.load(path)
        try:
            torch.testing.assert_close(got, golden, atol=atol, rtol=rtol,
                                       equal_nan=True)
        except AssertionError as e:
            pytest.fail(
                f"HRX result diverges from the native golden "
                f"'{path.name}' (atol={atol}, rtol={rtol}):\n{e}"
            )
        return got


@pytest.fixture
def ab(request):
    return _ABComparator(request)
