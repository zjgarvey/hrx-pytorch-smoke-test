"""Training works: real fwd/bwd/optimizer loops; loss decreases AND matches native.

Model init uses torch's (CPU) RNG seeded deterministically, and inputs come from
the seeded CPU helper, so native and HRX share identical weights+inputs — the
loss trajectory should match (within accumulation tolerance) and must decrease.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from _util import dev, dev_randint

cuda = pytest.mark.usefixtures("torch_cuda")


def _train(model_fn, x, y, *, steps=5, lr=0.1, momentum=0.0, seed=0):
    torch.manual_seed(seed)  # deterministic parameter init (CPU RNG)
    model = model_fn().to("cuda").train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    torch.cuda.synchronize()
    return torch.stack(losses)


@cuda
def test_mlp_train(ab):
    x = dev(64, 64, seed=30)
    y = dev_randint(10, 64, seed=31)
    losses = _train(
        lambda: nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)),
        x, y, steps=8, lr=0.1,
    )
    assert losses[-1] < losses[0], f"loss did not decrease: {losses.tolist()}"
    ab.check(losses, name="loss_curve", atol=1e-2, rtol=1e-2)


@cuda
def test_cnn_train(ab):
    x = dev(8, 3, 32, 32, seed=32)
    y = dev_randint(10, 8, seed=33)
    losses = _train(
        lambda: nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
        ),
        x, y, steps=5, lr=0.1,
    )
    assert losses[-1] < losses[0], f"loss did not decrease: {losses.tolist()}"
    ab.check(losses, name="loss_curve", atol=2e-2, rtol=2e-2)


@cuda
def test_resnet18_train(ab):
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    x = dev(4, 3, 224, 224, seed=34)
    y = dev_randint(1000, 4, seed=35)
    losses = _train(lambda: resnet18(weights=None), x, y, steps=3, lr=0.01, momentum=0.9)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses.tolist()}"
    # ResNet accumulates more numeric difference across its depth; looser tol.
    ab.check(losses, name="loss_curve", atol=5e-2, rtol=5e-2)
