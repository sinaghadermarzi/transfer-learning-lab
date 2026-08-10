"""Models for the transfer-learning lab.

Deliberately BatchNorm-free: frozen-BN running-statistic subtleties would muddy
the freezing and layer-transfer experiments, and plain conv+ReLU keeps the
co-adaptation story clean. All experiments share the same 5-block encoder so
"transfer the first k layers" means the same thing everywhere.
"""
import torch
from torch import nn

from . import config

EMB_DIM = 128


def _blocks() -> nn.ModuleList:
    return nn.ModuleList([
        nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU()),
        nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)),
        nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()),
        nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)),
        nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                      nn.Linear(64 * 16, EMB_DIM), nn.ReLU()),
    ])


class Encoder(nn.Module):
    """5-block conv encoder: 1x28x28 -> 128-d embedding (~196k params)."""

    def __init__(self):
        super().__init__()
        self.blocks = _blocks()

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class SmallCNN(nn.Module):
    """Encoder + linear head; the standard model for every single-input task."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(EMB_DIM, n_classes)

    @property
    def blocks(self):
        return self.encoder.blocks

    def features(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encoder(x))


class MultiHeadNet(nn.Module):
    """Shared encoder trunk with one linear head per task (multi-task learning)."""

    def __init__(self, head_classes: dict[str, int]):
        super().__init__()
        self.encoder = Encoder()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(EMB_DIM, n) for name, n in head_classes.items()})

    def forward(self, x, head: str):
        return self.heads[head](self.encoder(x))


class PairSumNet(nn.Module):
    """Twin (weight-shared) encoder on two images + combiner head -> sum class."""

    def __init__(self, encoder: Encoder | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder()
        self.combiner = nn.Sequential(
            nn.Linear(2 * EMB_DIM, 128), nn.ReLU(), nn.Linear(128, 19))

    def forward(self, xa, xb):
        return self.combiner(torch.cat([self.encoder(xa), self.encoder(xb)], dim=1))


def new_cnn(n_classes: int, seed: int) -> SmallCNN:
    """Freshly initialized SmallCNN with a reproducible init."""
    config.set_seed(seed)
    return SmallCNN(n_classes)


def transfer_first_k(src: SmallCNN, k: int, n_classes: int, seed: int) -> SmallCNN:
    """Yosinski-style transfer: copy blocks[0:k] from src, re-init the rest."""
    m = new_cnn(n_classes, seed)
    for i in range(k):
        m.blocks[i].load_state_dict(src.blocks[i].state_dict())
    return m


def freeze_blocks(model: SmallCNN, k: int) -> SmallCNN:
    """Freeze the first k encoder blocks (requires_grad=False)."""
    for i in range(k):
        for p in model.blocks[i].parameters():
            p.requires_grad_(False)
    return model


def freeze_encoder(model) -> nn.Module:
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return model


def clone_encoder(src: SmallCNN | Encoder) -> Encoder:
    """Independent copy of a model's encoder (so fine-tuning can't mutate src)."""
    enc = Encoder()
    src_enc = src.encoder if hasattr(src, "encoder") else src
    enc.load_state_dict(src_enc.state_dict())
    return enc


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
