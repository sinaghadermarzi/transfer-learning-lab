"""Datasets and the string-keyed task registry.

Base datasets are MNIST and Fashion-MNIST (the only dataset hosts reachable in
this environment); every other task is derived from them, so the whole suite
runs from two small downloads.

Registry keys (also used as checkpoint keys by tllab.train):
    mnist, fashion                  10-class base tasks
    <base>:0-4 / <base>:5-9         5-class split, labels remapped to 0..4
    <base>:rot90                    inputs rotated 90 degrees, same labels
    <base>:permuted                 fixed random pixel permutation, same labels
    <base>:randlabel                labels replaced with a fixed random shuffle
    <base>:rotpred                  4-way rotation-angle prediction (0/90/180/270)

All tasks are plain in-memory float32 tensors (no DataLoaders) — on a 4-core
CPU box this is a large speed win and the full datasets fit in RAM easily.
"""
import re
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from . import config

_STATS = {"mnist": (0.1307, 0.3081), "fashion": (0.2860, 0.3530)}

FASHION_CLASSES = ["tshirt", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "boot"]


@dataclass
class Task:
    name: str
    n_classes: int
    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor

    def __repr__(self):
        return (f"Task({self.name!r}, {self.n_classes} classes, "
                f"train={len(self.train_y)}, test={len(self.test_y)})")


_base_cache: dict[tuple[str, bool], tuple[Tensor, Tensor]] = {}
_task_cache: dict[str, Task] = {}


def get_base(name: str, train: bool) -> tuple[Tensor, Tensor]:
    """Normalized (x, y) tensors for a base dataset split, cached on disk."""
    key = (name, train)
    if key in _base_cache:
        return _base_cache[key]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.DATA_DIR / f"{name}_{'train' if train else 'test'}.pt"
    if cache.exists():
        x, y = torch.load(cache, weights_only=True)
    else:
        from torchvision import datasets

        cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[name]
        last_err = None
        for attempt in range(3):
            try:
                ds = cls(root=str(config.DATA_DIR), train=train, download=True)
                break
            except Exception as e:  # transient network failures behind the proxy
                last_err = e
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"could not download {name}") from last_err
        mean, std = _STATS[name]
        x = ((ds.data.float() / 255.0) - mean) / std
        x = x.unsqueeze(1).contiguous()
        y = ds.targets.clone()
        torch.save((x, y), cache)
    _base_cache[key] = (x, y)
    return x, y


def _rotpred(x: Tensor, gen_seed: int) -> tuple[Tensor, Tensor]:
    """Rotate each image by a random multiple of 90 degrees; label = multiple."""
    g = torch.Generator().manual_seed(gen_seed)
    k = torch.randint(0, 4, (len(x),), generator=g)
    out = x.clone()
    for rot in range(1, 4):
        m = k == rot
        out[m] = torch.rot90(x[m], rot, dims=(-2, -1))
    return out, k


def get_task(key: str) -> Task:
    """Build (and cache) a Task from a registry key."""
    if key in _task_cache:
        return _task_cache[key]

    base, _, variant = key.partition(":")
    if base not in _STATS:
        raise KeyError(f"unknown base dataset {base!r} in task key {key!r}")
    train_x, train_y = get_base(base, train=True)
    test_x, test_y = get_base(base, train=False)
    n_classes = 10

    if variant == "":
        pass
    elif m := re.fullmatch(r"(\d)-(\d)", variant):
        lo, hi = int(m.group(1)), int(m.group(2))
        tr = (train_y >= lo) & (train_y <= hi)
        te = (test_y >= lo) & (test_y <= hi)
        train_x, train_y = train_x[tr], train_y[tr] - lo
        test_x, test_y = test_x[te], test_y[te] - lo
        n_classes = hi - lo + 1
    elif variant == "rot90":
        train_x = torch.rot90(train_x, 1, dims=(-2, -1)).contiguous()
        test_x = torch.rot90(test_x, 1, dims=(-2, -1)).contiguous()
    elif variant == "permuted":
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(28 * 28, generator=g)
        train_x = train_x.flatten(1)[:, perm].view(-1, 1, 28, 28).contiguous()
        test_x = test_x.flatten(1)[:, perm].view(-1, 1, 28, 28).contiguous()
    elif variant == "randlabel":
        g = torch.Generator().manual_seed(0)
        train_y = train_y[torch.randperm(len(train_y), generator=g)]
        g = torch.Generator().manual_seed(1)
        test_y = test_y[torch.randperm(len(test_y), generator=g)]
    elif variant == "rotpred":
        train_x, train_y = _rotpred(train_x, gen_seed=0)
        test_x, test_y = _rotpred(test_x, gen_seed=1)
        n_classes = 4
    else:
        raise KeyError(f"unknown task variant {variant!r} in key {key!r}")

    task = Task(key, n_classes, train_x, train_y, test_x, test_y)
    _task_cache[key] = task
    return task


def stratified_indices(y: Tensor, n: int, seed: int) -> Tensor:
    """Indices of a class-balanced sample of size n (remainder spread evenly)."""
    g = torch.Generator().manual_seed(seed)
    classes = torch.unique(y)
    quota, extra = divmod(n, len(classes))
    picked = []
    order = torch.randperm(len(classes), generator=g)
    for rank, ci in enumerate(order.tolist()):
        c = classes[ci]
        pool = torch.nonzero(y == c, as_tuple=True)[0]
        take = quota + (1 if rank < extra else 0)
        sel = pool[torch.randperm(len(pool), generator=g)[:take]]
        picked.append(sel)
    idx = torch.cat(picked)
    return idx[torch.randperm(len(idx), generator=g)]


def subset(task: Task, n: int, seed: int) -> Task:
    """Stratified train subsample of size n; test set untouched."""
    n = min(n, len(task.train_y))
    idx = stratified_indices(task.train_y, n, seed)
    return Task(f"{task.name}[n={n},s={seed}]", task.n_classes,
                task.train_x[idx], task.train_y[idx], task.test_x, task.test_y)


_test_subset_cache: dict[tuple[str, int], tuple[Tensor, Tensor]] = {}


def test_subset(task: Task, n: int = 2000) -> tuple[Tensor, Tensor]:
    """Fixed (seed-0) test subsample used for cheap per-epoch tracking."""
    key = (task.name.split("[")[0], n)
    if key not in _test_subset_cache:
        n = min(n, len(task.test_y))
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(task.test_y), generator=g)[:n]
        _test_subset_cache[key] = (task.test_x[idx], task.test_y[idx])
    return _test_subset_cache[key]


@dataclass
class PairTask:
    """Two-digit task for the modular-composition notebook: y = digit_a + digit_b."""
    name: str
    n_classes: int
    xa: Tensor
    xb: Tensor
    y: Tensor

    def __repr__(self):
        return f"PairTask({self.name!r}, {self.n_classes} classes, n={len(self.y)})"


def make_pair_sum(n_pairs: int, seed: int, split: str = "train") -> PairTask:
    """Random MNIST digit pairs labeled only with their sum (0..18)."""
    x, y = get_base("mnist", train=(split == "train"))
    g = torch.Generator().manual_seed(seed + (100_000 if split == "test" else 0))
    ia = torch.randint(len(y), (n_pairs,), generator=g)
    ib = torch.randint(len(y), (n_pairs,), generator=g)
    return PairTask(f"pairsum-{split}", 19, x[ia], x[ib], y[ia] + y[ib])
