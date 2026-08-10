"""Training, evaluation, probing, checkpointing, and sweep caching.

Two caching layers keep re-runs cheap and notebooks independently re-runnable:
- train_or_load_source(): source models saved under artifacts/checkpoints/
- cached_sweep(): every experiment sweep saved as a CSV under artifacts/results/
"""
import math
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import config, data, models


@dataclass
class FitResult:
    final_acc: float          # accuracy on the FULL test set at end of training
    history: pd.DataFrame     # per-epoch: train_loss, test_acc (2k subset), extras
    model: nn.Module
    wall_s: float


def _batches(n: int, batch_size: int, generator: torch.Generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


def _acc(model: nn.Module, x: Tensor, y: Tensor, batch_size: int = 1024,
         head: str | None = None) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            xb = x[i:i + batch_size]
            out = model(xb) if head is None else model(xb, head)
            correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / len(y)


def evaluate(model: nn.Module, task: data.Task) -> float:
    """Accuracy on the task's full test set."""
    return _acc(model, task.test_x, task.test_y)


def evaluate_head(net: nn.Module, task: data.Task, head: str) -> float:
    """Accuracy of one head of a MultiHeadNet on the task's full test set."""
    return _acc(net, task.test_x, task.test_y, head=head)


def fit(model: nn.Module, task: data.Task, *, epochs: int, lr: float = 1e-3,
        head_lr: float | None = None,
        batch_size: int = config.DEFAULT_BATCH, seed: int = 0,
        track: dict[str, Callable[[nn.Module], float]] | None = None,
        eval_every: int = 1, verbose: bool = False) -> FitResult:
    """Adam + cross-entropy on in-memory tensors; per-epoch eval on a 2k test
    subset plus any `track` callables (used e.g. for forgetting curves).

    head_lr enables the standard fine-tuning recipe of discriminative learning
    rates: the (fresh, randomly initialized) head trains at head_lr while the
    pretrained encoder moves gently at lr. Without it, a low fine-tuning lr
    starves the new head and understates transfer."""
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    if head_lr is not None and hasattr(model, "head"):
        head_ids = {id(p) for p in model.head.parameters()}
        groups = [
            {"params": [p for p in model.parameters()
                        if p.requires_grad and id(p) not in head_ids], "lr": lr},
            {"params": [p for p in model.head.parameters() if p.requires_grad],
             "lr": head_lr},
        ]
        opt = torch.optim.Adam(groups)
    else:
        opt = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=lr)
    tx, ty = data.test_subset(task)
    rows = []
    for ep in range(epochs):
        model.train()
        losses = []
        for idx in _batches(len(task.train_y), batch_size, g):
            loss = F.cross_entropy(model(task.train_x[idx]), task.train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % eval_every and ep != epochs - 1:
            continue
        row = {"epoch": ep + 1, "train_loss": sum(losses) / len(losses),
               "test_acc": _acc(model, tx, ty)}
        for name, fn in (track or {}).items():
            row[name] = fn(model)
        rows.append(row)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs}: loss={row['train_loss']:.3f} "
                  f"acc={row['test_acc']:.3f}")
    return FitResult(evaluate(model, task), pd.DataFrame(rows), model,
                     time.time() - t0)


# ---------------------------------------------------------------- probing

_emb_cache: dict[str, tuple[Tensor, Tensor]] = {}


def _partial_features(enc: nn.Module, x: Tensor, depth: int) -> Tensor:
    """Features after the first `depth` encoder blocks; conv maps are pooled
    to 4x4 and flattened (the standard mid-layer probing protocol)."""
    h = x
    for b in enc.blocks[:depth]:
        h = b(h)
    if h.dim() == 4:
        h = F.adaptive_avg_pool2d(h, 4).flatten(1)
    return h


def embeddings(encoder_like: nn.Module, task: data.Task,
               cache_key: str | None = None,
               depth: int | None = None) -> tuple[Tensor, Tensor]:
    """Embeddings of the task's train/test inputs (cached per cache_key).

    depth=None uses the full encoder (the 128-d embedding); depth=k probes
    the representation after block k instead."""
    key = f"{cache_key}|{task.name}|d{depth}" if cache_key else None
    if key and key in _emb_cache:
        return _emb_cache[key]
    enc = encoder_like.encoder if hasattr(encoder_like, "encoder") else encoder_like
    enc.eval()
    d = depth if depth is not None else len(enc.blocks)

    def run(x):
        with torch.no_grad():
            return torch.cat([_partial_features(enc, x[i:i + 1024], d)
                              for i in range(0, len(x), 1024)])

    out = (run(task.train_x), run(task.test_x))
    if key:
        _emb_cache[key] = out
    return out


def linear_probe(encoder_like: nn.Module, task: data.Task, *, n_train: int,
                 seed: int, epochs: int = 60, lr: float = 1e-2,
                 cache_key: str | None = None, depth: int | None = None) -> float:
    """Train a linear head on frozen precomputed embeddings; return test acc."""
    emb_tr, emb_te = embeddings(encoder_like, task, cache_key=cache_key,
                                depth=depth)
    idx = data.stratified_indices(task.train_y, min(n_train, len(task.train_y)), seed)
    x, y = emb_tr[idx], task.train_y[idx]
    config.set_seed(seed)
    head = nn.Linear(emb_tr.shape[1], task.n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for bidx in _batches(len(y), 256, g):
            loss = F.cross_entropy(head(x[bidx]), y[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float((head(emb_te).argmax(1) == task.test_y).float().mean())


# ------------------------------------------------------------ checkpoints

def train_or_load_source(task_key: str, *, n_train: int | None = None,
                         epochs: int | None = None, seed: int = 0,
                         lr: float = 1e-3) -> models.SmallCNN:
    """The shared-checkpoint mechanism: any notebook can ask for a source model
    by task key and gets a cached checkpoint if a previous notebook trained it."""
    n_train = n_train if n_train is not None else config.q(20_000, 2_000)
    epochs = epochs if epochs is not None else config.q(6, 1)
    task = data.get_task(task_key)
    n_train = min(n_train, len(task.train_y))
    fname = (f"{task_key.replace(':', '_')}__smallcnn"
             f"__n{n_train}__e{epochs}__s{seed}.pt")
    path = config.CKPT_DIR / fname
    if path.exists():
        ck = torch.load(path, weights_only=True)
        model = models.SmallCNN(task.n_classes)
        model.load_state_dict(ck["state_dict"])
        print(f"[ckpt cached] {task_key}: test_acc={ck['meta']['test_acc']:.3f}")
        return model
    model = models.new_cnn(task.n_classes, seed)
    res = fit(model, data.subset(task, n_train, seed), epochs=epochs, lr=lr, seed=seed)
    meta = {"task_key": task_key, "n_train": n_train, "epochs": epochs,
            "seed": seed, "lr": lr, "test_acc": res.final_acc,
            "wall_s": round(res.wall_s, 1), "torch": str(torch.__version__)}
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    print(f"[ckpt trained {res.wall_s:.0f}s] {task_key}: "
          f"test_acc={res.final_acc:.3f}")
    return model


# ------------------------------------------------------------- multi-task

def _trunk_grad_cosine(net: models.MultiHeadNet, tasks: dict[str, data.Task],
                       generator: torch.Generator, batch: int = 256) -> float:
    """Cosine similarity between the two tasks' loss gradients on the trunk."""
    enc_params = [p for p in net.encoder.parameters()]
    vecs = []
    for name, task in tasks.items():
        n = len(task.train_y)
        idx = torch.randint(n, (min(batch, n),), generator=generator)
        loss = F.cross_entropy(net(task.train_x[idx], name), task.train_y[idx])
        grads = torch.autograd.grad(loss, enc_params, allow_unused=True)
        vecs.append(torch.cat([gr.flatten() for gr in grads if gr is not None]))
    return float(F.cosine_similarity(vecs[0], vecs[1], dim=0))


def fit_multitask(tasks: dict[str, data.Task], *, main: str, epochs: int,
                  lr: float = 1e-3, batch_size: int = config.DEFAULT_BATCH,
                  seed: int = 0, loss_weights: dict[str, float] | None = None,
                  grad_log_every: int = 20,
                  init_encoder: nn.Module | None = None,
                  track: dict[str, Callable[[nn.Module], float]] | None = None,
                  ) -> tuple[FitResult, pd.DataFrame]:
    """Train a shared trunk on all tasks jointly (one batch per task per step).

    Steps per epoch follow the main task's size. init_encoder warm-starts the
    trunk from a pretrained encoder (used for rehearsal-style continual
    learning). Returns (result-on-main-task, gradient-cosine trace df) — the
    trace is empty unless exactly 2 tasks."""
    config.set_seed(seed)
    net = models.MultiHeadNet({k: t.n_classes for k, t in tasks.items()})
    if init_encoder is not None:
        src_enc = (init_encoder.encoder if hasattr(init_encoder, "encoder")
                   else init_encoder)
        net.encoder.load_state_dict(src_enc.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    weights = loss_weights or {k: 1.0 for k in tasks}
    g = torch.Generator().manual_seed(seed)
    gcos = torch.Generator().manual_seed(seed + 777)

    iters = {k: iter([]) for k in tasks}

    def next_idx(k):
        nonlocal iters
        idx = next(iters[k], None)
        if idx is None:
            iters[k] = _batches(len(tasks[k].train_y), batch_size, g)
            idx = next(iters[k])
        return idx

    main_task = tasks[main]
    tx, ty = data.test_subset(main_task)
    steps_per_epoch = math.ceil(len(main_task.train_y) / batch_size)
    rows, grad_rows = [], []
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        net.train()
        for _ in range(steps_per_epoch):
            opt.zero_grad()
            total = 0.0
            for name, task in tasks.items():
                idx = next_idx(name)
                loss = F.cross_entropy(net(task.train_x[idx], name),
                                       task.train_y[idx])
                total = total + weights[name] * loss
            total.backward()
            opt.step()
            if len(tasks) == 2 and step % grad_log_every == 0:
                net.eval()
                grad_rows.append({"step": step,
                                  "cos": _trunk_grad_cosine(net, tasks, gcos)})
                net.train()
            step += 1
        row = {"epoch": ep + 1, "test_acc": _acc(net, tx, ty, head=main)}
        for name, fn in (track or {}).items():
            row[name] = fn(net)
        rows.append(row)
    final = _acc(net, main_task.test_x, main_task.test_y, head=main)
    return (FitResult(final, pd.DataFrame(rows), net, time.time() - t0),
            pd.DataFrame(grad_rows))


# ------------------------------------------------------------- pair tasks

def evaluate_pair(net: models.PairSumNet, pair: data.PairTask,
                  batch_size: int = 1024) -> tuple[float, float]:
    """(accuracy, MAE) on a PairTask."""
    net.eval()
    correct, abs_err = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(pair.y), batch_size):
            pred = net(pair.xa[i:i + batch_size],
                       pair.xb[i:i + batch_size]).argmax(1)
            yb = pair.y[i:i + batch_size]
            correct += (pred == yb).sum().item()
            abs_err += (pred - yb).abs().sum().item()
    return correct / len(pair.y), abs_err / len(pair.y)


def fit_pair(net: models.PairSumNet, pair_train: data.PairTask,
             pair_test: data.PairTask, *, epochs: int, lr: float = 1e-3,
             batch_size: int = config.DEFAULT_BATCH, seed: int = 0) -> FitResult:
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=lr)
    rows = []
    for ep in range(epochs):
        net.train()
        losses = []
        for idx in _batches(len(pair_train.y), batch_size, g):
            loss = F.cross_entropy(net(pair_train.xa[idx], pair_train.xb[idx]),
                                   pair_train.y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        acc, mae = evaluate_pair(net, pair_test)
        rows.append({"epoch": ep + 1, "train_loss": sum(losses) / len(losses),
                     "test_acc": acc, "test_mae": mae})
    acc, _ = evaluate_pair(net, pair_test)
    return FitResult(acc, pd.DataFrame(rows), net, time.time() - t0)


# ------------------------------------------------------------ sweep cache

def cached_sweep(name: str, fn: Callable[[], pd.DataFrame],
                 force: bool = False) -> pd.DataFrame:
    """Run an experiment sweep once; afterwards re-runs load the saved CSV."""
    path = config.RESULTS_DIR / f"{name}.csv"
    if path.exists() and not force:
        print(f"[sweep cached] {path.name}")
        return pd.read_csv(path)
    t0 = time.time()
    df = fn()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[sweep computed {time.time() - t0:.0f}s] {path.name}")
    return df
