# %% [markdown]
# # 02 — Layer transferability: the AnB/BnB protocol in miniature
#
# A small-scale replication of Yosinski, Clune, Bengio & Lipson, *How
# transferable are features in deep neural networks?* (NeurIPS 2014).
#
# **Question.** When we copy the first k layers of a trained network into a
# new one, which layers carry *general* features (useful for other tasks) and
# which carry *task-specific* ones? Yosinski et al. answered this with two
# 500-class halves of ImageNet; we use two 5-class halves of MNIST — task
# **A = digits 0–4**, task **B = digits 5–9** (labels remapped to 0..4).
#
# **What the literature says.**
# - Early layers learn general features (Gabor-like edges at ImageNet scale;
#   strokes here) and late layers specialize to the training task; frozen
#   cross-task transfer therefore degrades as more layers are copied
#   (Yosinski et al. 2014).
# - Yosinski et al. identified a *second*, subtler failure mode:
#   **co-adaptation**. Even same-task transfer (BnB) dips at middle k because
#   neighboring layers develop fragile joint solutions that retraining half
#   of them cannot rediscover.
# - Fine-tuning the copied layers repairs both failure modes and can even
#   improve on the baseline (Yosinski et al. 2014).
# - The modern view (Neyshabur, Sedghi & Zhang, *What is being transferred in
#   transfer learning?*, NeurIPS 2020) is that transfer benefits come largely
#   from low-level statistics and from landing in the same flat loss basin —
#   consistent with early-layer generality.
#
# **The naming convention** (theirs, kept here). A net whose first n layers
# are copied from a **B**-trained source and then retrained on **B** is
# **BnB** — a *selffer*: same task on both sides, so any drop is pure
# co-adaptation damage, not feature mismatch. Copying from an **A**-trained
# source instead gives **AnB** — a true *transfer*. A trailing **+** (BnB+,
# AnB+) means the copied layers are **fine-tuned** rather than frozen. Our
# unit of "layer" is an encoder block, and we plot against k ∈ {1..5} copied
# blocks.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "tllab").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

import tllab
from tllab import analysis, data, models, train

tllab.setup()

# %% [markdown]
# ## Tasks and source models
#
# The two disjoint digit halves, plus one standard source model per half
# (shared checkpoints: 20k examples, 6 epochs, seed 0 — the suite defaults).

# %%
task_a = data.get_task("mnist:0-4")
task_b = data.get_task("mnist:5-9")
print(task_a)
print(task_b)

srcA = train.train_or_load_source("mnist:0-4")
srcB = train.train_or_load_source("mnist:5-9")

# %% [markdown]
# ## The experiment
#
# Target regime: **n = 1000** training examples of B, **12 epochs**, batch
# 128, **lr 1e-3 for every condition** — including the fine-tuned ones.
# Notebook 01 used a gentler lr for fine-tuning; here we deliberately follow
# Yosinski's protocol and hold the entire training budget identical across
# conditions, so the *only* difference between curves is where the first k
# blocks came from and whether they may move. 3 seeds; the B-subset is
# re-drawn per seed and shared by all conditions within that seed.
#
# | condition | blocks 1..k init | blocks 1..k trained? | blocks k+1..5 + head |
# |---|---|---|---|
# | scratch | random | yes | random, trained |
# | BnB | copied from srcB | frozen | random, trained |
# | BnB+ | copied from srcB | fine-tuned | random, trained |
# | AnB | copied from srcA | frozen | random, trained |
# | AnB+ | copied from srcA | fine-tuned | random, trained |
#
# That is 3 seeds × (1 scratch + 5 k-values × 4 conditions) = 63 runs at
# roughly 8 s each.

# %%
N_B = tllab.q(1000, 100)
EPOCHS = tllab.q(12, 2)
CONDITIONS = [("BnB", True), ("BnB+", False), ("AnB", True), ("AnB+", False)]


def yosinski_grid() -> pd.DataFrame:
    rows = []
    for seed in tllab.seeds():
        sub = data.subset(task_b, N_B, seed)

        m = models.new_cnn(task_b.n_classes, seed)
        acc = train.fit(m, sub, epochs=EPOCHS, seed=seed).final_acc
        rows.append({"condition": "scratch", "k": 0, "seed": seed, "acc": acc})
        print(f"  scratch      seed={seed}: {acc:.3f}")

        for k in range(1, 6):
            for cond, frozen in CONDITIONS:
                src = srcA if cond.startswith("A") else srcB
                m = models.transfer_first_k(src, k, task_b.n_classes, seed)
                if frozen:
                    models.freeze_blocks(m, k)
                acc = train.fit(m, sub, epochs=EPOCHS, seed=seed).final_acc
                rows.append({"condition": cond, "k": k, "seed": seed, "acc": acc})
                print(f"  {cond:<5s} k={k} seed={seed}: {acc:.3f}")
    return pd.DataFrame(rows)


df = train.cached_sweep("02_yosinski_grid", yosinski_grid)
scratch_mean = df[df.condition == "scratch"]["acc"].mean()
print(f"scratch baseline (mean over seeds): {scratch_mean:.3f}")
df[df.condition != "scratch"].pivot_table(
    index="k", columns="condition", values="acc").round(3)

# %%
curves = df[df.condition != "scratch"]
plt.figure(figsize=(6.8, 4.5))
ax = plt.gca()
ax.axhline(scratch_mean, color="#898781", linestyle="--", linewidth=1.2,
           label=f"scratch on B, same budget ({scratch_mean:.3f})")
analysis.plot_curves(curves, x="k", y="acc", hue="condition", ax=ax,
                     ylabel="task-B test accuracy",
                     title="Yosinski AnB/BnB in miniature "
                           "(frozen vs fine-tuned, digits 0–4 → 5–9)")
ax.set_xticks(range(1, 6))
ax.set_xlabel("k = number of encoder blocks copied from the source")
analysis.savefig("02_yosinski_curves")

# %% [markdown]
# ## Takeaways
#
# - **Early features are general.** Frozen AnB sits near the scratch baseline
#   at k=1–2: the first conv blocks of a digits-0–4 network detect strokes
#   and curves that serve digits 5–9 just as well.
# - **Specificity concentrates at the top.** Frozen AnB actually *rises*
#   through k=2–4: those copied mid-blocks were trained on 20k examples of
#   task A, and generic stroke features learned from 20k images beat what
#   1000 examples of B can grow from scratch — the source's data advantage
#   outweighs any task mismatch. Only at k=5, when the *entire* encoder is
#   locked to the other digit half and just a linear head can compensate,
#   does the curve fall off a cliff (~10 points below baseline). In a
#   network this small, the generality-to-specificity gradient is real but
#   compressed into the final block; in deeper networks it spreads across
#   many layers.
# - **Fine-tuning repairs the damage.** AnB+ tracks the baseline at every k
#   (sometimes a touch above): letting the copied blocks move lets the
#   representation re-specialize, exactly as in the original and as
#   re-interpreted by Neyshabur et al. (2020) — the copied weights mostly
#   serve as a good starting basin.
# - **The famous co-adaptation dip is small or absent here — honestly.**
#   In the original, frozen BnB dropped at middle k because splitting
#   co-adapted neighboring layers hurt even without any task shift. Our
#   5-block MNIST CNN shows little of this, for two reasons: (1) small,
#   shallow networks on easy data have far fewer fragile co-adaptations than
#   8-layer AlexNet on ImageNet; (2) our srcB saw 20k examples of B while the
#   retraining sees only 1000, so frozen B-features carry an information
#   *advantage* that can push BnB above the scratch line and mask any dip.
#   The robust take-homes at this scale are the generality-vs-specificity
#   gradient (AnB) and fine-tuning recovery (AnB+/BnB+), not the dip.
#
# **References.** Yosinski, Clune, Bengio & Lipson, *How transferable are
# features in deep neural networks?* (NeurIPS 2014). Neyshabur, Sedghi &
# Zhang, *What is being transferred in transfer learning?* (NeurIPS 2020).

# %%
if not tllab.config.QUICK:
    g = df[df.condition != "scratch"].groupby(["condition", "k"])["acc"].mean()
    worst_anbp = min(g["AnB+"][k] - scratch_mean for k in range(1, 6))
    analysis.claim("AnB+ recovers to within 2 pts of scratch at every k",
                   worst_anbp > -0.02, f"worst gap {worst_anbp:+.3f}")
    analysis.claim("AnB frozen degrades with depth (k=5 < k=1)",
                   g["AnB"][5] < g["AnB"][1],
                   f"k=5 {g['AnB'][5]:.3f} vs k=1 {g['AnB'][1]:.3f}")
    analysis.claim("AnB frozen at k=5 falls below scratch",
                   g["AnB"][5] < scratch_mean,
                   f"{g['AnB'][5]:.3f} vs scratch {scratch_mean:.3f}")
    worst_bnbp = min(g["BnB+"][k] - scratch_mean for k in range(1, 6))
    analysis.claim("BnB+ stays within 2 pts of scratch at every k",
                   worst_bnbp > -0.02, f"worst gap {worst_bnbp:+.3f}")
