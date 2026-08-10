# transfer-learning-lab

Small, CPU-friendly notebook experiments that demonstrate **what is known about
transferability between tasks in neural networks**: when reusing a trained
network helps, when it hurts, and how tasks interact when learned together.

Everything runs on a laptop-class CPU in ~1–2 hours total. Two real datasets
(MNIST, Fashion-MNIST) plus derived task variants, one shared ~200k-parameter
CNN everywhere, and every experiment compares against a from-scratch baseline.

## The notebooks

| # | Notebook | Question it answers |
|---|----------|---------------------|
| 00 | [setup & baselines](notebooks/00_setup_and_baselines.ipynb) | How data-hungry is from-scratch training? (the reference curve) |
| 01 | [pretrain → fine-tune](notebooks/01_pretrain_finetune.ipynb) | Scratch vs frozen features vs fine-tuning across target-data sizes |
| 02 | [layer transferability](notebooks/02_layer_transferability.ipynb) | Which layers are general, which are task-specific? (Yosinski et al. in miniature) |
| 03 | [task transfer matrix](notebooks/03_task_transfer_matrix.ipynb) | Which tasks transfer to which — and can LEEP predict it cheaply? |
| 04 | [negative transfer & forgetting](notebooks/04_negative_transfer_forgetting.ipynb) | Three ways reuse hurts + catastrophic forgetting and rehearsal |
| 05 | [multi-task synergy & interference](notebooks/05_multitask_synergy_interference.ipynb) | When does a second task help or hurt a shared trunk? |
| 06 | [modular composition](notebooks/06_modular_composition.ipynb) | A pretrained encoder as a module in a compound architecture (two-digit sum) |
| 07 | [self-supervised pretext](notebooks/07_selfsupervised_pretext.ipynb) | Transferable features without any source labels (RotNet-style) |

## Key findings (from the committed executed run)

Every notebook ends in machine-checked `CLAIM CHECK` assertions; all 36 pass
in the committed run. One-line versions:

- **00** From-scratch accuracy climbs steeply with data (MNIST 0.83→0.98,
  Fashion 0.65→0.85 from 100→10k examples) — the low-data gap is where
  transfer can pay.
- **01** A *related* source (digits 5–9 → 0–4) lifts few-shot accuracy by
  ~14 points at n=10 and the gap closes by n=10k (sample efficiency, not a
  better optimum — He et al. 2019 in miniature). A *distant* source
  (Fashion) helps ~not at all: what you pretrained on decides everything.
- **02** Yosinski in miniature: copied-and-frozen layers track or beat the
  baseline through block 4 (the source's 20× data advantage), then fall off
  a cliff (−7 pts) when the task-specific final block is locked too;
  fine-tuning recovers to baseline at every depth.
- **03** The transfer matrix is structured and *asymmetric*
  (fashion→mnist 0.85 ≫ mnist→fashion 0.73); memorization-shaped features
  land 33 points *below* an untrained encoder; LEEP rank-predicts the whole
  matrix from one forward pass (Spearman ρ = 0.78).
- **04** Negative transfer has preconditions: a poisoned source frozen in
  place costs 48 points, honest-but-frozen features cap 7 points below
  scratch once data is plentiful, and an aggressive encoder LR erases a
  12-point transfer advantage. Catastrophic forgetting drops old-task
  accuracy 0.95→0.55 in five epochs; replaying 5% of old data restores 0.92.
- **05** Auxiliary tasks: semantic relatedness predicts nothing — the
  "obviously related" rotation auxiliary *hurts* 6 points (competing
  invariances), a disjoint-classes auxiliary helps, and contradictory labels
  on the same images are absorbed. Trunk gradient cosine ranks the three
  correctly: it detects feature-space conflict, the only kind that matters.
- **06** Composability: a frozen digit encoder dropped into a two-digit-sum
  network (supervised only by the sum) scores 0.72 where end-to-end scratch
  manages 0.29 at 500 pairs; scratch needs 16× more data to catch up.
- **07** Self-supervision: rotation-pretext features probe within 2 points
  of supervised training *with zero labels* — but only at mid-depth (the
  pretext-adjacent layer collapses by 19 points), only on the pretext's own
  domain (−24 crossing to Fashion), and without beating supervised training
  at this easy-task scale (Newell & Deng 2020's regime story).

## How to run

```bash
# one-time setup + execute the whole suite in order (~1-2 h on 4 CPU cores)
bash run_all.sh

# fast smoke pass (~10x smaller, separate artifact dirs)
TLLAB_QUICK=1 bash run_all.sh
```

Notebooks are authored as jupytext percent scripts in `notebooks/src/` and
executed into `notebooks/*.ipynb`. Each notebook is independently re-runnable:
expensive steps go through checkpoint caching (`artifacts/checkpoints/`) and
sweep caching (`artifacts/results/*.csv`), so a re-run is plot-only and takes
seconds.

Dependencies: `pip install -r requirements.txt` (plain PyPI). On a CPU-only
Linux machine you can save several GB of CUDA dependencies with
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Layout

```
tllab/               shared mini-library (task registry, models, training, analysis)
notebooks/src/       jupytext .py sources (the authoritative code)
notebooks/*.ipynb    executed notebooks with outputs (the deliverable)
artifacts/figures/   PNGs saved by the notebooks
artifacts/results/   CSVs of every experiment sweep
```

## References

- Yosinski, Clune, Bengio, Lipson. *How transferable are features in deep neural networks?* NeurIPS 2014.
- Donahue et al. *DeCAF: A deep convolutional activation feature for generic visual recognition.* ICML 2014.
- Zamir et al. *Taskonomy: Disentangling task transfer learning.* CVPR 2018.
- Kornblith, Shlens, Le. *Do better ImageNet models transfer better?* CVPR 2019.
- He, Girshick, Dollár. *Rethinking ImageNet pre-training.* ICCV 2019.
- Wang, Dai, Póczos, Carbonell. *Characterizing and avoiding negative transfer.* CVPR 2019.
- Zhang et al. *Understanding deep learning requires rethinking generalization.* ICLR 2017.
- McCloskey, Cohen. *Catastrophic interference in connectionist networks.* 1989. · French. *Catastrophic forgetting in connectionist networks.* 1999.
- Robins. *Catastrophic forgetting, rehearsal and pseudorehearsal.* 1995. · Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks (EWC).* PNAS 2017.
- Caruana. *Multitask learning.* 1997. · Ruder. *An overview of multi-task learning in deep neural networks.* 2017.
- Standley et al. *Which tasks should be learned together in multi-task learning?* ICML 2020. · Yu et al. *Gradient surgery for multi-task learning (PCGrad).* NeurIPS 2020.
- Nguyen, Hassner, Seeger, Archambeau. *LEEP: A new measure to evaluate transferability of learned representations.* ICML 2020.
- Gidaris, Singh, Komodakis. *Unsupervised representation learning by predicting image rotations (RotNet).* ICLR 2018. · Doersch, Gupta, Efros. *Unsupervised visual representation learning by context prediction.* ICCV 2015.
- Neyshabur, Sedghi, Zhang. *What is being transferred in transfer learning?* NeurIPS 2020.
