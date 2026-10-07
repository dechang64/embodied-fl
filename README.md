<div align="center">

# Embodied-FL v2

### Federated Learning Platform for Embodied Intelligence

**Distributed robot training with data privacy — data never leaves the factory.**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[Repair log](docs/VLA_REPAIR.md) · [Experiments](experiments/)

</div>

---

> **Read this first.** Every number in this repository is produced by
> **synthetic simulation** on CPU. There is no real robot, no real camera and
> no real factory deployment. The "vision" channel is a fixed random
> projection of a simulated scene descriptor. Federated gains here are the
> classic *more data* effect, not evidence about real embodied deployments.
> See [Limitations](#-limitations-and-open-defects).

---

## 🎯 Problem

Embodied AI requires massive training data. But factory data is proprietary, home robots capture private environments, and regulations (HIPAA, GDPR) restrict data sharing.

**Result**: Each company trains in isolation → suboptimal models.

**Embodied-FL**: Train robots together, keep data apart.

---

## 🏗️ Architecture

```
Factory A ──┐                    ┌── YOLOv11 (scene detection)
            │    FedAvg          ├── DINOv2 + Classifier (task understanding)
Factory B ──┼──────────────────→ ├── Policy MLP (robot control)
            │    Aggregation     └── Grad-CAM (explainability)
Factory C ──┘
```

### Multi-Task FL

| Model | Task | Shared Weights | Frozen |
|-------|------|---------------|--------|
| YOLOv11 | Scene detection | Backbone only | Detection head |
| DINOv2 + Linear | Task classification | Linear head | Full ViT |
| Policy MLP | Robot control | Full MLP | — |

### Federated VLA model (`analysis/vla_model.py`)

| Component | Role | Aggregated |
|---|---|---|
| Vision / language / state projectors | modality → `d_model` | ✅ |
| Fusion encoder (`CrossAttentionFusion`) | encode all modality tokens, mean-pool | ✅ |
| Action head (`ActionHead`) | `d_model` → per-dimension action bins | selectable |

The action head vocabulary is `num_action_bins + 3`, matching the tokenizer's
`pad/eos/sos` offset. It is *not* `num_action_bins` — see
[repair log, D6](docs/VLA_REPAIR.md).

---

## 🚀 Quick Start

```bash
# Python dependency (torch, numpy, ...). Python 3.10 recommended.
pip install -r python/requirements.txt

# 1. Six classification experiments (NumPy, CPU, ~2 min)
PYTHONPATH=. python experiments/run_experiment.py

# 2. Federated VLA experiment — smoke test (~2 min)
PYTHONPATH=. python experiments/vla_fed/run_vla_federated.py --mode quick

# 3. VLA experiment — reportable run (CPU, ~30 min)
PYTHONPATH=. python experiments/vla_fed/run_vla_federated.py --mode paper

# 4. Information-ceiling diagnostic (always run this before trusting #2/#3)
PYTHONPATH=. python experiments/vla_fed/diagnose_vla.py --mode paper

# 5. Scaling studies — sample count vs goal coverage (~2 h, CPU)
PYTHONPATH=. python experiments/vla_fed/run_scaling.py --study samples --seeds 0,1,2 --methods local_only,fedavg_full
PYTHONPATH=. python experiments/vla_fed/run_scaling.py --study goals   --seeds 0,1,2 --methods fedavg_full

# 6. Tests
PYTHONPATH=. python -m pytest python/tests/ -q
```

Rust server: `cargo run` (gRPC :50051, REST :8080).

---

## 📊 Experiments — classification (Part A)

All numbers below are the **actual output** of `experiments/run_experiment.py`
(80 rounds, 5 local epochs), stored in
`experiments/results/experiment_results.json`. "Ours" is the task-aware
aggregation in `agg_ours()`.

### Exp1 — 5 clients, Non-IID (α = 0.5)

| Method | Accuracy | Loss |
|---|---|---|
| FedAvg | 0.9150 | 0.2937 |
| FedProx | 0.9156 | 0.2936 |
| **Ours** | **0.9446** | **0.1787** |

### Exp2 — 10 clients (scalability)

| Method | Accuracy | Loss |
|---|---|---|
| **FedAvg** | **0.6786** | **0.9757** |
| Ours | 0.6648 | 1.0530 |

### Exp3 — Non-IID severity sweep

| Severity | FedAvg | Ours | Δ |
|---|---|---|---|
| IID (α = 5.0) | 0.8589 | 0.8616 | +0.3% |
| Low (α = 1.0) | 0.8790 | 0.8927 | +1.6% |
| Medium (α = 0.5) | 0.9150 | 0.9446 | +3.2% |
| High (α = 0.1) | 0.9450 | 0.9406 | −0.5% |

### Exp4 — Heterogeneous tasks (shared backbone)

| Method | Accuracy |
|---|---|
| **FedAvg** | **0.8289** |
| Ours | 0.8255 |

### Exp5 — Continual learning (old-class retention after learning 4 new classes)

| Method | Old-class acc | New-class acc |
|---|---|---|
| Fine-tune | 0.9649 | 0.3600 |
| EWC only | 0.9737 | 0.3187 |
| Replay only | 0.9724 | 0.3438 |
| **EWC + Replay** | **0.9749** | 0.3113 |

### Exp6 — Gradient compression (validation accuracy)

| Method | Ratio | Accuracy |
|---|---|---|
| No compression | 1.0× | 0.9750 |
| TopK-50 | 2.0× | 0.9850 |
| TopK-70 | 3.3× | 0.9800 |
| **TopK-90** | **10.0×** | **0.9600** |
| TopK-95 | 20.0× | 0.9350 |
| TopK-99 | 100.0× | 0.3250 |
| Quant-4/8/16-bit | 8/4/2× | 0.9700 / 0.9700 / 0.9750 |

**How to read these honestly.** Task-aware aggregation wins on Exp1 (+3.2 %)
and on moderate non-IID splits, but it **does not beat FedAvg on Exp2, Exp4, or
the most extreme non-IID setting**. Continual-learning gains are real but small
(+1.0 point of old-class retention), and every method still loses most of its
new-class accuracy. Only the compression results are unambiguously strong.

---

## 🤖 Experiments — federated VLA

See **[docs/VLA_REPAIR.md](docs/VLA_REPAIR.md)** for the full defect list. The
short version: the v2 result files reported a `global_accuracy` below the
random baseline, because (a) the trainer performed one full-batch step per
"epoch", (b) evaluation ran on the training tensors, and (c) the global model's
action head was **never trained** — only the backbone was aggregated.

### Information ceiling (`diagnose_vla.py --mode paper`)

| Quantity | Value |
|---|---|
| random (uniform over 16 bins) | 0.0625 |
| per-dimension majority class (trivial floor) | 0.2812 |
| ridge — linear readout of the model's own inputs | 0.7958 |
| oracle — perfect reader of the observable scene | 0.8543 |
| **headroom above the trivial floor** | **+0.573** |

(`diagnose_vla.py` draws its own data seeds, so its floor is 0.2812 while the
3-seed run below reports 0.2741. Both are the same estimator.)

A *linear* probe of the exact tensors the network receives reaches 0.80. Any
network result far below that is a pipeline defect, not a finding.

### Methods (held-out episodes, mean ± std over 3 seeds)

| Method | Test accuracy | vs. local-only |
|---|---|---|
| random | 0.0625 ± 0.0000 | — |
| majority (trivial floor) | 0.2741 ± 0.0046 | — |
| local_only (no federation) | 0.5516 ± 0.0016 | — |
| fedavg_backbone | 0.6188 ± 0.0046 | **+6.7 pt** |
| **fedavg_full** | **0.7429 ± 0.0030** | **+19.1 pt** |
| ridge ceiling (linear readout) | 0.7958 | (bound, not a method) |
| oracle (perfect scene reader) | 0.8543 | (bound, not a method) |

Protocol: episode-wise train/test split (no adjacent-frame leakage), 400
episodes per client, per-dimension quantile action bins fitted on pooled
**train** actions only, evaluation on held-out episodes only. Raw output in
`results/vla_fed/results_paper.json`.

Two things fall out of this table:

1. **Federation helps, and it helps a lot here** — pooling 5 clients moves
   test accuracy from 0.552 to 0.743. This is the *more data* effect on a
   shared dynamics model, not a claim about heterogeneous physics.
2. **Aggregating the action head matters more than the backbone.**
   `fedavg_backbone` (shared backbone, per-client head) gains only +6.7 pt;
   `fedavg_full` gains +19.1 pt and lands at 93 % of the ridge ceiling.
   Keeping the head local leaves most of the federated benefit on the table.

### What sets the ceiling: goal coverage, not sample count

The binding budget for this task is **which goals you have seen**, not how many
frames you have. Hold the training set at 19,200 samples (5 clients x 400
episodes x 12 steps) and change only the number of *distinct* goals in it --
3 seeds, held-out goals never seen in training (`goal_overlap = 0.000`):

| distinct goals in train | trajectories per goal | test accuracy | trivial floor |
|---|---|---|---|
| 20 | 80 | 0.633 +/- 0.022 | 0.292 |
| 40 | 40 | 0.658 +/- 0.010 | 0.294 |
| 80 | 20 | 0.736 +/- 0.003 | 0.300 |
| 160 | 10 | 0.738 +/- 0.002 | 0.291 |
| 320 | 5 | 0.744 +/- 0.006 | 0.283 |

Spreading the same budget over more goals (20 -> 80) is worth **+10.3 pt**;
80 -> 320 goals adds **+0.8 pt**. The trivial floor is flat across the sweep
(0.283-0.300) and so is the action distribution, so this is not a binning
artefact.

Growing the budget instead -- episodes 50 -> 800, i.e. 2,400 -> 38,400 samples
-- does help, but saturates: 0.405 / 0.516 / 0.663 / 0.743 / **0.776**, with the
last doubling buying only +3.3 pt.

Reproduce with `run_scaling.py` (see [docs/VLA_REPAIR.md](docs/VLA_REPAIR.md),
section 5). Raw output: `results/vla_fed/scaling_goals.json`,
`results/vla_fed/scaling_samples.json`.

---


## 🔍 Limitations and open defects

* **Simulation only.** No real robot, camera, or deployment.
* **Shared dynamics.** All five "factories" use one dynamics model; heterogeneity comes from different goals and instructions. Federated gains are a data-pooling effect.
* **Language channel is a hash.** One constant instruction per client — this is not evidence about language grounding.
* **Duplicated module trees.** `analysis/` and `python/analysis/` are near-copies of the same code (`python/tests/` resolves to the former when run from the repo root and to the latter when run from `python/`). They are currently kept in sync manually; one of them should be removed.
* **3 test modules need an absent optional dependency** (`twc_core`) and fail to collect when pytest is invoked from `python/`.
* **Rust/C++ and the Streamlit dashboard are not covered by CI here**; the reported results come from the Python experiment scripts only.

---

## 📄 License

Apache-2.0

---

<div align="center">

**Embodied-FL v2** — Train robots together, keep data apart.

</div>
