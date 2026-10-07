# VLA federation repair — what was broken and what changed

This document records why the `experiments/vla_fed` results in v2 could not be
used, what was changed, and how the new numbers should be read.

Everything below is reproducible:

```bash
cd <repo root>
PYTHONPATH=. python experiments/vla_fed/diagnose_vla.py --mode paper
PYTHONPATH=. python experiments/vla_fed/run_vla_federated.py --mode paper
PYTHONPATH=. python experiments/vla_fed/run_scaling.py --study samples --seeds 0,1,2 --methods local_only,fedavg_full
PYTHONPATH=. python experiments/vla_fed/run_scaling.py --study goals   --seeds 0,1,2 --methods fedavg_full
```

---

## 1. Summary

The v2 result files reported a `global_accuracy` of **2.7 %** (paper mode) and
**5.0 %** (quick mode). Both are **below the random baseline** (3.1 % / 6.25 %)
and far below the accuracy of a constant predictor (32.2 % / 28.5 %).

That number was not a modelling result. The pipeline was not training, and the
model being evaluated was not the model that had been trained.

---

## 2. Defects

| # | Defect | Location | Effect |
|---|---|---|---|
| D1 | `vision_features = torch.randn(N, vision_dim)` | `run_vla_federated.py` `prepare_training_tensors` | visual channel was i.i.d. noise — zero information |
| D2 | one instruction per client → hash embedding constant within a client | same | language channel degenerated to a client id |
| D3 | `train_local` did **one full-batch step per "epoch"** | `vla_model.py` `VLAFLTrainer.train_local` | quick mode ran **3 gradient updates in total**; paper mode 60 |
| D4 | docstring claimed "held-out test data"; evaluation used the **training** tensors | `run_vla_federated.py` `run_federated_training` | reported number was not a generalisation metric |
| **D5** | the global model's `action_head` was **freshly initialised and never trained** — only the backbone was aggregated | `run_vla_federated.py`: `global_model = VLAFLModel(...)` then `load_shared_params(aggregated)` | the evaluated head was random; `global_accuracy` was meaningless by construction |
| D6 | `ActionTokenizer.encode` returns `bin_index + 3`, but `ActionHead` emitted `num_action_bins` classes | `action_tokenizer.py` vs `vla_model.py` | latent `cross_entropy` out-of-bounds — a random crash waiting to happen |
| D7 | no baseline was reported | — | 6.83 % could not be distinguished from chance |
| D8 | gripper action dim was **pure noise** in the generator | `vla_collector.py` | one of eight target dims was unlearnable by any model |
| D9 | action tokenization used fixed `[-1, 1]` bins | `action_tokenizer.py` | ~1/3 of all actions land in one central bin; a constant predictor scored 32–44 % |
| D10 | `run_experiment.py` missing `from typing import List` | `experiments/run_experiment.py` | script could not start at all |
| D11 | `run_all_experiments.py` Exp5: phase-1 one-hot was 6-wide, phase-2 10-wide | `run_all_experiments.py` `make_phase_data` | replay `np.vstack` raised a shape error |
| D12 | `run_all_experiments.py` EWC fisher built **blocked** but indexed **interleaved** | `run_all_experiments.py` `EWC.consolidate` | `consolidate()` raised a broadcast error |

D5 is the one that produced the headline number. D1/D2/D8 mean that even a
perfectly working trainer had almost nothing to learn from.

### Measured evidence for D5

`global_model` is constructed fresh and only receives `shared` parameters
(projectors + fusion). The action head — the only part that maps a fused
representation to a token — keeps its random initialisation, so
`argmax` over its logits is close to a coin flip. The round-1 logs show
`avg_client_accuracy = 0.33` while `global_accuracy = 0.027` in the *same*
round: the local models were learning, and the aggregation step was discarding
exactly the part that mattered.

---

## 3. What changed

**Data generating process** (`analysis/vla_collector.py`)

* The commanded action is now a reflexive controller on the *observable* scene
  `(robot_state, goal_pose, gripper_aperture)`:
  `a = [0.3 * (goal - state), gripper_target - gripper] + noise`.
  Previously the gripper dim was pure noise.
* The episode goal is recorded in `Episode.metadata["target_state"]` and
  carried through `VLASample.goal`.
* Per-step noise is configurable (`action_noise`, default `0.005`). The
  default matters: the clean command has mean magnitude ≈ `0.073`, so the old
  hard-coded `0.01` noise was ~14 % of the signal and `0.03` drowns it.

**Model** (`analysis/vla_model.py`)

* `ActionHead` now emits `num_action_bins + 3` classes, matching the tokenizer
  vocabulary (D6).
* `train_local` performs real minibatch SGD: shuffled batches, `local_epochs`
  full passes (D3), with `batch_size` / `shuffle` arguments.
* `CrossAttentionFusion` runs a standard encoder over **all** modality tokens
  and mean-pools them. The old version let the action head read only the
  vision token, which gave the other modalities ~one scalar of bandwidth.
* Added `get_full_state_dict` / `load_full_params` / `get_head_state_dict` /
  `load_head_params`.

**Experiment** (`experiments/vla_fed/run_vla_federated.py`, rewritten)

* Simulated visual encoder: `--vision scene` projects the observable scene
  descriptor with a **fixed** random matrix plus noise. `--vision random`
  restores the v2 behaviour for use as an ablation.
* Shared action tokenizer fitted on pooled **train** actions with per-dimension
  quantile bins (`--binning quantile`, the RT-1 / Octo convention).
  `--binning uniform` restores fixed `[-1,1]` bins.
* **Episode-wise** train/test split (no adjacent-frame leakage).
* Five methods, all evaluated on held-out episodes:
  `random`, `majority`, `local_only`, `fedavg_backbone`, `fedavg_full`.
* Multi-seed, mean ± std, results written to `results/vla_fed/`.

**Other scripts**

* `experiments/run_experiment.py`: added the missing `from typing import List` (D10).
* `experiments/run_all_experiments.py`: aligned the Exp5 label space (D11) and
  fixed the EWC Fisher layout (D12).

---

## 4. Ceilings (measured, `diagnose_vla.py`)

The benchmark is only meaningful if the target is recoverable from the model's
own inputs. Three independent ceilings are reported on the same split:

| | quick mode | paper mode |
|---|---|---|
| split (train / test samples) | 2880 / 720 | 19200 / 4800 |
| random (uniform over bins) | 0.0625 | 0.0625 |
| majority (per-dim mode) — **trivial floor** | 0.2850 | 0.2812 |
| ridge — linear readout of the model's inputs | 0.7167 | 0.7958 |
| oracle — perfect reader of the observable scene | 0.8292 | 0.8543 |
| **headroom above the floor** | **+0.544** | **+0.573** |

A linear probe reaching 0.72–0.80 while the *original* network sat at the floor
is direct evidence that the information was present and the **pipeline**, not
the data, was the problem.

> `diagnose_vla.py` draws its own data seeds, so the floor it reports for paper
> mode (0.2812) differs in the third decimal from the floor computed inside the
> 3-seed run (0.2741). Same estimator, different draws.

Two practical lessons came out of this sweep and are encoded in `MODES`:

* **Spend the budget on episodes, not on steps.** Steps within an episode are
  highly correlated, so episode count buys far more distinct goals per gradient
  step. This is now measured formally in §5.
* **32 action bins were too fine** for the available sample size: identical
  settings scored 0.26 test accuracy at 32 bins vs 0.44 at 16 bins.

---

## 5. Learnability is governed by goal coverage, not sample size

The claim behind FedCover-WM — aggregate *coverage statistics* rather than
parameters or sample counts — needs a controlled measurement: hold the sample
budget fixed, change only how many distinct goals the training set covers.
`experiments/vla_fed/run_scaling.py` runs two studies, 3 seeds each, evaluated
on held-out episodes only.

**Shared setup.** Paper-mode network, 16 quantile action bins,
`action_noise=0.005`, 12 rounds, 3 local epochs, 5 clients. In **Study B** the
training set is held at 19,200 samples (5 clients × 400 episodes × 12 steps,
pooled, 4/5 split) and only the goal-pool size changes. The goal pool is
injected through `SyntheticCollector.collect(..., goal_pool=...)`, which
defaults to `None` and leaves the original data path bit-identical.

### Study B — coverage changes, sample size does not

| distinct goals in train | trajectories per goal | test accuracy | trivial floor |
|---|---|---|---|
| 20 | 80 | 0.633 ± 0.022 | 0.292 |
| 40 | 40 | 0.658 ± 0.010 | 0.294 |
| 80 | 20 | 0.736 ± 0.003 | 0.300 |
| 160 | 10 | 0.738 ± 0.002 | 0.291 |
| 320 | 5 | 0.744 ± 0.006 | 0.283 |

At a fixed budget, spreading it over more goals (20 → 80) is worth **+10.3 pt**;
going from 80 to 320 goals adds only **+0.8 pt**. Repeating each goal 80 times
is worse than covering four times as many goals with the same samples.

The trivial floor is flat across the sweep (0.283–0.300), and so is the action
distribution (`mean |a|` 0.066–0.068) — the improvement cannot be attributed to
binning or a drifting data distribution. Every held-out goal is absent from the
training set (`goal_overlap = 0.000`): this is pure extrapolation.

Raw output: `results/vla_fed/scaling_goals.json`.

### Study A — sample size changes (goals grow with it)

| episodes / client | train samples | distinct goals | test accuracy | trivial floor |
|---|---|---|---|---|
| 50 | 2,400 | 200 | 0.405 ± 0.013 | 0.294 |
| 100 | 4,800 | 400 | 0.516 ± 0.003 | 0.303 |
| 200 | 9,600 | 800 | 0.663 ± 0.015 | 0.285 |
| 400 | 19,200 | 1,600 | 0.743 ± 0.003 | 0.274 |
| 800 | 38,400 | 3,200 | 0.776 ± 0.002 | 0.277 |

Monotone but saturating: the last doubling buys **+3.3 pt**. In this arm sample
count and goal count rise together — Study B is what separates them.

Raw output: `results/vla_fed/scaling_samples.json`.

**Cross-check.** Study A's 400-episode arm reproduces the §6 headline table
per seed (local_only 0.5538 / 0.5512 / 0.5498 and fedavg_full 0.7445 / 0.7388 /
0.7455, identical to `results_paper.json`), which pins both files to the same
data path.

> **A defect this study caught.** `run_federated()` used to fall back *silently*
> to `fedavg_backbone` for any unrecognised method name, so the first Study A run
> labelled a `fedavg_backbone` column `local_only`. The values matched the
> paper-mode backbone seeds bit-for-bit, which is how it was spotted. The
> function now raises on unknown methods, and the column was re-measured; the
> mis-dispatched values are kept under `_dispatch_artifact` in
> `scaling_samples.json`.

---

## 6. Results

`run_vla_federated.py --mode paper`, 3 seeds, every method evaluated on
**held-out episodes only**:

| Method | Test accuracy (mean ± std) | Δ vs. local-only |
|---|---|---|
| random | 0.0625 ± 0.0000 | — |
| majority (trivial floor) | 0.2741 ± 0.0046 | — |
| local_only | 0.5516 ± 0.0016 | — |
| fedavg_backbone | 0.6188 ± 0.0046 | +6.7 pt |
| **fedavg_full** | **0.7429 ± 0.0030** | **+19.1 pt** |
| ridge ceiling | 0.7958 | (bound) |
| oracle | 0.8543 | (bound) |

Raw output: `results/vla_fed/results_paper.json`.

For reference, the v2 numbers produced by the same code path before the repair
were `global_accuracy = 0.0269` (paper) and `0.0500` (quick) — both **below the
random baseline**, measured on training data, through an untrained action head.

Interpretation:

* Federation moves test accuracy from 0.552 to 0.743 (+19.1 pt). On this
  benchmark that is the *more data* effect: the five clients share one dynamics
  model and differ only in goals and instructions.
* `fedavg_backbone` (shared backbone, per-client head) captures only a third of
  that gain. Keeping the action head local — which the module docstring
  previously described as the point of the design — costs +12.4 pt here.
* The repaired model reaches 93 % of the linear ceiling, so the remaining gap
  to the oracle is mostly quantisation plus action noise, not optimisation.


---

## 7. What is deliberately NOT claimed

* These are **synthetic simulation** results. The goal pose is given to the
  simulated encoder directly; no real camera, no real robot, no real
  deployment.
* The five "factories" share one dynamics model. Heterogeneity comes from
  different goals and instructions, not from different physics. Any advantage
  of federation here is the classic *more data* effect.
* The language channel is a deterministic hash of a per-client constant
  instruction. It is **not** evidence about language grounding.
* `fedavg_backbone` and `fedavg_full` differ only in whether the action head is
  aggregated; the "heterogeneous action space" story in the README header is
  not exercised, because all clients are padded to a common `action_dim`.
