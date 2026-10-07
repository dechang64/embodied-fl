#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLA Federated Experiment — Diagnostic / Sanity Harness
======================================================
Answers three questions BEFORE trusting any accuracy number:

  Q1. What is the trivial-predictor floor? (random + per-dimension majority)
  Q2. Is the prediction target recoverable from the model's actual inputs?
      Reported as three ceilings on the SAME split:
        oracle  - a perfect reader of the observable scene (upper bound)
        ridge   - a *linear* readout of the exact tensors the network gets
        k-NN    - a nonparametric readout of the same tensors
      If the network cannot clear `ridge`, the bottleneck is the model, not
      the data.
  Q3. Do the tokenizer's token ids fit inside the action head's vocabulary?

Reuses the real data path from run_vla_federated.py so the numbers cannot
drift away from the experiment.

Usage:
  PYTHONPATH=. python experiments/vla_fed/diagnose_vla.py --mode quick
  PYTHONPATH=. python experiments/vla_fed/diagnose_vla.py --mode paper
"""

import os
import sys
import json
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root
sys.path.insert(0, _HERE)

from run_vla_federated import (                                # noqa: E402
    ExperimentConfig, MODES, CLIENT_SCENARIOS, build_client_dataset,
    split_by_episode, make_tokenizer, build_tensors, SceneEncoder, pad_or_truncate,
)
from analysis.instruction_embedding import InstructionEmbedder, EmbeddingConfig  # noqa: E402


def ridge_fit_predict(X, Y, Xt, lam=1.0):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    Xta = np.hstack([Xt, np.ones((len(Xt), 1))])
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    W = np.linalg.solve(A, Xa.T @ Y)
    return Xta @ W


def knn_ceiling(Xtr, ytr, Xte, yte, k=5):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    d2 = (Xte ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None, :] - 2 * Xte @ Xtr.T
    k = min(k, Xtr.shape[0])
    nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
    A = ytr.shape[1]
    hit = 0
    for i in range(Xte.shape[0]):
        nb = ytr[nn[i]]
        for a in range(A):
            vals, cnts = np.unique(nb[:, a], return_counts=True)
            hit += int(vals[cnts.argmax()] == yte[i, a])
    return hit / (Xte.shape[0] * A)


def majority_floor(ytr, yte):
    hit = 0
    for a in range(ytr.shape[1]):
        vals, cnts = np.unique(ytr[:, a], return_counts=True)
        hit += int((yte[:, a] == vals[cnts.argmax()]).sum())
    return hit / (yte.shape[0] * yte.shape[1])


def oracle_actions(samples, cfg):
    """The command a perfect reader of the observable scene would emit."""
    S = np.array([s.robot_state for s in samples], dtype=np.float32)
    G = np.array([s.goal for s in samples], dtype=np.float32)
    Q = np.array([[s.gripper] for s in samples], dtype=np.float32)
    A = np.zeros((len(samples), cfg.action_dim), dtype=np.float32)
    A[:, :cfg.state_dim] = (G - S) * 0.3
    if cfg.action_dim > cfg.state_dim:
        A[:, cfg.state_dim] = np.where(Q[:, 0] > 0.5, 1.0, 0.0) - Q[:, 0]
    return np.clip(A, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "paper"], default="quick")
    ap.add_argument("--max_kn", type=int, default=3000,
                    help="cap rows used for the k-NN ceiling (it is O(N*M))")
    args = ap.parse_args()

    cfg = ExperimentConfig(mode=args.mode, **MODES[args.mode])

    print("=" * 74)
    print(f"  VLA DIAGNOSTIC — {cfg.mode} mode  (noise={cfg.action_noise}, "
          f"bins={cfg.num_action_bins}, {cfg.binning})")
    print("=" * 74)

    clients = CLIENT_SCENARIOS[:cfg.n_clients]
    datasets = [split_by_episode(build_client_dataset(c, cfg, seed=i * 7 + 1).samples,
                                 cfg.test_episode_stride)
                for i, c in enumerate(clients)]

    pooled = np.concatenate([
        pad_or_truncate(np.array([s.action for s in tr], dtype=np.float32), cfg.action_dim)
        for tr, _ in datasets])
    tok = make_tokenizer(pooled, cfg)
    emb = InstructionEmbedder(EmbeddingConfig(mode="hash", dimension=cfg.lang_dim))
    scene = SceneEncoder(desc_dim=2 * cfg.state_dim + 1, vision_dim=cfg.vision_dim)

    TR, TE, ORC = [], [], []
    oob = tot = 0
    for tr, te in datasets:
        b_tr = build_tensors(tr, cfg, emb, tok, scene)
        b_te = build_tensors(te, cfg, emb, tok, scene)
        TR.append(b_tr); TE.append(b_te)
        # oracle: perfect reader of the observable scene
        ORC.append((oracle_actions(tr, cfg), b_tr["tokens"].numpy()))
        tk = b_tr["tokens"].numpy()
        oob += int((tk >= cfg.num_action_bins + 3).sum()); tot += tk.size

    Xtr = np.vstack([b["vision"].numpy() for b in TR])
    Str = np.vstack([b["state"].numpy() for b in TR])
    ytr = np.vstack([b["tokens"].numpy() for b in TR])
    Xte = np.vstack([b["vision"].numpy() for b in TE])
    Ste = np.vstack([b["state"].numpy() for b in TE])
    yte = np.vstack([b["tokens"].numpy() for b in TE])

    print(f"\n  split: train={len(ytr)}  test={len(yte)}  "
          f"({cfg.n_clients} clients, 1 episode in {cfg.test_episode_stride} held out)")

    # --- Q3 ---
    print("\n  Q3) tokenizer vocabulary vs action-head vocabulary")
    print(f"      tokenizer vocab_size  : {tok.vocab_size} (bins {cfg.num_action_bins} "
          f"+ pad/eos/sos)")
    print(f"      action head classes   : {cfg.num_action_bins + 3}  "
          f"(token ids >= vocab: {oob}/{tot})")

    # --- Q1 ---
    rand = 1.0 / cfg.num_action_bins
    maj = majority_floor(ytr, yte)
    print("\n  Q1) trivial-predictor floor")
    print(f"      random (uniform over bins) : {rand:.4f}")
    print(f"      per-dim majority class     : {maj:.4f}")

    # --- Q2 ---
    Atr_c = np.vstack([a for a, _ in ORC])
    oracle = float((tok.encode_batch(Atr_c) == ytr).mean())
    Ate_c = np.vstack([oracle_actions(te, cfg) for tr, te in datasets])
    oracle_te = float((tok.encode_batch(Ate_c) == yte).mean())

    print("\n  Q2) information ceiling — is the target recoverable from the model inputs?")
    print(f"      oracle (perfect scene reader)  train={oracle:.4f}  test={oracle_te:.4f}")

    for name, Xa, Xb in [("ridge on [state]        ", Str, Ste),
                         ("ridge on [vision]       ", Xtr, Xte),
                         ("ridge on [state+vision] ", np.hstack([Str, Xtr]),
                          np.hstack([Ste, Xte]))]:
        P = ridge_fit_predict(Xa, Atr_c, Xb)
        print(f"      {name} test={float((tok.encode_batch(P) == yte).mean()):.4f}")

    rng = np.random.RandomState(0)
    ti = rng.choice(len(Xtr), min(args.max_kn, len(Xtr)), replace=False)
    vi = rng.choice(len(Xte), min(args.max_kn, len(Xte)), replace=False)
    print(f"      k-NN  on [state+vision]  test="
          f"{knn_ceiling(np.hstack([Str[ti], Xtr[ti]]), ytr[ti], np.hstack([Ste[vi], Xte[vi]]), yte[vi]):.4f}")

    # --- verdict ---
    floor = max(rand, maj)
    print("\n  VERDICT")
    print(f"      trivial floor = {floor:.4f}   oracle = {oracle_te:.4f}")
    if oracle_te <= floor * 1.1:
        print("      the task is NOT LEARNABLE from the observable scene either.")
        print("      -> fix the data-generating process before reporting anything.")
    else:
        print("      the task is learnable; headroom above the floor = "
              f"{oracle_te - floor:+.4f}.")
        print("      a network that sits at the floor is a PIPELINE defect.")
    print("=" * 74)

    out = {
        "mode": args.mode, "config": vars(cfg) if hasattr(cfg, "__dict__") else {},
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "random_baseline": rand, "majority_baseline": maj,
        "oracle_train": oracle, "oracle_test": oracle_te,
        "trivial_floor": floor, "headroom": oracle_te - floor,
        "tokens_out_of_vocab": oob, "tokens_total": tot,
        "tokenizer_vocab_size": tok.vocab_size,
        "action_head_classes": cfg.num_action_bins + 3,
    }
    p = f"results/vla_fed/diagnose_{args.mode}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  saved -> {p}")


if __name__ == "__main__":
    main()
