#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLA Federated — scaling study (what actually makes this task learnable?)
=======================================================================
Two controlled studies, both reusing the *real* data path and the *real*
training code from `run_vla_federated.py`, so the numbers cannot drift away
from the headline experiment.

STUDY A — "samples"   episodes per client ∈ {50, 100, 200, 400, 800},
                      steps per episode FIXED at 12.
  Sample count and the number of distinct episode goals grow together, so this
  study alone cannot say which one is doing the work. It is the formal version
  of the scaling observation quoted in the docs.

STUDY B — "goals"     episodes per client FIXED at 400, steps FIXED at 12,
                      distinct goal pool ∈ {25, 50, 100, 200, 400}.
  The *sample budget is identical in every arm* (400 x 12 = 4800 per client);
  only the number of distinct goals changes. Any slope here is attributable to
  goal diversity, not to sample count.

Why the pool exists: `SyntheticCollector.collect(goal_pool=...)` makes episode
`i` target `goal_pool[i % G]`, so a caller can hold G fixed while varying the
budget (or the reverse). The marginal goal distribution is unchanged, so the
pooled action distribution — and therefore the tokenizer's quantile bin edges —
stay comparable across arms. The overlap diagnostic below reports how often a
held-out episode's goal was also seen during training, which is exactly the
quantity that changes between the arms.

Everything is episode-wise split, all accuracies are on HELD-OUT episodes, and
the trivial floor (per-dimension majority) is recomputed per arm because it
depends on the tokenizer, which depends on the arm's own training actions.

Usage:
  PYTHONPATH=. python experiments/vla_fed/run_scaling.py --study goals
  PYTHONPATH=. python experiments/vla_fed/run_scaling.py --study samples --seeds 0,1,2
  PYTHONPATH=. python experiments/vla_fed/run_scaling.py --study all
"""

import os
import sys
import json
import time
import argparse
from dataclasses import asdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root
sys.path.insert(0, _HERE)

from run_vla_federated import (                                # noqa: E402
    ExperimentConfig, MODES, CLIENT_SCENARIOS, VLADataset,
    SyntheticCollector, split_by_episode, make_tokenizer, build_tensors,
    SceneEncoder, pad_or_truncate, majority_predictor_accuracy,
    run_federated, run_local_only,
    InstructionEmbedder, EmbeddingConfig, VLAConfig,
)

METHODS = ("local_only", "fedavg_backbone", "fedavg_full")

STUDY_A_EPISODES = [50, 100, 200, 400, 800]
STUDY_B_GOAL_POOLS = [25, 50, 100, 200, 400]
BASE_STEPS = 12
BASE_EPISODES = 400
BASE_MODE = "paper"


# ── data path (identical to the main experiment, plus the goal pool) ──

def build_dataset(client, cfg, seed, goal_pool=None) -> VLADataset:
    """Same as run_vla_federated.build_client_dataset, but goal-injectable."""
    collector = SyntheticCollector(
        robot_type=client.robot_type,
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        seed=seed,
        action_noise=cfg.action_noise,
    )
    episodes = collector.collect(
        num_episodes=cfg.episodes_per_client,
        steps_per_episode=cfg.steps_per_episode,
        task_type=client.task_type,
        instruction=client.instruction,
        goal_pool=goal_pool,
    )
    return VLADataset.from_episodes(episodes, skip_no_action=False)


def _goal_key(g):
    return tuple(np.round(np.asarray(g, dtype=np.float64), 6))


def goal_diagnostics(datasets):
    """Distinct goals per arm and the train/test goal overlap."""
    tr_keys, te_keys = set(), set()
    n_tr = n_te = 0
    for tr, te in datasets:
        for s in tr:
            if s.goal is not None:
                tr_keys.add(_goal_key(s.goal))
        for s in te:
            if s.goal is not None:
                te_keys.add(_goal_key(s.goal))
        n_tr += len(tr)
        n_te += len(te)
    overlap = (len(te_keys & tr_keys) / len(te_keys)) if te_keys else float("nan")
    return {
        "n_train_total": n_tr,
        "n_test_total": n_te,
        "distinct_train_goals": len(tr_keys),
        "distinct_test_goals": len(te_keys),
        "test_goal_seen_in_train_frac": float(overlap),
    }


def one_arm(cfg: ExperimentConfig, seed: int, methods, verbose=True) -> dict:
    """Build the arm's data once, then run every requested method on it."""
    cfg.seed = seed
    clients = CLIENT_SCENARIOS[:cfg.n_clients]

    pool = None
    if cfg.goal_pool_size is not None:
        rng = np.random.RandomState(cfg.goal_pool_seed)
        pool = (rng.randn(cfg.goal_pool_size, cfg.state_dim) * 0.3).astype(np.float32)

    datasets = [
        split_by_episode(
            build_dataset(client, cfg, seed=seed * 100 + i, goal_pool=pool).samples,
            cfg.test_episode_stride,
        )
        for i, client in enumerate(clients)
    ]

    # shared tokenizer on pooled TRAIN actions (per-arm, as in the main run)
    pooled = np.concatenate([
        pad_or_truncate(np.array([s.action for s in tr], dtype=np.float32), cfg.action_dim)
        for tr, _ in datasets
    ])
    tokenizer = make_tokenizer(pooled, cfg)

    embedder = InstructionEmbedder(EmbeddingConfig(mode="hash", dimension=cfg.lang_dim))
    scene = SceneEncoder(desc_dim=2 * cfg.state_dim + 1, vision_dim=cfg.vision_dim,
                         seed=7, noise=0.05)

    train_bundles, test_bundles = [], []
    for (tr, te) in datasets:
        train_bundles.append(build_tensors(tr, cfg, embedder, tokenizer, scene))
        test_bundles.append(build_tensors(te, cfg, embedder, tokenizer, scene))

    vla_cfg = VLAConfig(
        vision_dim=cfg.vision_dim, lang_dim=cfg.lang_dim, state_dim=cfg.state_dim,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_fusion_layers=cfg.n_fusion_layers,
        action_dim=cfg.action_dim, num_action_bins=cfg.num_action_bins,
        lr=cfg.lr, local_epochs=cfg.local_epochs, batch_size=cfg.batch_size,
    )

    out_methods = {}
    # ── trivial floor (recomputed per arm: the tokenizer depends on the arm) ──
    maj = [majority_predictor_accuracy(train_bundles[k], test_bundles[k])
           for k in range(len(clients))]
    out_methods["majority"] = {"final_test_accuracy": float(np.mean(maj)),
                               "per_client_final": [float(a) for a in maj]}
    out_methods["random"] = {"final_test_accuracy": 1.0 / cfg.num_action_bins}

    for method in methods:
        t0 = time.time()
        if method == "local_only":
            res = run_local_only(cfg, vla_cfg, train_bundles, test_bundles, seed)
        else:
            res = run_federated(cfg, vla_cfg, train_bundles, test_bundles, method, seed)
        res["elapsed_total"] = time.time() - t0
        out_methods[method] = res
        if verbose:
            print(f"      {method:16s} test_acc={res['final_test_accuracy']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    diag = goal_diagnostics(datasets)
    diag["mean_abs_train_action"] = float(np.mean(np.abs(pooled)))
    diag["std_train_action"] = float(np.std(pooled))

    return {"seed": seed, "diagnostics": diag, "methods": out_methods}


def summarize(points):
    """Add per-arm mean/std across seeds for every method."""
    for pt in points:
        labels = list(pt["per_seed"][0]["methods"].keys())
        pt["summary"] = {}
        for m in labels:
            accs = [r["methods"][m]["final_test_accuracy"] for r in pt["per_seed"]]
            pt["summary"][m] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "per_seed": [float(a) for a in accs],
            }
        floor = max(pt["summary"]["random"]["mean"], pt["summary"]["majority"]["mean"])
        pt["trivial_floor"] = float(floor)
    return points


def arm_cfg(base: ExperimentConfig, **over) -> ExperimentConfig:
    cfg = ExperimentConfig(**{**asdict(base), **over})
    return cfg


def run_study(name, base_cfg, arms, seeds, methods, results_dir, out_tag=""):
    """arms: list of dicts of ExperimentConfig overrides + a label."""
    print("=" * 78)
    print(f"  STUDY {name.upper()}   seeds={seeds}  methods={methods}")
    print("=" * 78, flush=True)

    points = []
    t_study = time.time()
    for arm in arms:
        cfg = arm_cfg(base_cfg, seeds=list(seeds), **arm["overrides"])
        label = arm["label"]
        print(f"\n  ── {label}  (goal_pool={cfg.goal_pool_size}) ──", flush=True)
        per_seed = [one_arm(cfg, s, methods) for s in seeds]
        d0 = per_seed[0]["diagnostics"]
        print(f"    data: train={d0['n_train_total']} test={d0['n_test_total']} "
              f"| distinct train/test goals = {d0['distinct_train_goals']}/"
              f"{d0['distinct_test_goals']} | test-goal-seen-in-train="
              f"{d0['test_goal_seen_in_train_frac']:.3f}", flush=True)
        points.append({
            "label": label,
            **arm["overrides"],
            "per_seed": per_seed,
        })

    summarize(points)
    print("\n" + "=" * 78)
    print(f"  STUDY {name.upper()} — SUMMARY (mean ± std over {len(seeds)} seeds)")
    print("=" * 78)
    hdr = f"  {'arm':<16} {'train N':>8} {'#goals':>7} {'floor':>7}"
    for m in methods:
        hdr += f" {m:>16}"
    print(hdr)
    for pt in points:
        d = pt["per_seed"][0]["diagnostics"]
        row = (f"  {pt['label']:<16} {d['n_train_total']:>8} "
               f"{d['distinct_train_goals']:>7} {pt['trivial_floor']:>7.3f}")
        for m in methods:
            s = pt["summary"][m]
            row += f" {s['mean']:>8.3f}±{s['std']:.3f}"
        print(row)
    print("=" * 78)
    print(f"  study wall-clock: {(time.time()-t_study)/60:.1f} min")

    os.makedirs(results_dir, exist_ok=True)
    path = f"{results_dir}/scaling_{name}{out_tag}.json"
    with open(path, "w") as f:
        json.dump({
            "study": name,
            "base_config": asdict(base_cfg),
            "seeds": list(seeds),
            "methods": methods,
            "points": points,
        }, f, indent=2, default=str)
    print(f"  saved -> {path}\n")
    return points


def default_base() -> ExperimentConfig:
    base = ExperimentConfig(mode=BASE_MODE, **MODES[BASE_MODE])
    base.mode = BASE_MODE
    return base


def build_arms(study):
    if study == "samples":
        return [{
            "label": f"ep={n}",
            "overrides": dict(episodes_per_client=n, steps_per_episode=BASE_STEPS,
                              goal_pool_size=None),
        } for n in STUDY_A_EPISODES]
    return [{
        "label": f"goals={g}",
        "overrides": dict(episodes_per_client=BASE_EPISODES, steps_per_episode=BASE_STEPS,
                          goal_pool_size=g),
    } for g in STUDY_B_GOAL_POOLS]


def main():
    ap = argparse.ArgumentParser(description="VLA federated scaling study")
    ap.add_argument("--study", choices=["samples", "goals", "all"], default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="fedavg_full",
                    help="comma separated; any of " + ",".join(METHODS))
    ap.add_argument("--results_dir", default="results/vla_fed")
    ap.add_argument("--out_tag", default="",
                    help="suffix for the output file, e.g. '_localonly'")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [m for m in args.methods.split(",") if m]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"unknown method(s) {bad}; valid: {list(METHODS)}")
    base = default_base()

    print(f"  base = mode '{BASE_MODE}', {base.n_clients} clients, {base.rounds} rounds, "
          f"{base.local_epochs} local epochs, {base.num_action_bins} bins, "
          f"action_noise={base.action_noise}")
    print(f"  (steps fixed at {BASE_STEPS}; Study B holds samples at "
          f"{BASE_EPISODES}x{BASE_STEPS}={BASE_EPISODES*BASE_STEPS} per client)\n")

    studies = ["samples", "goals"] if args.study == "all" else [args.study]
    all_pts = {}
    for st in studies:
        all_pts[st] = run_study(st, base, build_arms(st), seeds, methods,
                                args.results_dir, args.out_tag)

    print("=" * 78)
    print("  ALL DONE")
    for st, pts in all_pts.items():
        print(f"    {st}: " + "  ".join(
            f"{p['label']}->{p['summary'][methods[0]]['mean']:.3f}" for p in pts))
    print("=" * 78)


if __name__ == "__main__":
    main()
