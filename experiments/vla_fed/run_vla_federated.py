# ── experiments/vla_fed/run_vla_federated.py ──
"""
VLA Federated Learning — End-to-End Imitation Experiment
=========================================================
Synthetic multi-robot VLA federated training.  *Simulation*, not a real-robot
result: the "vision" channel is a simulated encoder output, see VISION below.

Pipeline:
  1. Generate synthetic VLA episodes per client (robot / factory)
  2. Parse instructions -> language embeddings (per-client constant)
  3. Fit a *shared* action tokenizer on pooled TRAIN actions (quantile bins)
  4. Build VLADataset per client, split train/test BY EPISODE
  5. Baselines + federated training (local epochs -> sample-weighted FedAvg)
  6. Evaluate every method on held-out episodes only

VISION
------
`--vision scene` (default): the visual feature is a fixed random projection of
the observable scene descriptor [robot_state, goal_pose, gripper_aperture] plus
Gaussian noise — i.e. a *simulated* visual encoder that can see the workspace.
`--vision random`: pure i.i.d. noise (the original v2 behaviour). Kept as an
ablation; with this setting the task is unlearnable by construction and every
method sits at the trivial floor.

BINNING
-------
`--binning quantile` (default): per-dimension uniform bins between the 1st and
99th percentile of the pooled train actions (the RT-1 / Octo convention).
`--binning uniform`: fixed [-1, 1] bins (the original v2 behaviour), which puts
~1/3 of all actions in a single central bin and makes a constant predictor look
strong.

Modes:
  python run_vla_federated.py --mode quick    # seconds, smoke test
  python run_vla_federated.py --mode paper    # minutes, reportable
  python run_vla_federated.py --mode full     # longer, all clients
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.vla_collector import SyntheticCollector, compute_episode_statistics
from analysis.vla_dataset import VLADataset
from analysis.action_tokenizer import ActionTokenizer, TokenizerConfig
from analysis.instruction_parser import InstructionParser
from analysis.instruction_embedding import InstructionEmbedder, EmbeddingConfig
from analysis.vla_model import VLAFLModel, VLAFLTrainer, VLAConfig


# ── Experiment Configuration ──

@dataclass
class ExperimentConfig:
    mode: str = "quick"
    n_clients: int = 3
    rounds: int = 10
    local_epochs: int = 3
    episodes_per_client: int = 10
    steps_per_episode: int = 20
    d_model: int = 64
    n_heads: int = 2
    n_fusion_layers: int = 1
    lr: float = 3e-3
    action_dim: int = 8
    num_action_bins: int = 32
    batch_size: int = 64
    vision_dim: int = 384
    lang_dim: int = 384
    state_dim: int = 7
    seed: int = 42
    results_dir: str = "results/vla_fed"
    # newly explicit experimental knobs
    action_noise: float = 0.005     # per-step Gaussian noise on the commanded delta
                                    # (|clean delta| ~ 0.073; see sweep in docs)
    vision_mode: str = "scene"      # "scene" | "random"
    binning: str = "quantile"       # "quantile" | "uniform"
    test_episode_stride: int = 5    # every k-th episode -> test split
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    # Goal-pool control (used by run_scaling.py; None = original behaviour of
    # drawing a fresh goal per episode). Setting `goal_pool_size = G` makes the
    # episodes cycle through G distinct goals, which holds the *number of
    # distinct goals* fixed independently of the sample budget.
    goal_pool_size: Optional[int] = None
    goal_pool_seed: int = 12345


@dataclass
class ClientConfig:
    client_id: str
    robot_type: str
    task_type: str
    instruction: str
    state_dim: int = 7
    action_dim: int = 8


CLIENT_SCENARIOS = [
    ClientConfig("factory_a_smt", "franka_panda", "grasping",
                 "pick up the electronic component from the conveyor belt"),
    ClientConfig("factory_b_assembly", "ur5e", "assembly",
                 "insert the connector into the phone frame"),
    ClientConfig("factory_c_inspect", "franka_panda", "inspection",
                 "inspect the PCB for solder defects"),
    ClientConfig("warehouse_a_pick", "ur5e", "grasping",
                 "pick up the package from the shelf"),
    ClientConfig("warehouse_b_place", "franka_panda", "placing",
                 "place the box on the pallet"),
]


# ── Shared action tokenizer ──

class QuantileActionTokenizer(ActionTokenizer):
    """Action tokenizer with per-dimension quantile bin edges.

    Edges are computed once on pooled *training* actions and shared by every
    client, which keeps the FedAvg action head well-defined across clients.
    """

    def __init__(self, train_actions: np.ndarray, num_bins: int = 32,
                 clip_quantile: float = 0.01):
        cfg = TokenizerConfig(
            action_dim=train_actions.shape[1],
            num_bins=num_bins,
            low=-1.0, high=1.0,
        )
        super().__init__(cfg)
        edges = []
        for d in range(train_actions.shape[1]):
            col = train_actions[:, d]
            lo = float(np.quantile(col, clip_quantile))
            hi = float(np.quantile(col, 1.0 - clip_quantile))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(col.min()), float(col.max())
            if hi <= lo:
                lo, hi = lo - 1e-3, hi + 1e-3
            edges.append(np.linspace(lo, hi, num_bins + 1, dtype=np.float32))
        self._bin_edges = edges


def make_tokenizer(pooled_train_actions: np.ndarray, cfg: ExperimentConfig) -> ActionTokenizer:
    if cfg.binning == "quantile":
        return QuantileActionTokenizer(pooled_train_actions, cfg.num_action_bins)
    tok = ActionTokenizer(TokenizerConfig(
        action_dim=pooled_train_actions.shape[1],
        num_bins=cfg.num_action_bins,
        low=-1.0, high=1.0,
    ))
    return tok


# ── Data generation ──

def build_client_dataset(client: ClientConfig, cfg: ExperimentConfig, seed: int) -> VLADataset:
    """Generate the raw VLA dataset for a single client.

    The *simulation* dimensions are governed by `cfg` so that the synthetic
    state/action vectors line up with the model config exactly.
    """
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
    )
    return VLADataset.from_episodes(episodes, skip_no_action=False)


def pad_or_truncate(arr: np.ndarray, dim: int) -> np.ndarray:
    if arr.shape[1] < dim:
        pad = np.zeros((arr.shape[0], dim), dtype=np.float32)
        pad[:, :arr.shape[1]] = arr
        return pad
    if arr.shape[1] > dim:
        return arr[:, :dim]
    return arr


def split_by_episode(samples: List, stride: int) -> Tuple[List, List]:
    """Episode-wise train/test split. Prevents adjacent-frame leakage."""
    eids = [s.episode_id for s in samples]
    uniq = sorted(set(eids))
    test_eps = set(uniq[::stride]) if stride > 1 else set()
    tr = [s for s in samples if s.episode_id not in test_eps]
    te = [s for s in samples if s.episode_id in test_eps]
    if not te:                                  # degenerate small configs
        tr, te = samples[:-1], samples[-1:]
    if not tr:
        tr, te = te, te
    return tr, te


class SceneEncoder:
    """Fixed (non-learned) random projection standing in for a visual encoder."""

    def __init__(self, desc_dim: int, vision_dim: int, seed: int = 0,
                 noise: float = 0.05):
        rng = np.random.RandomState(seed)
        self.W = (rng.randn(desc_dim, vision_dim) / np.sqrt(desc_dim)).astype(np.float32)
        self.noise = noise

    def __call__(self, desc: np.ndarray) -> np.ndarray:
        h = desc @ self.W
        return h + np.random.RandomState(1234).randn(*h.shape).astype(np.float32) * self.noise


def build_tensors(samples: List, cfg: ExperimentConfig, embedder: InstructionEmbedder,
                  tokenizer: ActionTokenizer, scene: SceneEncoder) -> Dict[str, torch.Tensor]:
    """Convert a list of VLASample into the tensors the model consumes."""
    N = len(samples)

    states = np.array([s.robot_state for s in samples], dtype=np.float32)
    states = pad_or_truncate(states, cfg.state_dim)

    actions = np.array([s.action for s in samples], dtype=np.float32)
    actions = pad_or_truncate(actions, cfg.action_dim)

    grippers = np.array([[s.gripper] for s in samples], dtype=np.float32)
    if any(s.goal is not None for s in samples):
        goals = np.array([
            s.goal if s.goal is not None else np.zeros(cfg.state_dim, dtype=np.float32)
            for s in samples
        ], dtype=np.float32)
        goals = pad_or_truncate(goals, cfg.state_dim)
    else:
        goals = np.zeros((N, cfg.state_dim), dtype=np.float32)

    desc = np.concatenate([states, goals, grippers], axis=1)

    if cfg.vision_mode == "scene":
        vision = scene(desc)
    else:                                       # "random" ablation
        vision = np.random.RandomState(cfg.seed).randn(N, cfg.vision_dim).astype(np.float32)

    instructions = [s.instruction for s in samples]
    lang = embedder.embed_batch(instructions).astype(np.float32)
    lang = lang.reshape(N, 1, -1)

    tokens = tokenizer.encode_batch(actions)

    return {
        "vision": torch.from_numpy(vision).float(),
        "lang": torch.from_numpy(lang).float(),
        "state": torch.from_numpy(states).float(),
        "tokens": torch.from_numpy(tokens).long(),
        "actions": torch.from_numpy(actions).float(),
    }


# ── Evaluation ──

@torch.no_grad()
def token_accuracy(model: VLAFLModel, bundle: Dict[str, torch.Tensor]) -> float:
    model.eval()
    logits = model(bundle["vision"], bundle["lang"], bundle["state"])
    pred = logits.argmax(dim=-1)
    return float((pred == bundle["tokens"]).float().mean().item())


def majority_predictor_accuracy(tr: Dict[str, torch.Tensor], te: Dict[str, torch.Tensor]) -> float:
    """Per-dimension most frequent train token evaluated on test."""
    tr_t, te_t = tr["tokens"].numpy(), te["tokens"].numpy()
    A = tr_t.shape[1]
    hit = 0
    for a in range(A):
        vals, cnts = np.unique(tr_t[:, a], return_counts=True)
        hit += int((te_t[:, a] == vals[cnts.argmax()]).sum())
    return hit / (te_t.shape[0] * A)


# ── Federated training ──

def fedavg(updates: List[dict], sizes: List[int]) -> dict:
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("Total sample count is zero, cannot aggregate")
    return {
        k: sum(u[k] * (s / total) for u, s in zip(updates, sizes))
        for k in updates[0]
    }


def run_federated(cfg: ExperimentConfig, vla_cfg: VLAConfig,
                  train_bundles: List[Dict], test_bundles: List[Dict],
                  method: str, seed: int) -> Dict:
    """method in {"fedavg_backbone", "fedavg_full"}.

    Anything else is rejected rather than silently treated as
    `fedavg_backbone` — an unknown method used to degrade silently, which
    produced a column that looked plausible and was simply the wrong method.
    `local_only` has its own entry point (`run_local_only`).
    """
    if method not in ("fedavg_backbone", "fedavg_full"):
        raise ValueError(
            f"unknown method {method!r}; expected 'fedavg_backbone' or "
            f"'fedavg_full' (use run_local_only for the no-federation arm)"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)

    trainers = [VLAFLTrainer(vla_cfg) for _ in train_bundles]
    head_shared = method == "fedavg_full"
    global_model = VLAFLModel(vla_cfg)
    global_params = (global_model.get_full_state_dict() if head_shared
                     else global_model.get_shared_state_dict())

    history = []
    for rnd in range(cfg.rounds):
        t0 = time.time()
        updates, sizes, local_losses = [], [], []
        for k, tr in enumerate(train_bundles):
            if head_shared:
                trainers[k].model.load_full_params(global_params)
            else:
                trainers[k].model.load_shared_params(global_params)

            res = trainers[k].train_local(
                tr["vision"], tr["lang"], tr["state"], tr["tokens"],
                n_epochs=cfg.local_epochs, batch_size=cfg.batch_size,
            )
            updates.append(trainers[k].model.get_full_state_dict() if head_shared
                           else trainers[k].model.get_shared_state_dict())
            sizes.append(int(tr["tokens"].shape[0]))
            local_losses.append(res["final_loss"])

        global_params = fedavg(updates, sizes)

        # Evaluate on HELD-OUT episodes
        per_client = []
        for k, te in enumerate(test_bundles):
            if head_shared:
                global_model.load_full_params(global_params)
                acc = token_accuracy(global_model, te)
            else:
                # global backbone + this client's own locally trained head
                trainers[k].model.load_shared_params(global_params)
                acc = token_accuracy(trainers[k].model, te)
            per_client.append(acc)

        history.append({
            "round": rnd + 1,
            "test_accuracy": float(np.mean(per_client)),
            "per_client_test_accuracy": [float(a) for a in per_client],
            "avg_local_loss": float(np.mean(local_losses)),
            "elapsed": time.time() - t0,
        })

    return {
        "method": method,
        "history": history,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "first_test_accuracy": history[0]["test_accuracy"],
        "per_client_final": history[-1]["per_client_test_accuracy"],
    }


def run_local_only(cfg: ExperimentConfig, vla_cfg: VLAConfig,
                   train_bundles: List[Dict], test_bundles: List[Dict],
                   seed: int) -> Dict:
    """No federation: each client trains alone for the same total local budget."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    total_epochs = cfg.rounds * cfg.local_epochs
    per_client = []
    for k, tr in enumerate(train_bundles):
        trainer = VLAFLTrainer(vla_cfg)
        trainer.train_local(tr["vision"], tr["lang"], tr["state"], tr["tokens"],
                            n_epochs=total_epochs, batch_size=cfg.batch_size)
        per_client.append(token_accuracy(trainer.model, test_bundles[k]))
    return {
        "method": "local_only",
        "history": [],
        "final_test_accuracy": float(np.mean(per_client)),
        "first_test_accuracy": float("nan"),
        "per_client_final": [float(a) for a in per_client],
        "total_local_epochs": total_epochs,
    }


# ── Main ──

MODES = {
    # NOTE ON THE BUDGET. Two controlled studies back these values
    # (`run_scaling.py`, 3 seeds each; raw output in
    # `results/vla_fed/scaling_samples.json` and `scaling_goals.json`):
    #
    #   Study A — episodes/client 50/100/200/400/800, steps FIXED at 12.
    #     Federated test accuracy 0.405 / 0.516 / 0.663 / 0.743 / 0.776.
    #     Monotone but clearly saturating: the last doubling buys only +3.3 pt.
    #
    #   Study B — sample budget HELD at 400x12 = 4800 per client, only the
    #     number of distinct goals varied (pool 25/50/100/200/400):
    #     0.633 / 0.658 / 0.736 / 0.738 / 0.744, against a floor that stays
    #     flat at 0.283-0.300 in every arm. At a fixed sample count, going
    #     from 20 to 80 distinct training goals is worth +10.3 pt, whereas
    #     repeating each goal 16 times instead of once is worth nothing.
    #
    # Coverage — not sample count — is the binding constraint, which is the
    # empirical motivation for FedCover-WM. Steps *within* an episode are
    # highly correlated, so the budget is spent on episodes, not step count.
    # 32 action bins also proved too fine for this sample size (0.26 vs 0.44
    # test accuracy at 16 bins, otherwise identical settings).
    "quick": dict(n_clients=3, rounds=8, local_epochs=3, episodes_per_client=120,
                  steps_per_episode=10, d_model=64, action_dim=5, num_action_bins=16,
                  vision_dim=64, lang_dim=64, state_dim=4, seeds=[0, 1]),
    "paper": dict(n_clients=5, rounds=12, local_epochs=3, episodes_per_client=400,
                  steps_per_episode=12, d_model=64, action_dim=8, num_action_bins=16,
                  vision_dim=384, lang_dim=384, state_dim=7, seeds=[0, 1, 2]),
    "full": dict(n_clients=5, rounds=25, local_epochs=5, episodes_per_client=800,
                 steps_per_episode=20, d_model=128, action_dim=8, num_action_bins=32,
                 vision_dim=384, lang_dim=384, state_dim=7, seeds=[0, 1, 2]),
}


def one_seed(cfg: ExperimentConfig, seed: int, verbose: bool = True) -> Dict:
    """Build data once per seed, then evaluate every method on it."""
    cfg.seed = seed
    clients = CLIENT_SCENARIOS[:cfg.n_clients]

    # 1. raw datasets
    datasets = []
    for i, client in enumerate(clients):
        ds = build_client_dataset(client, cfg, seed=seed * 100 + i)
        datasets.append(split_by_episode(ds.samples, cfg.test_episode_stride))

    # 2. shared tokenizer on pooled TRAIN actions
    pooled = np.concatenate([
        pad_or_truncate(np.array([s.action for s in tr], dtype=np.float32), cfg.action_dim)
        for tr, _ in datasets
    ])
    tokenizer = make_tokenizer(pooled, cfg)

    # 3. tensors
    embedder = InstructionEmbedder(EmbeddingConfig(mode="hash", dimension=cfg.lang_dim))
    scene = SceneEncoder(desc_dim=2 * cfg.state_dim + 1, vision_dim=cfg.vision_dim,
                         seed=7, noise=0.05)
    train_bundles, test_bundles, meta = [], [], []
    for (tr, te), client in zip(datasets, clients):
        train_bundles.append(build_tensors(tr, cfg, embedder, tokenizer, scene))
        test_bundles.append(build_tensors(te, cfg, embedder, tokenizer, scene))
        meta.append({
            "client_id": client.client_id,
            "robot_type": client.robot_type,
            "task_type": client.task_type,
            "n_train": len(tr),
            "n_test": len(te),
        })

    if verbose:
        print(f"    clients={len(clients)}  train/test samples="
              f"{[ (m['n_train'], m['n_test']) for m in meta ]}")

    vla_cfg = VLAConfig(
        vision_dim=cfg.vision_dim, lang_dim=cfg.lang_dim, state_dim=cfg.state_dim,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_fusion_layers=cfg.n_fusion_layers,
        action_dim=cfg.action_dim, num_action_bins=cfg.num_action_bins,
        lr=cfg.lr, local_epochs=cfg.local_epochs, batch_size=cfg.batch_size,
    )

    # 4. baselines + methods
    res = {}
    res["random"] = {
        "method": "random", "final_test_accuracy": 1.0 / cfg.num_action_bins,
        "history": [], "per_client_final": [], "note": "analytic uniform baseline",
    }
    maj = [majority_predictor_accuracy(train_bundles[k], test_bundles[k])
           for k in range(len(clients))]
    res["majority"] = {
        "method": "majority", "final_test_accuracy": float(np.mean(maj)),
        "history": [], "per_client_final": [float(a) for a in maj],
        "note": "per-dim most frequent training token (trivial predictor floor)",
    }

    for method in ["local_only", "fedavg_backbone", "fedavg_full"]:
        t0 = time.time()
        if method == "local_only":
            out = run_local_only(cfg, vla_cfg, train_bundles, test_bundles, seed)
        else:
            out = run_federated(cfg, vla_cfg, train_bundles, test_bundles, method, seed)
        out["elapsed_total"] = time.time() - t0
        res[method] = out
        if verbose:
            print(f"    {method:16s} test_acc={out['final_test_accuracy']:.4f} "
                  f"({time.time()-t0:.1f}s)")

    return {"seed": seed, "clients": meta, "methods": res}


def main():
    ap = argparse.ArgumentParser(description="VLA Federated Learning Experiment")
    ap.add_argument("--mode", choices=["quick", "paper", "full"], default="quick")
    ap.add_argument("--vision", choices=["scene", "random"], default="scene")
    ap.add_argument("--binning", choices=["quantile", "uniform"], default="quantile")
    ap.add_argument("--seeds", default=None, help="comma separated, e.g. 0,1,2")
    ap.add_argument("--results_dir", default="results/vla_fed")
    args = ap.parse_args()

    cfg = ExperimentConfig(mode=args.mode, results_dir=args.results_dir, **MODES[args.mode])
    cfg.vision_mode = args.vision
    cfg.binning = args.binning
    if args.seeds:
        cfg.seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 74)
    print(f"  VLA Federated Learning — {cfg.mode.upper()} mode")
    print(f"  clients={cfg.n_clients}  rounds={cfg.rounds}  local_epochs={cfg.local_epochs}  "
          f"bins={cfg.num_action_bins}")
    print(f"  vision={cfg.vision_mode}  binning={cfg.binning}  "
          f"action_noise={cfg.action_noise}  seeds={cfg.seeds}")
    print("=" * 74)

    runs = []
    for seed in cfg.seeds:
        print(f"\n  ── seed {seed} ──")
        runs.append(one_seed(cfg, seed))

    # aggregate over seeds
    methods = list(runs[0]["methods"].keys())
    summary = {}
    for m in methods:
        accs = [r["methods"][m]["final_test_accuracy"] for r in runs]
        summary[m] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "per_seed": [float(a) for a in accs],
        }

    floor = max(summary["random"]["mean"], summary["majority"]["mean"])
    print("\n" + "=" * 74)
    print(f"  RESULTS (held-out episodes, mean ± std over {len(cfg.seeds)} seeds)")
    print("=" * 74)
    print(f"  trivial floor = max(random, majority) = {floor:.4f}")
    for m in methods:
        s = summary[m]
        print(f"    {m:18s} {s['mean']:.4f} ± {s['std']:.4f}   per-seed={['%.3f'%a for a in s['per_seed']]}")
    print("=" * 74)

    os.makedirs(cfg.results_dir, exist_ok=True)
    tag = cfg.mode if (cfg.vision_mode == "scene" and cfg.binning == "quantile") \
        else f"{cfg.mode}_{cfg.vision_mode}_{cfg.binning}"
    path = f"{cfg.results_dir}/results_{tag}.json"
    with open(path, "w") as f:
        json.dump({
            "config": {**asdict(cfg)},
            "summary": summary,
            "trivial_floor": floor,
            "runs": runs,
        }, f, indent=2, default=str)
    print(f"\n  saved -> {path}")


if __name__ == "__main__":
    main()
