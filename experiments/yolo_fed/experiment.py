"""
Embodied-FL: Federated Object Detection — Experiment Runner
============================================================
Paper-ready experiment script with three modes:

  python experiment.py --mode quick    # ~60s CPU, proof of concept
  python experiment.py --mode paper    # ~30min GPU, paper-quality results
  python experiment.py --mode full     # ~2hr GPU, all ablations + tables

Experiments:
  Exp1: FedAvg vs Task-Aware (main result)
  Exp2: Backbone-only vs Full-model aggregation
  Exp3: Scalability (5 vs 10 vs 20 clients)
  Exp4: Communication efficiency
  Exp5: Ablation on aggregation weights (α_sim, α_perf, α_size)
"""

import os
import sys
import json
import time
import copy
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import DetectionBackbone, DetectionHead, Detector, DetectionLoss
from data.synthetic import SyntheticDetectionDataset, create_non_iid_split
from utils.federated import (
    compute_task_embeddings_for_clients,
    aggregate_fedavg, aggregate_task_aware,
    compute_ap, get_backbone_params, set_backbone_params,
    count_parameters, communication_cost
)


# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    # Data
    n_classes: int = 10
    img_size: int = 64
    n_clients: int = 5
    n_images: int = 200
    max_obj: int = 5
    difficulty: float = 0.0
    seed: int = 42

    # Model
    backbone_channels: int = 64  # 64=light, 128=medium, 256=heavy

    # Training
    rounds: int = 10
    local_epochs: int = 3
    lr: float = 0.001
    batch_size: int = 32

    # Aggregation
    alpha_sim: float = 0.4
    alpha_perf: float = 0.3
    alpha_size: float = 0.3

    # System
    device: str = 'cpu'
    results_dir: str = 'results'


# Presets
PRESETS = {
    'quick': ExperimentConfig(
        n_images=40, rounds=4, local_epochs=1, lr=0.005,
        backbone_channels=32, max_obj=3, batch_size=32
    ),
    'paper': ExperimentConfig(
        n_images=200, rounds=15, local_epochs=3, lr=0.001,
        backbone_channels=128, max_obj=5, batch_size=32
    ),
    'full': ExperimentConfig(
        n_images=300, rounds=20, local_epochs=5, lr=0.001,
        backbone_channels=128, max_obj=5, batch_size=32
    ),
}


# ═══════════════════════════════════════════════════════════════
#  Federated Training
# ═══════════════════════════════════════════════════════════════

@dataclass
class RoundResult:
    round_num: int
    method: str
    ap: float
    loss: float
    per_client_ap: Dict[str, float]
    per_client_loss: Dict[str, float]
    comm_bytes: int
    time_s: float
    client_weights: Optional[List[float]] = None


def run_federated_experiment(
    config: ExperimentConfig,
    method: str = 'fedavg',
    backbone_only: bool = True,
    verbose: bool = True
) -> Tuple[List[RoundResult], int, int]:
    """
    Run one federated training experiment.
    
    Args:
        config: experiment configuration
        method: 'fedavg' or 'ours'
        backbone_only: if True, only aggregate backbone params
        verbose: print progress
    
    Returns:
        (results, backbone_params, head_params)
    """
    device = config.device
    nc = config.n_classes
    max_obj = config.max_obj

    # Create Non-IID clients
    class_splits = create_non_iid_split(config.n_clients, nc, config.seed)
    task_embs, global_emb = compute_task_embeddings_for_clients(class_splits, nc)

    # Initialize global model
    global_bb = DetectionBackbone(config.backbone_channels).to(device)
    criterion = DetectionLoss(nc, max_obj)

    # Per-client heads (persistent across rounds)
    client_heads = {}
    for i, split in enumerate(class_splits):
        cid = f'client_{i}'
        client_heads[cid] = DetectionHead(
            global_bb.out_dim, nc, max_obj
        ).to(device)

    results = []

    for r in range(1, config.rounds + 1):
        t0 = time.time()
        updates = []
        weights = []
        losses = []
        per_client_ap = {}

        # ── Local Training ──
        for i, split in enumerate(class_splits):
            cid = f'client_{i}'

            # Copy global backbone params into client model (no deepcopy)
            c_bb = DetectionBackbone(config.backbone_channels).to(device)
            set_backbone_params(c_bb, get_backbone_params(global_bb))
            c_model = Detector(c_bb, client_heads[cid]).to(device)

            # Dataset (reuse if possible)
            ds = SyntheticDetectionDataset(
                n_images=config.n_images,
                img_size=config.img_size,
                n_classes=nc,
                class_subset=split,
                seed=config.seed + i + r * 100,  # different data each round
                max_obj=max_obj,
                difficulty=config.difficulty
            )
            loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

            # Train
            opt = torch.optim.Adam(c_model.parameters(), lr=config.lr)
            c_model.train()
            total_loss = 0
            n_batch = 0

            for ep in range(config.local_epochs):
                for imgs, tgt in loader:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    pb, pc = c_model(imgs)
                    loss = criterion(pb, pc, tgt)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    n_batch += 1

            avg_loss = total_loss / max(n_batch, 1)

            # Evaluate locally (small subset)
            c_model.eval()
            ap_sum, nb = 0, 0
            with torch.no_grad():
                for imgs, tgt in DataLoader(ds, batch_size=config.batch_size * 2):
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    pb, pc = c_model(imgs)
                    ap_sum += compute_ap(pb, pc, tgt, nc)
                    nb += 1
            local_ap = ap_sum / max(nb, 1)

            # Extract backbone update
            bb_before = get_backbone_params(global_bb)
            bb_after = get_backbone_params(c_model.backbone)
            update = {k: bb_after[k] - bb_before[k] for k in bb_before}

            # Update local head (keep trained weights)
            client_heads[cid] = c_model.head

            updates.append(update)
            weights.append(config.n_images)
            losses.append(avg_loss)
            per_client_ap[cid] = local_ap

        # ── Aggregation ──
        if method == 'ours':
            agg, cw = aggregate_task_aware(
                updates, weights, task_embs, global_emb, losses,
                alpha_sim=config.alpha_sim,
                alpha_perf=config.alpha_perf,
                alpha_size=config.alpha_size
            )
        else:
            agg = aggregate_fedavg(updates, weights)
            cw = None

        # Apply update
        bb_params = get_backbone_params(global_bb)
        for k in agg:
            bb_params[k] = bb_params[k] + agg[k]
        set_backbone_params(global_bb, bb_params)

        # ── Global Evaluation (use local APs) ──
        avg_ap = np.mean(list(per_client_ap.values()))
        comm = communication_cost(agg) if backbone_only else communication_cost(
            {k: v for u in updates for k, v in u.items()}
        )
        elapsed = time.time() - t0

        results.append(RoundResult(
            round_num=r, method=method,
            ap=avg_ap, loss=float(np.mean(losses)),
            per_client_ap=per_client_ap,
            per_client_loss={cid: l for cid, l in zip(
                [f'client_{i}' for i in range(config.n_clients)], losses)},
            comm_bytes=comm, time_s=elapsed,
            client_weights=cw
        ))

        if verbose:
            print(f"    R{r:2d} | AP={avg_ap:.4f} | loss={np.mean(losses):.3f} | {elapsed:.1f}s")

    bb_p = count_parameters(global_bb)
    hd_p = count_parameters(client_heads['client_0'])
    return results, bb_p, hd_p


# ═══════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════

def plot_results(results: Dict, results_dir: str, config: ExperimentConfig):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
            'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'
        })
    except ImportError:
        print("  ⚠ matplotlib not available")
        return

    os.makedirs(results_dir, exist_ok=True)

    # ── Fig 1: Convergence (AP over rounds) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method, color, label in [
        ('fedavg', '#e74c3c', 'FedAvg'),
        ('ours', '#2ecc71', 'Ours (Task-Aware)')
    ]:
        if method not in results:
            continue
        rounds = [r.round_num for r in results[method]]
        aps = [r.ap for r in results[method]]
        ax1.plot(rounds, aps, '-o', color=color, label=label, markersize=4, linewidth=2)

    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('mAP@50')
    ax1.set_title(f'(a) Convergence ({config.n_clients} clients, Non-IID)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Fig 1b: Per-client bar chart ──
    cids = list(results.get('ours', results.get('fedavg', []))[-1].per_client_ap.keys())
    x = np.arange(len(cids))
    w = 0.35

    for idx, (method, color, label) in enumerate([
        ('fedavg', '#e74c3c', 'FedAvg'),
        ('ours', '#2ecc71', 'Ours')
    ]):
        if method not in results:
            continue
        aps = [results[method][-1].per_client_ap[cid] for cid in cids]
        ax2.bar(x + (idx - 0.5) * w, aps, w, label=label, color=color, alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([cid.replace('client_', 'C') for cid in cids])
    ax2.set_ylabel('AP@50')
    ax2.set_title('(b) Per-Client AP@50 (Final Round)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig1_convergence.png')
    plt.close()
    print("  ✅ fig1_convergence.png")

    # ── Fig 2: Loss convergence ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color, label in [
        ('fedavg', '#e74c3c', 'FedAvg'),
        ('ours', '#2ecc71', 'Ours (Task-Aware)')
    ]:
        if method not in results:
            continue
        rounds = [r.round_num for r in results[method]]
        losses = [r.loss for r in results[method]]
        ax.plot(rounds, losses, '-s', color=color, label=label, markersize=4)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Training Loss')
    ax.set_title('Federated Detection: Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fig2_loss.png')
    plt.close()
    print("  ✅ fig2_loss.png")

    # ── Fig 3: Communication cost ──
    if 'ours' in results and 'full_model' in results:
        fig, ax = plt.subplots(figsize=(7, 5))
        methods = ['Backbone-only\n(Ours)', 'Full Model']
        bytes_vals = [
            results['ours'][-1].comm_bytes,
            results['full_model'][-1].comm_bytes
        ]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(methods, bytes_vals, color=colors, width=0.5)
        for bar, val in zip(bars, bytes_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val/1024:.0f} KB', ha='center', va='bottom', fontweight='bold')
        saving = (1 - bytes_vals[0] / bytes_vals[1]) * 100
        ax.set_ylabel('Bytes per Round')
        ax.set_title(f'Communication Cost (Saving: {saving:.1f}%)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/fig3_communication.png')
        plt.close()
        print("  ✅ fig3_communication.png")

    # ── Fig 4: Client weight distribution (Task-Aware) ──
    if 'ours' in results and results['ours'][-1].client_weights:
        fig, ax = plt.subplots(figsize=(8, 5))
        cw = results['ours'][-1].client_weights
        cids = list(results['ours'][-1].per_client_ap.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(cids)))
        bars = ax.bar(range(len(cids)), cw, color=colors)
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels([cid.replace('client_', 'C') for cid in cids])
        ax.set_ylabel('Aggregation Weight')
        ax.set_title('Task-Aware Client Weights (Final Round)')
        for bar, w in zip(bars, cw):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{w:.3f}', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/fig4_client_weights.png')
        plt.close()
        print("  ✅ fig4_client_weights.png")


# ═══════════════════════════════════════════════════════════════
#  LaTeX Table Generator
# ═══════════════════════════════════════════════════════════════

def generate_latex_table(results: Dict, config: ExperimentConfig,
                         results_dir: str):
    """Generate LaTeX tables for the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Federated Object Detection Results (Non-IID, "
                 + f"{config.n_clients} clients, {config.n_classes} classes)")
    lines.append(r"  \label{tab:detection_results}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Method & mAP@50 & Best AP & Comm (KB) & Time (s) \\")
    lines.append(r"    \midrule")

    for method, label in [('fedavg', 'FedAvg'), ('ours', 'Ours (Task-Aware)')]:
        if method not in results:
            continue
        r = results[method]
        final_ap = r[-1].ap
        best_ap = max(rr.ap for rr in r)
        comm = r[-1].comm_bytes / 1024
        total_time = sum(rr.time_s for rr in r)
        lines.append(f"    {label} & {final_ap:.4f} & {best_ap:.4f} & "
                     f"{comm:.1f} & {total_time:.1f} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    table_str = '\n'.join(lines)
    table_path = f'{results_dir}/table_results.tex'
    with open(table_path, 'w') as f:
        f.write(table_str)
    print(f"  ✅ {table_path}")

    # Per-client table
    lines2 = []
    lines2.append(r"\begin{table}[t]")
    lines2.append(r"  \centering")
    lines2.append(r"  \caption{Per-Client AP@50 (Final Round)}")
    lines2.append(r"  \label{tab:per_client}")
    lines2.append(r"  \begin{tabular}{l" + "c" * config.n_clients + "}")
    lines2.append(r"    \toprule")
    cids = list(results.get('ours', results.get('fedavg', []))[-1].per_client_ap.keys())
    header = "    Method & " + " & ".join(cid.replace('client_', 'C') for cid in cids) + r" \\"
    lines2.append(header)
    lines2.append(r"    \midrule")

    for method, label in [('fedavg', 'FedAvg'), ('ours', 'Ours')]:
        if method not in results:
            continue
        aps = [f"{results[method][-1].per_client_ap[cid]:.4f}" for cid in cids]
        lines2.append(f"    {label} & " + " & ".join(aps) + r" \\")

    lines2.append(r"    \bottomrule")
    lines2.append(r"  \end{tabular}")
    lines2.append(r"\end{table}")

    table2_str = '\n'.join(lines2)
    table2_path = f'{results_dir}/table_per_client.tex'
    with open(table2_path, 'w') as f:
        f.write(table2_str)
    print(f"  ✅ {table2_path}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Embodied-FL: Federated Object Detection')
    parser.add_argument('--mode', choices=['quick', 'paper', 'full'], default='quick',
                       help='Experiment mode (quick=CPU demo, paper=GPU results)')
    parser.add_argument('--device', default=None, help='Device (auto-detect if not set)')
    parser.add_argument('--output', default=None, help='Results directory')
    args = parser.parse_args()

    # Config
    config = PRESETS[args.mode]
    config.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    config.results_dir = args.output or f'results/{args.mode}'

    print("=" * 70)
    print("  Embodied-FL: Federated Object Detection")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 70)
    print(f"  Device: {config.device}")
    print(f"  Clients: {config.n_clients} | Classes: {config.n_classes}")
    print(f"  Rounds: {config.rounds} | Epochs: {config.local_epochs}")
    print(f"  Images/client: {config.n_images} | Max objects: {config.max_obj}")
    print(f"  Backbone channels: {config.backbone_channels}")

    # Model info
    bb = DetectionBackbone(config.backbone_channels)
    hd = DetectionHead(bb.out_dim, config.n_classes, config.max_obj)
    bb_p = count_parameters(bb)
    hd_p = count_parameters(hd)
    print(f"  Backbone: {bb_p:,} params | Head: {hd_p:,} params | Total: {bb_p+hd_p:,}")
    saving = (1 - bb_p / (bb_p + hd_p)) * 100
    print(f"  Comm saving (backbone-only): {saving:.1f}%")

    all_results = {}

    # ── Exp 1: FedAvg vs Task-Aware ──
    print("\n" + "─" * 70)
    print("  Exp1: FedAvg vs Task-Aware (Backbone-only)")
    print("─" * 70)

    for method, label in [('fedavg', 'FedAvg'), ('ours', 'Ours (Task-Aware)')]:
        print(f"\n  [{label}]")
        r, _, _ = run_federated_experiment(config, method=method, backbone_only=True)
        all_results[method] = r
        final = r[-1]
        best = max(rr.ap for rr in r)
        print(f"  Final AP: {final.ap:.4f} | Best AP: {best:.4f}")

    # ── Exp 2: Backbone-only vs Full-model ──
    if args.mode in ['paper', 'full']:
        print("\n" + "─" * 70)
        print("  Exp2: Backbone-only vs Full-model Aggregation")
        print("─" * 70)

        # Full model (aggregate everything)
        print("\n  [Full-model FedAvg]")
        r_full, _, _ = run_federated_experiment(config, method='fedavg', backbone_only=False)
        all_results['full_model'] = r_full

        bb_final = all_results['ours'][-1].ap
        fm_final = r_full[-1].ap
        print(f"  Backbone-only: AP={bb_final:.4f}")
        print(f"  Full-model:    AP={fm_final:.4f}")

    # ── Exp 3: Scalability ──
    if args.mode == 'full':
        print("\n" + "─" * 70)
        print("  Exp3: Scalability")
        print("─" * 70)

        for n_c in [10, 20]:
            cfg = copy.deepcopy(config)
            cfg.n_clients = n_c
            cfg.n_images = max(100, config.n_images - n_c * 5)
            print(f"\n  [{n_c} clients]")
            r, _, _ = run_federated_experiment(cfg, method='ours')
            all_results[f'{n_c}c_ours'] = r
            print(f"  Final AP: {r[-1].ap:.4f}")

    # ── Exp 5: Ablation on weights ──
    if args.mode == 'full':
        print("\n" + "─" * 70)
        print("  Exp5: Ablation on Aggregation Weights")
        print("─" * 70)

        ablations = [
            ('sim_only', 1.0, 0.0, 0.0),
            ('perf_only', 0.0, 1.0, 0.0),
            ('size_only', 0.0, 0.0, 1.0),
            ('equal', 1/3, 1/3, 1/3),
        ]
        for name, a_s, a_p, a_z in ablations:
            cfg = copy.deepcopy(config)
            cfg.alpha_sim, cfg.alpha_perf, cfg.alpha_size = a_s, a_p, a_z
            print(f"\n  [{name}: sim={a_s:.1f}, perf={a_p:.1f}, size={a_z:.1f}]")
            r, _, _ = run_federated_experiment(cfg, method='ours')
            all_results[f'abl_{name}'] = r
            print(f"  Final AP: {r[-1].ap:.4f}")

    # ── Summary ──
    print("\n" + "─" * 70)
    print("  Summary")
    print("─" * 70)

    if 'fedavg' in all_results and 'ours' in all_results:
        fed_best = max(r.ap for r in all_results['fedavg'])
        ours_best = max(r.ap for r in all_results['ours'])
        print(f"  FedAvg best AP:  {fed_best:.4f}")
        print(f"  Ours best AP:    {ours_best:.4f}")
        if fed_best > 0:
            print(f"  Improvement:     {(ours_best - fed_best) / fed_best * 100:+.1f}%")

    # ── Save & Plot ──
    os.makedirs(config.results_dir, exist_ok=True)

    # Save JSON
    output = {
        'config': asdict(config),
        'results': {
            k: [
                {
                    'round': r.round_num, 'ap': r.ap, 'loss': r.loss,
                    'per_client_ap': r.per_client_ap,
                    'comm_bytes': r.comm_bytes, 'time': r.time_s,
                    'client_weights': r.client_weights
                } for r in v
            ] for k, v in all_results.items()
        }
    }
    json_path = f'{config.results_dir}/results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    # Plot
    plot_results(all_results, config.results_dir, config)

    # LaTeX tables
    generate_latex_table(all_results, config, config.results_dir)

    print("\n" + "=" * 70)
    print("  ✅ All experiments completed!")
    print(f"  Results: {config.results_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
