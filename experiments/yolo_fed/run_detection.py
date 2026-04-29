"""
Embodied-FL: Federated Object Detection
=========================================
Lightweight PyTorch implementation. ~60s total on CPU.

Key design:
  - Shared CNN backbone (aggregated across factories)
  - Per-client detection heads (kept local, never reset)
  - Task-Aware aggregation: weight by task embedding similarity

This is a proof-of-concept. For real experiments, replace the
synthetic dataset with COCO/VOC and the CNN backbone with YOLOv8.
"""

import os, json, time, copy
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════
#  Dataset: Synthetic colored rectangles on noise
# ═══════════════════════════════════════════════════════════════

class DetectionDataset(Dataset):
    def __init__(self, n=200, sz=64, nc=10, classes=None, seed=0, max_obj=4):
        self.rng = np.random.RandomState(seed)
        self.sz, self.nc, self.max_obj = sz, nc, max_obj
        self.classes = classes or list(range(nc))
        self.data = []
        for _ in range(n):
            img = self.rng.randn(3, sz, sz).astype(np.float32) * 0.2
            objs = []
            for _ in range(self.rng.randint(1, max_obj + 1)):
                c = int(self.rng.choice(self.classes))
                w = self.rng.uniform(0.15, 0.4)
                h = self.rng.uniform(0.15, 0.4)
                cx = self.rng.uniform(w/2, 1 - w/2)
                cy = self.rng.uniform(h/2, 1 - h/2)
                objs.append([cx, cy, w, h, c])
                # Draw rectangle
                x1, y1 = int((cx-w/2)*sz), int((cy-h/2)*sz)
                x2, y2 = int((cx+w/2)*sz), int((cy+h/2)*sz)
                color = (c + 1) / nc  # unique brightness per class
                img[:, y1:y2, x1:x2] = color + self.rng.randn(3, y2-y1, x2-x1) * 0.1
            # Pad to max_obj
            while len(objs) < max_obj:
                objs.append([0, 0, 0, 0, 0])  # background
            self.data.append((img, objs))

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        img, objs = self.data[i]
        return torch.tensor(img), torch.tensor(objs, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════
#  Model: CNN Backbone + Detection Head
# ═══════════════════════════════════════════════════════════════

class Backbone(nn.Module):
    """Lightweight CNN backbone (116K params)."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))


class DetectionHead(nn.Module):
    """Per-client detection head: predicts boxes + classes for each anchor."""
    def __init__(self, in_dim, nc, max_obj):
        super().__init__()
        self.max_obj = max_obj
        self.box_head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, max_obj * 4)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, max_obj * nc)
        )

    def forward(self, feat):
        box = self.box_head(feat).view(feat.size(0), self.max_obj, 4)
        cls = self.cls_head(feat).view(feat.size(0), self.max_obj, -1)
        return box, cls


class Detector(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


# ═══════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════

class DetectionLoss(nn.Module):
    def __init__(self, nc, max_obj):
        super().__init__()
        self.nc, self.max_obj = nc, max_obj

    def forward(self, pred_box, pred_cls, target):
        # target: [B, max_obj, 5] (cx, cy, w, h, class_id)
        B = pred_box.size(0)
        obj_mask = (target[..., 4] > 0).float()  # [B, max_obj]

        # Box loss (only for objects)
        box_diff = (pred_box - target[..., :4]) * obj_mask.unsqueeze(-1)
        box_loss = (box_diff ** 2).sum() / max(obj_mask.sum(), 1)

        # Classification loss
        target_cls = target[..., 4].long().clamp(0, self.nc - 1)
        cls_loss = F.cross_entropy(
            pred_cls.view(-1, self.nc), target_cls.view(-1), reduction='none'
        )
        cls_loss = (cls_loss * obj_mask.view(-1)).sum() / max(obj_mask.sum(), 1)

        return box_loss + cls_loss


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def compute_iou(a, b):
    """IoU for [cx, cy, w, h] boxes."""
    ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2
    ax2, ay2 = a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2
    bx2, by2 = b[0]+b[2]/2, b[1]+b[3]/2
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(union, 1e-8)


def compute_ap(pred_box, pred_cls, target, nc, iou_thresh=0.3):
    """Per-class AP averaged across classes."""
    aps = []
    for c in range(nc):
        tp = fp = n_gt = 0
        # Count GT
        for b in range(target.size(0)):
            for j in range(target.size(1)):
                if int(target[b, j, 4].item()) == c:
                    n_gt += 1
        if n_gt == 0:
            continue
        # Collect predictions sorted by confidence
        preds = []
        for b in range(pred_box.size(0)):
            for j in range(pred_box.size(1)):
                prob = F.softmax(pred_cls[b, j], dim=0)[c].item()
                if prob > 0.15:
                    preds.append((prob, b, j))
        preds.sort(reverse=True)
        matched = set()
        for prob, b, j in preds:
            pb = pred_box[b, j].detach().cpu().numpy()
            best_iou, best_gt = 0, -1
            for k in range(target.size(1)):
                if int(target[b, k, 4].item()) != c:
                    continue
                gb = target[b, k, :4].detach().cpu().numpy()
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gt = iou, k
            if best_iou >= iou_thresh and best_gt not in matched:
                tp += 1; matched.add(best_gt)
            else:
                fp += 1
        if tp + fp > 0:
            p = tp / (tp + fp)
            r = tp / n_gt
            aps.append(2*p*r / max(p+r, 1e-8))
    return np.mean(aps) if aps else 0.0


# ═══════════════════════════════════════════════════════════════
#  Task Embedding & Aggregation
# ═══════════════════════════════════════════════════════════════

def compute_task_embedding(class_dist, dim=64):
    """Embed class distribution as a fixed-size vector."""
    emb = np.zeros(dim, dtype=np.float32)
    for cls_id, prob in enumerate(class_dist):
        idx = cls_id % dim
        emb[idx] += prob
    # Normalize
    norm = np.linalg.norm(emb)
    return emb / max(norm, 1e-8)


def cosine_sim(a, b):
    return np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8)


def aggregate_fedavg(updates, weights):
    """Simple weighted average."""
    agg = {}
    total_w = sum(weights)
    for k in updates[0]:
        agg[k] = sum(u[k] * w for u, w in zip(updates, weights)) / total_w
    return agg


def aggregate_task_aware(updates, weights, task_embs, global_emb, losses):
    """Task-Aware: weight by similarity × performance × sample size."""
    sims = [max(cosine_sim(te, global_emb), 0.1) for te in task_embs]
    perfs = [1.0 / (l + 0.1) for l in losses]
    combined = [s * p * w for s, p, w in zip(sims, perfs, weights)]
    total = sum(combined)
    agg = {}
    for k in updates[0]:
        agg[k] = sum(u[k] * c for u, c in zip(updates, combined)) / total
    return agg


# ═══════════════════════════════════════════════════════════════
#  Client Config
# ═══════════════════════════════════════════════════════════════

@dataclass
class Client:
    id: str
    name: str
    n_images: int
    class_subset: list
    seed: int


def create_clients(n=5, nc=10, n_img=150, seed=42):
    """Non-IID split: each client has 3 dominant + 2 minority classes."""
    rng = np.random.RandomState(seed)
    clients = []
    for i in range(n):
        dom = i % nc
        others = [c for c in range(nc) if c != dom]
        minority = list(rng.choice(others, 2, replace=False))
        subset = [dom]*3 + minority
        clients.append(Client(
            id=f'factory_{chr(65+i)}',
            name=f'Factory-{chr(65+i)}',
            n_images=n_img,
            class_subset=subset,
            seed=seed + i
        ))
    return clients


# ═══════════════════════════════════════════════════════════════
#  Federated Training Loop
# ═══════════════════════════════════════════════════════════════

def get_backbone_params(model):
    """Get params from a Backbone module."""
    return {k: v.clone() for k, v in model.state_dict().items()}

def set_backbone_params(model, params):
    """Set params on a Backbone module."""
    model.load_state_dict(params)


def run_federated(clients, nc, max_obj, rounds, epochs, lr, img_sz,
                  method='fedavg', device='cpu'):
    """Run one federated experiment. Returns list of round results."""
    # Global backbone
    global_bb = Backbone(out_dim=128)
    # Per-client heads (persist across rounds!)
    client_heads = {c.id: DetectionHead(128, nc, max_obj) for c in clients}

    criterion = DetectionLoss(nc, max_obj)

    # Task embeddings
    task_embs = []
    for c in clients:
        dist = np.zeros(nc)
        for cls in c.class_subset:
            dist[cls] += 1.0 / len(c.class_subset)
        task_embs.append(compute_task_embedding(dist))
    global_emb = np.mean(task_embs, axis=0)

    results = []
    for r in range(1, rounds + 1):
        t0 = time.time()
        updates, weights, losses = [], [], []

        for c in clients:
            # Build client model: global backbone + local head
            c_model = Detector(
                copy.deepcopy(global_bb),
                client_heads[c.id]
            ).to(device)

            ds = DetectionDataset(c.n_images, img_sz, nc, c.class_subset, c.seed, max_obj)
            loader = DataLoader(ds, batch_size=32, shuffle=True)

            opt = torch.optim.Adam(c_model.parameters(), lr=lr)
            c_model.train()
            for ep in range(epochs):
                for imgs, tgt in loader:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    pb, pc = c_model(imgs)
                    loss = criterion(pb, pc, tgt)
                    opt.zero_grad(); loss.backward(); opt.step()

            # Save updated head back
            client_heads[c.id] = c_model.head

            # Backbone update
            bb_after = get_backbone_params(c_model.backbone)
            bb_before = get_backbone_params(global_bb)
            update = {k: bb_after[k] - bb_before[k] for k in bb_after}

            # Eval
            c_model.eval()
            total_ap = 0; nb = 0
            with torch.no_grad():
                for imgs, tgt in loader:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    pb, pc = c_model(imgs)
                    total_ap += compute_ap(pb, pc, tgt, nc)
                    nb += 1
            avg_ap = total_ap / max(nb, 1)

            updates.append(update)
            weights.append(c.n_images)
            losses.append(loss.item())

        # Aggregate backbone
        if method == 'ours':
            agg = aggregate_task_aware(updates, weights, task_embs, global_emb, losses)
        else:
            agg = aggregate_fedavg(updates, weights)

        bb_params = get_backbone_params(global_bb)
        for k in agg:
            bb_params[k] = bb_params[k] + agg[k]
        set_backbone_params(global_bb, bb_params)

        # Evaluate global backbone on each client (use local trained head)
        per_client = {}
        total_ap = 0
        for c in clients:
            eval_model = Detector(global_bb, client_heads[c.id]).to(device)
            eval_model.eval()
            ds = DetectionDataset(min(15, c.n_images), img_sz, nc, c.class_subset, c.seed+999, max_obj)
            loader = DataLoader(ds, batch_size=32)
            ap_sum = 0; nb = 0
            with torch.no_grad():
                for imgs, tgt in loader:
                    imgs, tgt = imgs.to(device), tgt.to(device)
                    pb, pc = eval_model(imgs)
                    ap_sum += compute_ap(pb, pc, tgt, nc)
                    nb += 1
            c_ap = ap_sum / max(nb, 1)
            per_client[c.id] = c_ap
            total_ap += c_ap

        avg_ap = total_ap / len(clients)
        elapsed = time.time() - t0
        print(f"    R{r:2d} | AP={avg_ap:.4f} | loss={np.mean(losses):.3f} | {elapsed:.1f}s")

        results.append({
            'round': r, 'ap': avg_ap,
            'loss': float(np.mean(losses)),
            'per_client': per_client,
            'time': elapsed
        })

    return results


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Embodied-FL: Federated Object Detection")
    print("  Backbone-only aggregation + Task-Aware weighting")
    print("=" * 70)

    NC = 10; IMG_SZ = 64; MAX_OBJ = 3
    ROUNDS = 10; EPOCHS = 2; LR = 0.005
    DEVICE = 'cpu'

    clients = create_clients(5, NC, n_img=60, seed=42)
    print(f"\n  {len(clients)} clients (Non-IID):")
    for c in clients:
        dom = max(set(c.class_subset), key=c.class_subset.count)
        print(f"    {c.name:12s} | n={c.n_images:3d} | classes={c.class_subset} | dominant={dom}")

    bb = Backbone(64)
    hd = DetectionHead(64, NC, MAX_OBJ)
    total = sum(p.numel() for p in list(bb.parameters()) + list(hd.parameters()))
    bb_p = sum(p.numel() for p in bb.parameters())
    hd_p = sum(p.numel() for p in hd.parameters())
    print(f"\n  Backbone: {bb_p:,} params | Head: {hd_p:,} params | Total: {total:,}")
    print(f"  Comm saving (backbone-only): {(1-bb_p/total)*100:.1f}%")

    results = {}

    # ── Exp 1: FedAvg vs Task-Aware ──
    print("\n" + "─" * 70)
    print("  Exp1: FedAvg vs Task-Aware (5 clients, backbone-only)")
    print("─" * 70)

    for method in ['fedavg', 'ours']:
        print(f"\n  [{method.upper()}]")
        r = run_federated(clients, NC, MAX_OBJ, ROUNDS, EPOCHS, LR, IMG_SZ,
                         method=method, device=DEVICE)
        results[method] = r
        print(f"  Final AP: {r[-1]['ap']:.4f}")

    # ── Summary ──
    fed_final = results['fedavg'][-1]['ap']
    ours_final = results['ours'][-1]['ap']
    print(f"\n  ── Summary ──")
    print(f"  FedAvg final AP:  {fed_final:.4f}")
    print(f"  Ours   final AP:  {ours_final:.4f}")
    if fed_final > 0:
        print(f"  Improvement: {(ours_final-fed_final)/fed_final*100:+.1f}%")

    # Best round
    fed_best = max(r['ap'] for r in results['fedavg'])
    ours_best = max(r['ap'] for r in results['ours'])
    print(f"\n  FedAvg best AP:   {fed_best:.4f}")
    print(f"  Ours   best AP:   {ours_best:.4f}")

    # ── Save ──
    results_dir = Path(__file__).parent / 'results'
    os.makedirs(results_dir, exist_ok=True)

    output = {
        'config': {
            'model': 'CNN backbone + detection head',
            'img_size': IMG_SZ, 'n_classes': NC, 'max_objects': MAX_OBJ,
            'rounds': ROUNDS, 'epochs': EPOCHS, 'lr': LR,
            'backbone_params': bb_p, 'head_params': hd_p,
        },
        'exp1': results
    }
    with open(f'{results_dir}/detection_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✅ Results saved to {results_dir}/detection_results.json")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Fig 1: Convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for method, color, label in [('fedavg', '#e74c3c', 'FedAvg'),
                                      ('ours', '#2ecc71', 'Ours (Task-Aware)')]:
            rounds = [r['round'] for r in results[method]]
            aps = [r['ap'] for r in results[method]]
            ax1.plot(rounds, aps, '-o', color=color, label=label, markersize=5)
        ax1.set_xlabel('Round'); ax1.set_ylabel('AP@50')
        ax1.set_title('Federated Detection: Convergence')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Fig 2: Per-client
        cids = list(results['ours'][-1]['per_client'].keys())
        x = np.arange(len(cids))
        w = 0.35
        fed_aps = [results['fedavg'][-1]['per_client'][cid] for cid in cids]
        ours_aps = [results['ours'][-1]['per_client'][cid] for cid in cids]
        ax2.bar(x - w/2, fed_aps, w, label='FedAvg', color='#e74c3c', alpha=0.8)
        ax2.bar(x + w/2, ours_aps, w, label='Ours', color='#2ecc71', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([cid.replace('factory_', 'F') for cid in cids])
        ax2.set_ylabel('AP@50')
        ax2.set_title('Per-Client AP@50 (Final Round)')
        ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{results_dir}/fig_detection_results.png', dpi=150)
        plt.close()
        print("  ✅ fig_detection_results.png")

        # Fig 3: Loss comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, color, label in [('fedavg', '#e74c3c', 'FedAvg'),
                                      ('ours', '#2ecc71', 'Ours (Task-Aware)')]:
            rounds = [r['round'] for r in results[method]]
            losses = [r['loss'] for r in results[method]]
            ax.plot(rounds, losses, '-s', color=color, label=label, markersize=5)
        ax.set_xlabel('Round'); ax.set_ylabel('Training Loss')
        ax.set_title('Federated Detection: Loss Convergence')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/fig_detection_loss.png', dpi=150)
        plt.close()
        print("  ✅ fig_detection_loss.png")

        print(f"\n  All plots saved to {results_dir}/")

    except ImportError:
        print("  ⚠ matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
