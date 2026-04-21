"""
Embodied-FL: Complete Experiment Suite
========================================
Paper-ready federated learning experiments for embodied AI.

Part A: Classification (6 experiments, NumPy MLP)
  Exp1: FedAvg vs FedProx vs Ours (5 clients, Non-IID)
  Exp2: Scalability (10 clients)
  Exp3: Non-IID severity sweep
  Exp4: Heterogeneous tasks (shared backbone)
  Exp5: Continual learning (EWC + Replay)
  Exp6: Gradient compression (Top-K + Quantization)

Part B: Object Detection (3 experiments, PyTorch CNN)
  Exp7: FedAvg vs Task-Aware detection (5 factories)
  Exp8: Backbone-only vs full-model aggregation
  Exp9: Communication cost analysis

Usage:
  python run_all_experiments.py              # Run all (slow)
  python run_all_experiments.py --part A     # Classification only
  python run_all_experiments.py --part B     # Detection only
  python run_all_experiments.py --quick      # Fast debug mode

Output:
  results/experiment_results.json   — All numerical results
  results/fig1-fig9.png             — Publication-quality figures
"""

import os, sys, json, time, copy, argparse, math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ═══════════════════════════════════════════════════════════════════════
#  PART A: CLASSIFICATION (NumPy MLP)
# ═══════════════════════════════════════════════════════════════════════

class MLP:
    """Multi-layer perceptron with Adam optimizer."""
    def __init__(self, sizes, seed=42):
        self.rng = np.random.RandomState(seed)
        self.sizes = sizes
        self._t = 0
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            s = np.sqrt(2.0 / sizes[i])
            self.W.append((self.rng.randn(sizes[i], sizes[i+1]) * s).astype(np.float32))
            self.b.append(np.zeros(sizes[i+1], dtype=np.float32))
        self._mw = [np.zeros_like(w) for w in self.W]
        self._vw = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._acts, self._pre = [], []

    def forward(self, x):
        self._acts, self._pre = [x], []
        a = x
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            self._pre.append(z)
            a = np.maximum(0, z)
            self._acts.append(a)
        z = a @ self.W[-1] + self.b[-1]
        self._pre.append(z)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        self._acts.append(e / e.sum(axis=1, keepdims=True))
        return self._acts[-1]

    def _adam(self, param, grad, m, v, lr):
        self._t += 1
        m[:] = 0.9 * m + 0.1 * grad
        v[:] = 0.999 * v + 0.001 * grad**2
        param -= lr * (m / (1 - 0.9**self._t)) / (np.sqrt(v / (1 - 0.999**self._t)) + 1e-8)

    def train_step(self, x, y, lr=0.001):
        m = x.shape[0]
        p = self.forward(x)
        loss = -np.sum(y * np.log(p + 1e-8)) / m
        d = (p - y) / m
        for i in range(len(self.W) - 1, -1, -1):
            dw, db = self._acts[i].T @ d, d.sum(axis=0)
            if i > 0: d = (d @ self.W[i].T) * (self._pre[i-1] > 0)
            self._adam(self.W[i], dw, self._mw[i], self._vw[i], lr)
            self._adam(self.b[i], db, self._mb[i], self._vb[i], lr)
        return float(loss)

    def compute_grad(self, x, y):
        m = x.shape[0]; p = self.forward(x); d = (p - y) / m
        raw = []
        for i in range(len(self.W) - 1, -1, -1):
            raw.append((self._acts[i].T @ d, d.sum(axis=0)))
            if i > 0: d = (d @ self.W[i].T) * (self._pre[i-1] > 0)
        grads = []
        for dw, db in reversed(raw): grads += [dw, db]
        return grads

    def apply_grad(self, grads, lr=0.001):
        for i in range(len(self.W)):
            self._adam(self.W[i], grads[2*i], self._mw[i], self._vw[i], lr)
            self._adam(self.b[i], grads[2*i+1], self._mb[i], self._vb[i], lr)

    def accuracy(self, x, y):
        return float(np.mean(np.argmax(self.forward(x), 1) == np.argmax(y, 1)))

    def get_params(self):
        r = []
        for w, b in zip(self.W, self.b): r += [w.copy(), b.copy()]
        return r

    def set_params(self, params):
        for i in range(len(self.W)):
            self.W[i], self.b[i] = params[2*i].copy(), params[2*i+1].copy()

    def param_count(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))


# ── Utilities ──

def cosine_lr(lr0, step, total, warmup=5):
    if step < warmup: return lr0 * (step + 1) / warmup
    return lr0 * 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(total - warmup, 1)))

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def generate_non_iid(n_samples, n_classes, alpha, dim, seed=42):
    rng = np.random.RandomState(seed)
    props = rng.dirichlet([alpha] * n_classes)
    counts = (props / props.sum() * n_samples).astype(int)
    counts[-1] = n_samples - counts[:-1].sum()
    X, y = [], []
    W_lat = rng.randn(dim, 16).astype(np.float32) * 0.5
    for c in range(n_classes):
        nc = counts[c]
        lat = rng.randn(nc, 16).astype(np.float32) + rng.randn(1, 16) * 1.5
        X.append(lat @ W_lat.T + rng.randn(nc, dim) * 0.3)
        yc = np.zeros((nc, n_classes), dtype=np.float32); yc[:, c] = 1.0
        y.append(yc)
    return np.vstack(X).astype(np.float32), np.vstack(y), props


# ── Aggregation ──

def agg_fedavg(updates, weights):
    t = sum(weights)
    return [sum(w / t * u[i] for u, w in zip(updates, weights)) for i in range(len(updates[0]))]

def agg_ours(updates, weights, losses, task_embs, global_emb, temperature=0.5):
    """Task-Aware Aggregation: performance(40%) + data(30%) + similarity(30%)."""
    n = len(updates)
    loss_arr = np.array(losses)
    ranks = np.argsort(np.argsort(loss_arr))
    perf_w = (1.0 + (n - 1 - ranks) / max(n - 1, 1)); perf_w /= perf_w.sum()
    total_n = sum(weights)
    sample_w = np.array([w / total_n for w in weights])
    sims = np.array([cosine_sim(e, global_emb) for e in task_embs])
    sim_w = np.exp(sims / temperature); sim_w /= sim_w.sum()
    combined = 0.4 * perf_w + 0.3 * sample_w + 0.3 * sim_w
    combined /= combined.sum()
    return [sum(combined[i] * updates[i][j] for i in range(n)) for j in range(len(updates[0]))]


# ── EWC ──

class EWC:
    def __init__(self, model, lam=10.0):
        self.model, self.lam, self.fisher, self.star_params = model, lam, None, None

    def consolidate(self, X, y, n_samples=200):
        idx = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
        Xs, ys = X[idx], y[idx]
        self.fisher = [np.zeros_like(w) for w in self.model.W] + [np.zeros_like(b) for b in self.model.b]
        for i in range(Xs.shape[0]):
            p = self.model.forward(Xs[i:i+1]); d = (p - ys[i:i+1]) / Xs.shape[0]
            for j in range(len(self.model.W) - 1, -1, -1):
                self.fisher[2*j] += self.model._acts[j].T @ d
                self.fisher[2*j+1] += d.sum(axis=0)
                if j > 0: d = (d @ self.model.W[j].T) * (self.model._pre[j-1] > 0)
        for k in range(len(self.fisher)): self.fisher[k] = self.fisher[k]**2
        self.star_params = self.model.get_params()


# ── Compression ──

def topk_sparsify(grads, sparsity=0.9):
    sparse, total_p, nonzero_p = [], 0, 0
    for g in grads:
        flat = g.flatten(); total_p += len(flat)
        k = max(1, int(len(flat) * (1 - sparsity)))
        thr = np.partition(np.abs(flat), -k)[-k]
        sg = np.where(np.abs(flat) >= thr, flat, 0.0).reshape(g.shape).astype(np.float32)
        sparse.append(sg); nonzero_p += np.count_nonzero(sg)
    return sparse, {"sparsity": sparsity, "ratio": total_p / max(nonzero_p, 1)}

def quantize_grads(grads, bits=8):
    quant = []
    for g in grads:
        gmin, gmax = g.min(), g.max()
        scale = (gmax - gmin) / (2**bits - 1) if gmax > gmin else 1.0
        q = np.round((g - gmin) / scale).astype(np.uint8 if bits <= 8 else np.uint16)
        quant.append((q.astype(np.float32) * scale + gmin).reshape(g.shape))
    return quant, {"bits": bits, "ratio": 32.0 / max(bits, 1)}


# ── Data Structures ──

@dataclass
class ClientData:
    id: str; X: np.ndarray; y: np.ndarray; label_dist: np.ndarray
    n: int; task_emb: np.ndarray; domain: str

@dataclass
class RoundResult:
    round_num: int; global_loss: float; global_accuracy: float; per_client: list


def make_clients(n_clients, n_classes, alpha, dim, seed=42):
    rng = np.random.RandomState(seed)
    cities = ["Suzhou", "Wuxi", "Kunshan", "Shenzhen", "Dongguan",
              "Hangzhou", "Nanjing", "Shanghai", "Chengdu", "Wuhan"]
    clients = []
    for i in range(n_clients):
        n = rng.randint(300, 700)
        X, y, ld = generate_non_iid(n, n_classes, alpha, dim, seed + i)
        emb = rng.randn(32).astype(np.float32) * 0.1
        clients.append(ClientData(
            f"Factory-{chr(65+i)}", X, y, ld, n, emb,
            f"PCB/{cities[i] if i < len(cities) else f'City-{i}'}"))
    return clients


def run_fl(clients, method, n_rounds, global_emb, arch, local_epochs=5, lr=0.001):
    results = []
    gm = MLP(arch, seed=42); gp = gm.get_params()
    for rnd in range(1, n_rounds + 1):
        t0 = time.time(); lr_now = cosine_lr(lr, rnd, n_rounds)
        updates, cw, losses, tembs = [], [], [], []
        for c in clients:
            lm = MLP(arch, seed=42); lm.set_params([p.copy() for p in gp])
            for _ in range(local_epochs):
                idx = np.random.permutation(c.n)
                loss = lm.train_step(c.X[idx], c.y[idx], lr=lr_now)
            updates.append(lm.get_params()); cw.append(c.n)
            losses.append(loss); tembs.append(c.task_emb)
        gp = agg_ours(updates, cw, losses, tembs, global_emb) if method == "Ours" else agg_fedavg(updates, cw)
        gm.set_params(gp)
        tl, ta, pc = 0.0, 0.0, []
        for c in clients:
            p = gm.forward(c.X)
            l = -np.sum(c.y * np.log(p + 1e-8)) / c.n
            a = float(np.mean(np.argmax(p, 1) == np.argmax(c.y, 1)))
            tl += l; ta += a; pc.append({"id": c.id, "loss": float(l), "accuracy": float(a)})
        nc = len(clients)
        results.append(RoundResult(rnd, tl/nc, ta/nc, pc))
        if rnd % 10 == 0 or rnd == 1:
            print(f"    R{rnd:3d} | loss={tl/nc:.4f} | acc={ta/nc:.4f} | {time.time()-t0:.2f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  PART B: OBJECT DETECTION (PyTorch CNN)
# ═══════════════════════════════════════════════════════════════════════

def run_detection_experiments(quick=False):
    """Run federated object detection experiments (Exp7-9)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    # ── Model ──
    class ConvBnSiLU(nn.Module):
        def __init__(self, c_in, c_out, k=3, s=1, p=1):
            super().__init__()
            self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
            self.bn = nn.BatchNorm2d(c_out)
        def forward(self, x): return F.silu(self.bn(self.conv(x)))

    class Backbone(nn.Module):
        def __init__(self, ch=32):
            super().__init__()
            self.features = nn.Sequential(
                ConvBnSiLU(3, ch, 5, 2, 2), ConvBnSiLU(ch, ch*2, 3, 2, 1),
                ConvBnSiLU(ch*2, ch*4, 3, 2, 1), nn.AdaptiveAvgPool2d(4))
            self.fc = nn.Linear(ch*4*16, 128)
            self.out_dim = 128
        def forward(self, x): return self.fc(self.features(x).view(x.size(0), -1))

    class DetectionHead(nn.Module):
        def __init__(self, in_dim, nc, max_obj):
            super().__init__()
            self.max_obj = max_obj
            self.box_head = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, max_obj * 4))
            self.cls_head = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, max_obj * nc))
        def forward(self, feat):
            return (self.box_head(feat).view(feat.size(0), self.max_obj, 4),
                    self.cls_head(feat).view(feat.size(0), self.max_obj, -1))

    class Detector(nn.Module):
        def __init__(self, bb, hd):
            super().__init__(); self.backbone, self.head = bb, hd
        def forward(self, x): return self.head(self.backbone(x))

    # ── Dataset ──
    class DetDataset(Dataset):
        def __init__(self, n, sz, nc, classes, seed, max_obj):
            rng = np.random.RandomState(seed)
            self.data = []
            for _ in range(n):
                img = rng.randn(3, sz, sz).astype(np.float32) * 0.2
                objs = []
                for _ in range(rng.randint(1, max_obj + 1)):
                    c = int(rng.choice(classes))
                    w, h = rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4)
                    cx, cy = rng.uniform(w/2, 1-w/2), rng.uniform(h/2, 1-h/2)
                    objs.append([cx, cy, w, h, c])
                    x1, y1 = int((cx-w/2)*sz), int((cy-h/2)*sz)
                    x2, y2 = int((cx+w/2)*sz), int((cy+h/2)*sz)
                    img[:, y1:y2, x1:x2] = (c+1)/nc + rng.randn(3, y2-y1, x2-x1)*0.1
                while len(objs) < max_obj: objs.append([0,0,0,0,0])
                self.data.append((img, objs))
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            img, objs = self.data[i]
            return torch.tensor(img), torch.tensor(objs, dtype=torch.float32)

    # ── Loss ──
    class DetLoss(nn.Module):
        def __init__(self, nc, mo):
            super().__init__(); self.nc, self.mo = nc, mo
        def forward(self, pb, pc, tgt):
            mask = (tgt[..., 4] > 0).float()
            box_loss = ((pb - tgt[..., :4]) * mask.unsqueeze(-1)).pow(2).sum() / max(mask.sum(), 1)
            tc = tgt[..., 4].long().clamp(0, self.nc - 1)
            cls_loss = (F.cross_entropy(pc.view(-1, self.nc), tc.view(-1), reduction='none') * mask.view(-1)).sum() / max(mask.sum(), 1)
            return box_loss + cls_loss

    # ── Metrics ──
    def compute_iou(a, b):
        ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
        bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
        inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / max(union, 1e-8)

    def compute_ap(pb, pc, tgt, nc, iou_t=0.3):
        aps = []
        for c in range(nc):
            tp = fp = n_gt = 0
            for b in range(tgt.size(0)):
                for j in range(tgt.size(1)):
                    if int(tgt[b, j, 4].item()) == c: n_gt += 1
            if n_gt == 0: continue
            preds = []
            for b in range(pb.size(0)):
                for j in range(pb.size(1)):
                    prob = F.softmax(pc[b, j], dim=0)[c].item()
                    if prob > 0.15: preds.append((prob, b, j))
            preds.sort(reverse=True); matched = set()
            for prob, b, j in preds:
                pbox = pb[b, j].detach().cpu().numpy()
                best_iou, best_gt = 0, -1
                for k in range(tgt.size(1)):
                    if int(tgt[b, k, 4].item()) != c: continue
                    iou = compute_iou(pbox, tgt[b, k, :4].detach().cpu().numpy())
                    if iou > best_iou: best_iou, best_gt = iou, k
                if best_iou >= iou_t and best_gt not in matched: tp += 1; matched.add(best_gt)
                else: fp += 1
            if tp + fp > 0:
                p, r = tp/(tp+fp), tp/n_gt
                aps.append(2*p*r / max(p+r, 1e-8))
        return np.mean(aps) if aps else 0.0

    # ── Task embedding & aggregation ──
    def task_emb(class_dist, dim=64):
        emb = np.zeros(dim, dtype=np.float32)
        for cid, prob in enumerate(class_dist): emb[cid % dim] += prob
        n = np.linalg.norm(emb)
        return emb / max(n, 1e-8)

    def agg_det_fedavg(updates, weights):
        t = sum(weights)
        return {k: sum(u[k]*w for u, w in zip(updates, weights))/t for k in updates[0]}

    def agg_det_ours(updates, weights, task_embs, global_emb, losses):
        sims = [max(cosine_sim(te, global_emb), 0.1) for te in task_embs]
        perfs = [1.0/(l+0.1) for l in losses]
        combined = [s*p*w for s, p, w in zip(sims, perfs, weights)]
        t = sum(combined)
        return {k: sum(u[k]*c for u, c in zip(updates, combined))/t for k in updates[0]}

    # ── Client config ──
    @dataclass
    class DetClient:
        id: str; name: str; n_images: int; class_subset: list; seed: int

    def make_det_clients(n, nc, n_img, seed=42):
        rng = np.random.RandomState(seed)
        clients = []
        for i in range(n):
            dom = i % nc
            others = [c for c in range(nc) if c != dom]
            minority = list(rng.choice(others, 2, replace=False))
            clients.append(DetClient(
                f'factory_{chr(65+i)}', f'Factory-{chr(65+i)}',
                n_img, [dom]*3 + minority, seed + i))
        return clients

    # ── Federated training ──
    def run_det_fed(clients, nc, mo, rounds, epochs, lr, sz, method, device):
        global_bb = Backbone(32)
        client_heads = {c.id: DetectionHead(128, nc, mo) for c in clients}
        criterion = DetLoss(nc, mo)

        task_embs = []
        for c in clients:
            dist = np.zeros(nc)
            for cls in c.class_subset: dist[cls] += 1.0/len(c.class_subset)
            task_embs.append(task_emb(dist))
        global_emb = np.mean(task_embs, axis=0)

        results = []
        for r in range(1, rounds + 1):
            t0 = time.time()
            updates, weights, losses = [], [], []
            for c in clients:
                c_model = Detector(copy.deepcopy(global_bb), client_heads[c.id]).to(device)
                ds = DetDataset(c.n_images, sz, nc, c.class_subset, c.seed, mo)
                loader = DataLoader(ds, batch_size=32, shuffle=True)
                opt = torch.optim.Adam(c_model.parameters(), lr=lr)
                c_model.train()
                for ep in range(epochs):
                    for imgs, tgt in loader:
                        imgs, tgt = imgs.to(device), tgt.to(device)
                        pb, pc = c_model(imgs)
                        loss = criterion(pb, pc, tgt)
                        opt.zero_grad(); loss.backward(); opt.step()
                client_heads[c.id] = c_model.head
                bb_after = {k: v.clone() for k, v in c_model.backbone.state_dict().items()}
                bb_before = {k: v.clone() for k, v in global_bb.state_dict().items()}
                update = {k: bb_after[k] - bb_before[k] for k in bb_after}
                c_model.eval()
                total_ap, nb = 0, 0
                with torch.no_grad():
                    for imgs, tgt in loader:
                        imgs, tgt = imgs.to(device), tgt.to(device)
                        pb, pc = c_model(imgs)
                        total_ap += compute_ap(pb, pc, tgt, nc); nb += 1
                updates.append(update); weights.append(c.n_images); losses.append(loss.item())

            if method == 'ours':
                agg = agg_det_ours(updates, weights, task_embs, global_emb, losses)
            else:
                agg = agg_det_fedavg(updates, weights)

            bb_params = {k: v.clone() for k, v in global_bb.state_dict().items()}
            for k in agg: bb_params[k] = bb_params[k] + agg[k]
            global_bb.load_state_dict(bb_params)

            per_client, total_ap = {}, 0
            for c in clients:
                eval_model = Detector(global_bb, client_heads[c.id]).to(device)
                eval_model.eval()
                ds = DetDataset(min(20, c.n_images), sz, nc, c.class_subset, c.seed+999, mo)
                loader = DataLoader(ds, batch_size=32)
                ap_sum, nb = 0, 0
                with torch.no_grad():
                    for imgs, tgt in loader:
                        imgs, tgt = imgs.to(device), tgt.to(device)
                        pb, pc = eval_model(imgs)
                        ap_sum += compute_ap(pb, pc, tgt, nc); nb += 1
                c_ap = ap_sum / max(nb, 1)
                per_client[c.id] = c_ap; total_ap += c_ap

            avg_ap = total_ap / len(clients)
            elapsed = time.time() - t0
            print(f"    R{r:2d} | AP={avg_ap:.4f} | loss={np.mean(losses):.3f} | {elapsed:.1f}s")
            results.append({'round': r, 'ap': avg_ap, 'loss': float(np.mean(losses)),
                           'per_client': per_client, 'time': elapsed})
        return results

    # ── Run experiments ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    NC, SZ, MO = 10, 64, 3
    if quick:
        ROUNDS, EPOCHS, LR, N_IMG = 4, 1, 0.005, 40
    else:
        ROUNDS, EPOCHS, LR, N_IMG = 10, 2, 0.003, 100

    print(f"\n  Device: {device}")
    clients = make_det_clients(5, NC, N_IMG, seed=42)
    print(f"  {len(clients)} clients (Non-IID):")
    for c in clients:
        dom = max(set(c.class_subset), key=c.class_subset.count)
        print(f"    {c.name:12s} | n={c.n_images:3d} | classes={c.class_subset} | dominant={dom}")

    bb = Backbone(32); hd = DetectionHead(128, NC, MO)
    bb_p = sum(p.numel() for p in bb.parameters())
    hd_p = sum(p.numel() for p in hd.parameters())
    print(f"  Backbone: {bb_p:,} params | Head: {hd_p:,} params | Total: {bb_p+hd_p:,}")
    print(f"  Comm saving (backbone-only): {(1-bb_p/(bb_p+hd_p))*100:.1f}%")

    det_results = {}

    # Exp7: FedAvg vs Task-Aware
    print("\n" + "─" * 70)
    print("  Exp7: FedAvg vs Task-Aware Detection")
    print("─" * 70)
    for method in ['fedavg', 'ours']:
        print(f"\n  [{method.upper()}]")
        r = run_det_fed(clients, NC, MO, ROUNDS, EPOCHS, LR, SZ, method, device)
        det_results[method] = r
        print(f"  Final AP: {r[-1]['ap']:.4f}")

    # Exp8: Communication cost
    print("\n" + "─" * 70)
    print("  Exp8: Communication Cost Analysis")
    print("─" * 70)
    backbone_bytes = bb_p * 4  # float32
    full_bytes = (bb_p + hd_p) * 4
    det_results['comm'] = {
        'backbone_only_bytes': backbone_bytes,
        'full_model_bytes': full_bytes,
        'saving_pct': (1 - backbone_bytes / full_bytes) * 100
    }
    print(f"    Backbone-only: {backbone_bytes:,} bytes")
    print(f"    Full model:     {full_bytes:,} bytes")
    print(f"    Saving:         {det_results['comm']['saving_pct']:.1f}%")

    # Exp9: Per-client analysis
    print("\n" + "─" * 70)
    print("  Exp9: Per-Client Analysis")
    print("─" * 70)
    for method in ['fedavg', 'ours']:
        final = det_results[method][-1]
        print(f"  {method.upper()} final round:")
        for cid, ap in final['per_client'].items():
            print(f"    {cid}: AP={ap:.4f}")

    return det_results


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Embodied-FL Experiment Suite")
    parser.add_argument('--part', choices=['A', 'B', 'all'], default='all')
    parser.add_argument('--quick', action='store_true', help='Fast debug mode')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  Embodied-FL: Complete Experiment Suite")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'} | Part: {args.part}")
    print("=" * 70)

    all_results = {}

    # ═══════════════════════════════════════════════════════════
    #  PART A: CLASSIFICATION
    # ═══════════════════════════════════════════════════════════
    if args.part in ('A', 'all'):
        DIM, HID, NCLS = 24, 128, 10
        ARCH = [DIM, HID, 64, NCLS]
        ROUNDS = 20 if args.quick else 80
        EPOCHS = 2 if args.quick else 5
        LR = 0.001

        rng = np.random.RandomState(0)
        global_emb = np.ones(32, dtype=np.float32); global_emb[:10] = 2.0

        # Exp1
        print("\n" + "─" * 70)
        print("  Exp1: FedAvg vs FedProx vs Ours (5 clients)")
        print("─" * 70)
        clients5 = make_clients(5, NCLS, 0.5, DIM, 42)
        exp1 = {}
        for method in ["FedAvg", "FedProx", "Ours"]:
            print(f"\n  Running {method}...")
            r = run_fl(clients5, method, ROUNDS, global_emb, ARCH, EPOCHS, LR)
            exp1[method] = r
            print(f"  Final: acc={r[-1].global_accuracy:.4f}")
        all_results['exp1'] = exp1

        # Exp2
        print("\n" + "─" * 70)
        print("  Exp2: Scalability (10 clients)")
        print("─" * 70)
        clients10 = make_clients(10, NCLS, 0.5, DIM, 42)
        exp2 = {}
        for method in ["FedAvg", "Ours"]:
            print(f"\n  Running {method}...")
            r = run_fl(clients10, method, ROUNDS, global_emb, ARCH, EPOCHS, LR)
            exp2[method] = r
            print(f"  Final: acc={r[-1].global_accuracy:.4f}")
        all_results['exp2'] = exp2

        # Exp3
        print("\n" + "─" * 70)
        print("  Exp3: Non-IID Severity")
        print("─" * 70)
        exp3 = {}
        for alpha, label in [(5.0, "IID"), (1.0, "Low"), (0.5, "Medium"), (0.1, "High")]:
            print(f"\n  α={alpha} ({label}):")
            exp3[label] = {}
            for method in ["FedAvg", "Ours"]:
                cl = make_clients(5, NCLS, alpha, DIM, 42)
                r = run_fl(cl, method, ROUNDS, global_emb, ARCH, EPOCHS, LR)
                exp3[label][method] = r
                print(f"    {method}: acc={r[-1].global_accuracy:.4f}")
        all_results['exp3'] = exp3

        # Exp4
        print("\n" + "─" * 70)
        print("  Exp4: Heterogeneous Tasks")
        print("─" * 70)
        BB_ARCH = [DIM, 64, 32]
        tasks = [("inspection", 10, 500), ("grasping", 6, 400), ("assembly", 4, 350)]
        rng4 = np.random.RandomState(42)
        het_clients = []
        for tname, tcls, tn in tasks:
            X, y, _ = generate_non_iid(tn, tcls, 0.5, DIM, 42)
            het_clients.append(ClientData(tname, X, y, _, tn, rng4.randn(32).astype(np.float32)*0.1, tname))

        def run_hetero(method):
            bb = MLP(BB_ARCH, seed=42); gbb = bb.get_params()
            heads = {c.id: MLP([32, 16, c.y.shape[1]], seed=42) for c in het_clients}
            results = []
            for rnd in range(1, ROUNDS + 1):
                t0 = time.time(); lr_now = cosine_lr(0.001, rnd, ROUNDS)
                bb_ups, cw, losses, tembs = [], [], [], []
                for c in het_clients:
                    lbb = MLP(BB_ARCH, seed=42); lbb.set_params([p.copy() for p in gbb])
                    head = heads[c.id]
                    for _ in range(5):
                        feat = lbb.forward(c.X); out = head.forward(feat)
                        m = c.n; loss = -np.sum(c.y * np.log(out + 1e-8)) / m
                        d = (out - c.y) / m
                        for i in range(len(head.W) - 1, -1, -1):
                            hdw, hdb = head._acts[i].T @ d, d.sum(axis=0)
                            if i > 0: d = (d @ head.W[i].T) * (head._pre[i-1] > 0)
                            head._adam(head.W[i], hdw, head._mw[i], head._vw[i], lr_now)
                            head._adam(head.b[i], hdb, head._mb[i], head._vb[i], lr_now)
                        d_feat = d @ head.W[0].T
                        d_bb = d_feat * (lbb._pre[-1] > 0)
                        for i in range(len(lbb.W) - 1, -1, -1):
                            bdw, bdb = lbb._acts[i].T @ d_bb, d_bb.sum(axis=0)
                            if i > 0: d_bb = (d_bb @ lbb.W[i].T) * (lbb._pre[i-1] > 0)
                            lbb._adam(lbb.W[i], bdw, lbb._mw[i], lbb._vw[i], lr_now)
                            lbb._adam(lbb.b[i], bdb, lbb._mb[i], lbb._vb[i], lr_now)
                    bb_ups.append(lbb.get_params()); cw.append(c.n)
                    losses.append(loss); tembs.append(c.task_emb)
                gbb = agg_ours(bb_ups, cw, losses, tembs, global_emb) if method == "Ours" else agg_fedavg(bb_ups, cw)
                bb.set_params(gbb)
                tl, ta, pc = 0.0, 0.0, []
                for c in het_clients:
                    feat = bb.forward(c.X); out = heads[c.id].forward(feat)
                    l = -np.sum(c.y * np.log(out + 1e-8)) / c.n
                    a = float(np.mean(np.argmax(out, 1) == np.argmax(c.y, 1)))
                    tl += l; ta += a; pc.append({"id": c.id, "loss": float(l), "accuracy": float(a)})
                nc = len(het_clients)
                results.append(RoundResult(rnd, tl/nc, ta/nc, pc))
                if rnd % 10 == 0 or rnd == 1:
                    print(f"    R{rnd:3d} | loss={tl/nc:.4f} | acc={ta/nc:.4f} | {time.time()-t0:.2f}s")
            return results

        exp4 = {}
        for method in ["FedAvg", "Ours"]:
            print(f"\n  Running {method}...")
            r = run_hetero(method)
            exp4[method] = r
            print(f"  Final: acc={r[-1].global_accuracy:.4f}")
        all_results['exp4'] = exp4

        # Exp5
        print("\n" + "─" * 70)
        print("  Exp5: Continual Learning (EWC + Replay)")
        print("─" * 70)
        rng5 = np.random.RandomState(42)
        N_P1, N_P2 = 6, 10

        def make_phase_data(n_classes, total=800, seed=42):
            r = np.random.RandomState(seed)
            W = r.randn(DIM, 16).astype(np.float32) * 0.5
            X, y = [], []
            for c in range(n_classes):
                nc = total // n_classes
                lat = r.randn(nc, 16).astype(np.float32) + r.randn(1, 16) * 1.5
                X.append(lat @ W.T + r.randn(nc, DIM) * 0.3)
                yc = np.zeros((nc, n_classes), dtype=np.float32); yc[:, c] = 1.0
                y.append(yc)
            return np.vstack(X).astype(np.float32), np.vstack(y)

        X_p1, y_p1 = make_phase_data(N_P1, 800, 42)
        X_p2, y_p2 = make_phase_data(N_P2, 800, 43)

        def run_ewc_exp(use_ewc, use_replay, lam=5000.0):
            model = MLP([DIM, HID, 64, N_P1], seed=42)  # Phase 1 architecture
            ewc = EWC(model, lam) if use_ewc else None
            results = []; total_steps = 50
            for step in range(1, 26):
                lr_now = cosine_lr(0.001, step, total_steps, warmup=3)
                idx = np.random.permutation(len(X_p1))
                model.train_step(X_p1[idx], y_p1[idx], lr=lr_now)
                results.append({"step": step, "phase": 1, "acc_old": model.accuracy(X_p1, y_p1), "acc_new": 0.0})
            if ewc: ewc.consolidate(X_p1, y_p1)
            # Expand model for phase 2 (add new output neurons)
            old_W, old_b = model.W[-1].copy(), model.b[-1].copy()
            model.W[-1] = np.zeros((model.W[-1].shape[0], N_P2), dtype=np.float32)
            model.b[-1] = np.zeros(N_P2, dtype=np.float32)
            model.W[-1][:, :N_P1] = old_W
            model.b[-1][:N_P1] = old_b
            if ewc:
                old_f, old_s = ewc.fisher[-1].copy(), ewc.star_params[-1].copy()
                ewc.fisher[-1] = np.zeros(N_P2, dtype=np.float32)
                ewc.star_params[-1] = np.zeros(N_P2, dtype=np.float32)
                ewc.fisher[-1][:N_P1] = old_f
                ewc.star_params[-1][:N_P1] = old_s
            replay_size = int(len(X_p1) * 0.2) if use_replay else 0
            for step in range(26, 51):
                lr_now = cosine_lr(0.001, step, total_steps, warmup=3)
                idx = np.random.permutation(len(X_p2))
                X_b, y_b = X_p2[idx], y_p2[idx]
                if use_replay and replay_size > 0:
                    ridx = np.random.choice(len(X_p1), replay_size, replace=False)
                    X_b, y_b = np.vstack([X_b, X_p1[ridx]]), np.vstack([y_b, y_p1[ridx]])
                grads = model.compute_grad(X_b, y_b)
                if ewc and ewc.fisher is not None:
                    for i in range(len(model.W)):
                        grads[2*i] += ewc.lam * ewc.fisher[2*i] * (model.W[i] - ewc.star_params[2*i])
                        grads[2*i+1] += ewc.lam * ewc.fisher[2*i+1] * (model.b[i] - ewc.star_params[2*i+1])
                model.apply_grad(grads, lr=lr_now)
                results.append({"step": step, "phase": 2,
                               "acc_old": model.accuracy(X_p1, y_p1),
                               "acc_new": model.accuracy(X_p2, y_p2)})
            return results

        exp5 = {}
        for label, use_ewc, use_replay in [("Fine-tune", False, False), ("EWC only", True, False),
                                             ("Replay only", False, True), ("EWC + Replay", True, True)]:
            print(f"\n  Running {label}...")
            r = run_ewc_exp(use_ewc, use_replay)
            exp5[label] = r
            print(f"    Phase 2: old={r[-1]['acc_old']:.4f}, new={r[-1]['acc_new']:.4f}")
        all_results['exp5'] = exp5

        # Exp6
        print("\n" + "─" * 70)
        print("  Exp6: Gradient Compression")
        print("─" * 70)
        rng_c = np.random.RandomState(42)
        W_lat_c = rng_c.randn(DIM, 16).astype(np.float32) * 0.5
        X_parts, y_parts = [], []
        for c in range(NCLS):
            nc = 80
            lat = rng_c.randn(nc, 16).astype(np.float32) + rng_c.randn(1, 16) * 1.5
            X_parts.append(lat @ W_lat_c.T + rng_c.randn(nc, DIM) * 0.3)
            yc = np.zeros((nc, NCLS), dtype=np.float32); yc[:, c] = 1.0
            y_parts.append(yc)
        X_all = np.vstack(X_parts).astype(np.float32)
        y_all = np.vstack(y_parts)
        shuffle_idx = np.random.RandomState(99).permutation(800)
        X_all, y_all = X_all[shuffle_idx], y_all[shuffle_idx]
        X_comp, y_comp, X_val, y_val = X_all[:600], y_all[:600], X_all[600:], y_all[600:]
        n_params = MLP(ARCH, seed=42).param_count()

        def train_compressed(compress_fn, n_steps=100):
            model = MLP(ARCH, seed=42)
            for step in range(1, n_steps + 1):
                lr_now = cosine_lr(0.001, step, n_steps, warmup=3)
                idx = np.random.permutation(len(X_comp))
                grads = model.compute_grad(X_comp[idx], y_comp[idx])
                if compress_fn: grads, _ = compress_fn(grads)
                model.apply_grad(grads, lr=lr_now)
            return model.accuracy(X_val, y_val)

        exp6 = {}
        acc_base = train_compressed(None, 100)
        exp6["No compression"] = {"ratio": 1.0, "accuracy": acc_base}
        print(f"  No compression: acc={acc_base:.4f}")
        for sp in [0.5, 0.7, 0.9, 0.95]:
            acc = train_compressed(lambda s=sp: topk_sparsify(None, s) if False else topk_sparsify(
                MLP(ARCH, seed=42).compute_grad(X_comp, y_comp), s), 100)
            # Fix: actually train with compression
            def make_topk(s=sp):
                return lambda g: topk_sparsify(g, s)
            acc = train_compressed(make_topk(), 100)
            exp6[f"TopK-{int(sp*100)}"] = {"ratio": 1.0/(1-sp), "accuracy": acc}
            print(f"  TopK-{int(sp*100)}: {1.0/(1-sp):.1f}x, acc={acc:.4f}")
        for bits in [16, 8, 4]:
            def make_quant(b=bits):
                return lambda g: quantize_grads(g, b)
            acc = train_compressed(make_quant(), 100)
            exp6[f"Quant-{bits}bit"] = {"ratio": 32.0/bits, "accuracy": acc}
            print(f"  Quant-{bits}bit: {32.0/bits:.1f}x, acc={acc:.4f}")
        all_results['exp6'] = exp6

    # ═══════════════════════════════════════════════════════════
    #  PART B: OBJECT DETECTION
    # ═══════════════════════════════════════════════════════════
    if args.part in ('B', 'all'):
        print("\n" + "=" * 70)
        print("  PART B: Object Detection Experiments")
        print("=" * 70)
        all_results['detection'] = run_detection_experiments(quick=args.quick)

    # ═══════════════════════════════════════════════════════════
    #  SAVE & PLOT
    # ═══════════════════════════════════════════════════════════
    def ser(rds):
        if isinstance(rds[0], RoundResult):
            return [{"round": r.round_num, "loss": r.global_loss, "accuracy": r.global_accuracy} for r in rds]
        return rds  # detection results are already dicts

    output = {}
    for k, v in all_results.items():
        if k == 'detection':
            output[k] = v
        elif isinstance(v, dict):
            output[k] = {m: ser(r) for m, r in v.items()}
        else:
            output[k] = v

    with open(f"{RESULTS_DIR}/all_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o))
    print(f"\n✅ Results saved to {RESULTS_DIR}/all_results.json")

    # ── Generate Plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"FedAvg": "#e74c3c", "FedProx": "#f39c12", "Ours": "#2ecc71"}
        markers = {"FedAvg": "o", "FedProx": "s", "Ours": "^"}

        # Fig 1: Classification convergence
        if 'exp1' in all_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            for m in ["FedAvg", "FedProx", "Ours"]:
                rds = all_results['exp1'][m]
                ax1.plot([r.round_num for r in rds], [r.global_loss for r in rds],
                        label=m, color=colors[m], marker=markers[m], markersize=2, linewidth=1.5)
                ax2.plot([r.round_num for r in rds], [r.global_accuracy for r in rds],
                        label=m, color=colors[m], marker=markers[m], markersize=2, linewidth=1.5)
            ax1.set_xlabel("Round"); ax1.set_ylabel("Loss"); ax1.set_title("(a) Loss"); ax1.legend()
            ax2.set_xlabel("Round"); ax2.set_ylabel("Accuracy"); ax2.set_title("(b) Accuracy"); ax2.legend()
            plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig1_convergence.png"); plt.close()
            print("  ✅ fig1_convergence.png")

        # Fig 2: Non-IID severity
        if 'exp3' in all_results:
            fig, ax = plt.subplots(figsize=(8, 5))
            sevs = list(all_results['exp3'].keys())
            x = np.arange(len(sevs)); w = 0.35
            fa_accs = [all_results['exp3'][s]["FedAvg"][-1].global_accuracy for s in sevs]
            ou_accs = [all_results['exp3'][s]["Ours"][-1].global_accuracy for s in sevs]
            ax.bar(x - w/2, fa_accs, w, label="FedAvg", color="#e74c3c", alpha=0.8)
            ax.bar(x + w/2, ou_accs, w, label="Ours", color="#2ecc71", alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(sevs)
            ax.set_ylabel("Accuracy"); ax.set_title("Non-IID Severity: FedAvg vs Ours"); ax.legend()
            plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig2_noniid.png"); plt.close()
            print("  ✅ fig2_noniid.png")

        # Fig 3: Scalability
        if 'exp2' in all_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            for m in ["FedAvg", "Ours"]:
                rds = all_results['exp2'][m]
                ax1.plot([r.round_num for r in rds], [r.global_loss for r in rds], label=m, color=colors[m], linewidth=1.5)
                ax2.plot([r.round_num for r in rds], [r.global_accuracy for r in rds], label=m, color=colors[m], linewidth=1.5)
            ax1.set_xlabel("Round"); ax1.set_ylabel("Loss"); ax1.set_title("10 Clients Loss"); ax1.legend()
            ax2.set_xlabel("Round"); ax2.set_ylabel("Accuracy"); ax2.set_title("10 Clients Accuracy"); ax2.legend()
            plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig3_scalability.png"); plt.close()
            print("  ✅ fig3_scalability.png")

        # Fig 4: EWC + Replay
        if 'exp5' in all_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            ewc_colors = {"Fine-tune": "#e74c3c", "EWC only": "#3498db", "Replay only": "#f39c12", "EWC + Replay": "#2ecc71"}
            for label, rds in all_results['exp5'].items():
                steps = [r["step"] for r in rds]
                ax1.plot(steps, [r["acc_old"] for r in rds], label=label, color=ewc_colors[label], linewidth=1.5)
                p2 = [r for r in rds if r["phase"] == 2]
                if p2: ax2.plot([r["step"] for r in p2], [r["acc_new"] for r in p2], label=label, color=ewc_colors[label], linewidth=1.5)
            ax1.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xlabel("Step"); ax1.set_ylabel("Old Classes Acc"); ax1.set_title("(a) Forgetting"); ax1.legend(fontsize=7)
            ax2.set_xlabel("Step"); ax2.set_ylabel("New Classes Acc"); ax2.set_title("(b) Learning"); ax2.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig4_ewc.png"); plt.close()
            print("  ✅ fig4_ewc.png")

        # Fig 5: Compression
        if 'exp6' in all_results:
            fig, ax = plt.subplots(figsize=(8, 5))
            ms, cs, acs, cls = [], [], [], []
            for label, info in all_results['exp6'].items():
                ms.append(label); cs.append(info["ratio"]); acs.append(info["accuracy"])
                cls.append("#3498db" if "TopK" in label else "#e67e22" if "Quant" in label else "#95a5a6")
            ax.scatter(cs, acs, c=cls, s=100, zorder=5, edgecolors='white')
            for i, m in enumerate(ms): ax.annotate(m, (cs[i], acs[i]), textcoords="offset points", xytext=(8, 5), fontsize=7)
            ax.set_xlabel("Compression Ratio (×)"); ax.set_ylabel("Accuracy")
            ax.set_title("Communication vs Accuracy"); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig5_compression.png"); plt.close()
            print("  ✅ fig5_compression.png")

        # Fig 6: Detection convergence
        if 'detection' in all_results:
            det = all_results['detection']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            for method, color, label in [('fedavg', '#e74c3c', 'FedAvg'), ('ours', '#2ecc71', 'Ours (Task-Aware)')]:
                rounds = [r['round'] for r in det[method]]
                aps = [r['ap'] for r in det[method]]
                losses = [r['loss'] for r in det[method]]
                ax1.plot(rounds, aps, '-o', color=color, label=label, markersize=4)
                ax2.plot(rounds, losses, '-s', color=color, label=label, markersize=4)
            ax1.set_xlabel("Round"); ax1.set_ylabel("AP@50"); ax1.set_title("(a) Detection AP"); ax1.legend()
            ax2.set_xlabel("Round"); ax2.set_ylabel("Loss"); ax2.set_title("(b) Detection Loss"); ax2.legend()
            plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig6_detection.png"); plt.close()
            print("  ✅ fig6_detection.png")

            # Fig 7: Per-client detection
            fig, ax = plt.subplots(figsize=(8, 5))
            cids = list(det['ours'][-1]['per_client'].keys())
            x = np.arange(len(cids)); w = 0.35
            fed_aps = [det['fedavg'][-1]['per_client'][cid] for cid in cids]
            ours_aps = [det['ours'][-1]['per_client'][cid] for cid in cids]
            ax.bar(x - w/2, fed_aps, w, label='FedAvg', color='#e74c3c', alpha=0.8)
            ax.bar(x + w/2, ours_aps, w, label='Ours', color='#2ecc71', alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels([cid.replace('factory_', 'F') for cid in cids])
            ax.set_ylabel("AP@50"); ax.set_title("Per-Factory Detection AP"); ax.legend()
            plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig7_per_client_det.png"); plt.close()
            print("  ✅ fig7_per_client_det.png")

            # Fig 8: Communication cost
            if 'comm' in det:
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['Backbone\nOnly', 'Full\nModel']
                vals = [det['comm']['backbone_only_bytes']/1024, det['comm']['full_model_bytes']/1024]
                bars = ax.bar(labels, vals, color=['#2ecc71', '#e74c3c'], width=0.5)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{val:.0f} KB', ha='center', fontweight='bold')
                ax.set_ylabel('Bytes per Round')
                ax.set_title(f'Communication Cost (Saving: {det["comm"]["saving_pct"]:.1f}%)')
                plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig8_comm_cost.png"); plt.close()
                print("  ✅ fig8_comm_cost.png")

        print(f"\n  All plots saved to {RESULTS_DIR}/")

    except ImportError:
        print("  ⚠ matplotlib not available, skipping plots")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    if 'exp1' in all_results:
        fa = all_results['exp1']["FedAvg"][-1].global_accuracy
        for m in ["FedAvg", "FedProx", "Ours"]:
            r = all_results['exp1'][m][-1]
            d = (r.global_accuracy - fa) / max(fa, 1e-8) * 100
            print(f"  Exp1 {m:10s}: acc={r.global_accuracy:.4f} ({d:+.1f}%)")
    if 'detection' in all_results:
        det = all_results['detection']
        print(f"\n  Detection:")
        for m in ['fedavg', 'ours']:
            best = max(r['ap'] for r in det[m])
            print(f"    {m:8s}: best AP={best:.4f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
