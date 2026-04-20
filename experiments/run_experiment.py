"""
Embodied-FL Experiment Framework v5
=====================================
6 experiments, pure NumPy, Adam optimizer, cosine LR.

Exp1: FedAvg vs FedProx vs Ours (5 clients, Non-IID)
Exp2: Scalability (10 clients)
Exp3: Non-IID severity sweep
Exp4: Heterogeneous tasks (shared backbone)
Exp5: Continual learning (EWC + Replay vs Fine-tune)
Exp6: Gradient compression (Top-K + Quantization)
"""

import numpy as np
import json, os, time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from copy import deepcopy


# ═══════════════════════════════════════════════════════════════
#  MLP with Adam
# ═══════════════════════════════════════════════════════════════

class MLP:
    def __init__(self, sizes: List[int], seed=42, use_adam=True):
        self.rng = np.random.RandomState(seed)
        self.sizes = sizes
        self.use_adam = use_adam
        self._t = 0
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            s = np.sqrt(2.0 / sizes[i])
            self.W.append((self.rng.randn(sizes[i], sizes[i+1]) * s).astype(np.float32))
            self.b.append(np.zeros(sizes[i+1], dtype=np.float32))
        if use_adam:
            self._mw = [np.zeros_like(w) for w in self.W]
            self._vw = [np.zeros_like(w) for w in self.W]
            self._mb = [np.zeros_like(b) for b in self.b]
            self._vb = [np.zeros_like(b) for b in self.b]
        self._acts = []
        self._pre = []

    def forward(self, x):
        self._acts = [x]
        self._pre = []
        a = x
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            self._pre.append(z)
            a = np.maximum(0, z)
            self._acts.append(a)
        z = a @ self.W[-1] + self.b[-1]
        self._pre.append(z)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        self._acts.append(p)
        return p

    def _adam(self, param, grad, m, v, lr):
        self._t += 1
        m[:] = 0.9 * m + 0.1 * grad
        v[:] = 0.999 * v + 0.001 * grad**2
        mh = m / (1 - 0.9**self._t)
        vh = v / (1 - 0.999**self._t)
        param -= lr * mh / (np.sqrt(vh) + 1e-8)

    def train_step(self, x, y, lr=0.001):
        """Forward + backward + update. Returns loss."""
        m = x.shape[0]
        p = self.forward(x)
        loss = -np.sum(y * np.log(p + 1e-8)) / m
        d = (p - y) / m
        for i in range(len(self.W) - 1, -1, -1):
            dw = self._acts[i].T @ d
            db = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (self._pre[i-1] > 0)
            if self.use_adam:
                self._adam(self.W[i], dw, self._mw[i], self._vw[i], lr)
                self._adam(self.b[i], db, self._mb[i], self._vb[i], lr)
            else:
                self.W[i] -= lr * dw
                self.b[i] -= lr * db
        return float(loss)

    def compute_grad(self, x, y):
        """Compute gradients WITHOUT updating weights. Returns [dw0,db0,dw1,db1,...]."""
        m = x.shape[0]
        p = self.forward(x)
        d = (p - y) / m
        raw = []
        for i in range(len(self.W) - 1, -1, -1):
            dw = self._acts[i].T @ d
            db = d.sum(axis=0)
            raw.append((dw, db))
            if i > 0:
                d = (d @ self.W[i].T) * (self._pre[i-1] > 0)
        grads = []
        for dw, db in reversed(raw):
            grads.append(dw)
            grads.append(db)
        return grads

    def apply_grad(self, grads, lr=0.001):
        """Manually apply gradients (for compression experiments)."""
        for i in range(len(self.W)):
            if self.use_adam:
                self._adam(self.W[i], grads[2*i], self._mw[i], self._vw[i], lr)
                self._adam(self.b[i], grads[2*i+1], self._mb[i], self._vb[i], lr)
            else:
                self.W[i] -= lr * grads[2*i]
                self.b[i] -= lr * grads[2*i+1]

    def accuracy(self, x, y):
        p = self.forward(x)
        return float(np.mean(np.argmax(p, axis=1) == np.argmax(y, axis=1)))

    def get_params(self):
        r = []
        for w, b in zip(self.W, self.b):
            r.append(w.copy())
            r.append(b.copy())
        return r

    def set_params(self, params):
        for i in range(len(self.W)):
            self.W[i] = params[2*i].copy()
            self.b[i] = params[2*i+1].copy()

    def param_count(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def cosine_lr(lr0, step, total, warmup=5):
    if step < warmup:
        return lr0 * (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return lr0 * 0.5 * (1 + np.cos(np.pi * p))

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def generate_non_iid(n_samples, n_classes, alpha, dim, seed=42):
    """Dirichlet Non-IID split."""
    rng = np.random.RandomState(seed)
    proportions = rng.dirichlet([alpha] * n_classes)
    proportions = proportions / proportions.sum()
    counts = (proportions * n_samples).astype(int)
    counts[-1] = n_samples - counts[:-1].sum()

    X, y = [], []
    W_latent = rng.randn(dim, 16).astype(np.float32) * 0.5
    for c in range(n_classes):
        nc = counts[c]
        latent = rng.randn(nc, 16).astype(np.float32) + rng.randn(1, 16) * 1.5
        X.append(latent @ W_latent.T + rng.randn(nc, dim) * 0.3)
        yc = np.zeros((nc, n_classes), dtype=np.float32)
        yc[:, c] = 1.0
        y.append(yc)
    return np.vstack(X).astype(np.float32), np.vstack(y), proportions


# ═══════════════════════════════════════════════════════════════
#  Federated Aggregation
# ═══════════════════════════════════════════════════════════════

def agg_fedavg(updates, weights):
    total = sum(weights)
    return [sum(w / total * u[i] for u, w in zip(updates, weights)) for i in range(len(updates[0]))]

def agg_fedprox(updates, weights):
    return agg_fedavg(updates, weights)

def agg_ours(updates, weights, losses, task_embs, global_emb, temperature=0.5):
    """Task-Aware Aggregation for Embodied FL.

    Three signals blended:
      1. Performance rank (40%): clients with lower loss contribute more
      2. Data volume (30%): standard FL fairness
      3. Task similarity (30%): cosine sim to global task embedding

    Motivation: In embodied settings, some robots operate in easier environments
    (clean factory floor) while others face harder conditions (variable lighting).
    Performance weighting ensures reliable clients dominate aggregation.
    """
    n = len(updates)

    # 1. Rank-based performance weight
    loss_arr = np.array(losses)
    ranks = np.argsort(np.argsort(loss_arr))  # 0 = best (lowest loss)
    perf_w = 1.0 + (n - 1 - ranks) / max(n - 1, 1)  # best=2.0, worst=1.0
    perf_w = perf_w / perf_w.sum()

    # 2. Sample-count weight
    total_n = sum(weights)
    sample_w = np.array([w / total_n for w in weights])

    # 3. Similarity weight
    sims = np.array([cosine_sim(e, global_emb) for e in task_embs])
    sim_w = np.exp(sims / temperature)
    sim_w = sim_w / sim_w.sum()

    # Blend
    combined = 0.4 * perf_w + 0.3 * sample_w + 0.3 * sim_w
    combined = combined / combined.sum()

    return [sum(combined[i] * updates[i][j] for i in range(n)) for j in range(len(updates[0]))]


# ═══════════════════════════════════════════════════════════════
#  EWC + Replay Buffer
# ═══════════════════════════════════════════════════════════════

class EWC:
    def __init__(self, model, lam=10.0):
        self.model = model
        self.lam = lam
        self.fisher = None
        self.star_params = None

    def consolidate(self, X, y, n_samples=200):
        """Compute Fisher + store optimal params after a task."""
        model = self.model
        idx = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
        Xs, ys = X[idx], y[idx]

        self.fisher = []
        for w, b in zip(model.W, model.b):
            self.fisher.append(np.zeros_like(w))
            self.fisher.append(np.zeros_like(b))

        for i in range(Xs.shape[0]):
            p = model.forward(Xs[i:i+1])
            d = p - ys[i:i+1]
            for j in range(len(model.W) - 1, -1, -1):
                dw = model._acts[j].T @ d
                db = d.sum(axis=0)
                if j > 0:
                    d = (d @ model.W[j].T) * (model._pre[j-1] > 0)
                self.fisher[2*j] += dw**2
                self.fisher[2*j+1] += db**2
        for k in range(len(self.fisher)):
            self.fisher[k] /= Xs.shape[0]
        self.star_params = model.get_params()

    def penalty(self):
        """EWC regularization loss."""
        if self.fisher is None:
            return 0.0
        cur = self.model.get_params()
        return self.lam * sum(np.sum(f * (c - s)**2) for f, c, s in zip(self.fisher, cur, self.star_params))

    def grad_penalty(self, lr):
        """Apply EWC gradient penalty (clipped for stability)."""
        if self.fisher is None:
            return
        for i in range(len(self.model.W)):
            dw = self.lam * self.fisher[2*i] * (self.model.W[i] - self.star_params[2*i])
            db = self.lam * self.fisher[2*i+1] * (self.model.b[i] - self.star_params[2*i+1])
            dw = np.clip(dw, -0.5, 0.5)
            db = np.clip(db, -0.5, 0.5)
            self.model.W[i] -= lr * dw
            self.model.b[i] -= lr * db


# ═══════════════════════════════════════════════════════════════
#  Gradient Compression
# ═══════════════════════════════════════════════════════════════

def topk_sparsify(grads, sparsity=0.9):
    """Top-K sparsification: keep top (1-sparsity) values by magnitude."""
    sparse = []
    total_p, nonzero_p = 0, 0
    for g in grads:
        flat = g.flatten()
        total_p += len(flat)
        k = max(1, int(len(flat) * (1 - sparsity)))
        thr = np.partition(np.abs(flat), -k)[-k]
        mask = np.abs(flat) >= thr
        sg = np.where(mask, flat, 0.0).reshape(g.shape).astype(np.float32)
        sparse.append(sg)
        nonzero_p += np.count_nonzero(sg)
    ratio = total_p / max(nonzero_p, 1)
    return sparse, {"sparsity": sparsity, "ratio": ratio, "nonzero": nonzero_p, "total": total_p}

def quantize_grads(grads, bits=8):
    """Min-max uniform quantization."""
    quant = []
    for g in grads:
        gmin, gmax = g.min(), g.max()
        scale = (gmax - gmin) / (2**bits - 1) if gmax > gmin else 1.0
        q = np.round((g - gmin) / scale).astype(np.uint8 if bits <= 8 else np.uint16)
        dq = (q.astype(np.float32) * scale + gmin).reshape(g.shape)
        quant.append(dq)
    bpv = max(1, bits // 8)
    orig = sum(g.nbytes for g in grads)
    comp = sum(g.size * bpv for g in grads)
    return quant, {"bits": bits, "ratio": orig / max(comp, 1), "bytes": comp}


# ═══════════════════════════════════════════════════════════════
#  Experiment Runner
# ═══════════════════════════════════════════════════════════════

@dataclass
class ClientData:
    id: str
    X: np.ndarray
    y: np.ndarray
    label_dist: np.ndarray
    n: int
    task_emb: np.ndarray
    domain: str

@dataclass
class RoundResult:
    round_num: int
    global_loss: float
    global_accuracy: float
    per_client: list


def run_fl(clients, method, n_rounds, global_emb, arch,
           local_epochs=5, lr=0.001, mu=0.01):
    """Run federated learning experiment."""
    results = []
    gm = MLP(arch, seed=42)
    gp = gm.get_params()

    for rnd in range(1, n_rounds + 1):
        t0 = time.time()
        lr_now = cosine_lr(lr, rnd, n_rounds)
        updates, cw, losses, tembs = [], [], [], []

        for c in clients:
            lm = MLP(arch, seed=42)
            lm.set_params([p.copy() for p in gp])
            for _ in range(local_epochs):
                idx = np.random.permutation(c.n)
                loss = lm.train_step(c.X[idx], c.y[idx], lr=lr_now)
                if method == "FedProx":
                    for i in range(len(lm.W)):
                        lm.W[i] -= lr_now * mu * (lm.W[i] - gp[2*i])
                        lm.b[i] -= lr_now * mu * (lm.b[i] - gp[2*i+1])
            updates.append(lm.get_params())
            cw.append(c.n)
            losses.append(loss)
            tembs.append(c.task_emb)

        if method == "Ours":
            gp = agg_ours(updates, cw, losses, tembs, global_emb)
        elif method == "FedProx":
            gp = agg_fedprox(updates, cw)
        else:
            gp = agg_fedavg(updates, cw)

        gm.set_params(gp)
        tl, ta, pc = 0.0, 0.0, []
        for c in clients:
            p = gm.forward(c.X)
            l = -np.sum(c.y * np.log(p + 1e-8)) / c.n
            a = float(np.mean(np.argmax(p, 1) == np.argmax(c.y, 1)))
            tl += l; ta += a
            pc.append({"id": c.id, "loss": float(l), "accuracy": float(a)})
        nc = len(clients)
        results.append(RoundResult(rnd, tl/nc, ta/nc, pc))
        if rnd % 10 == 0 or rnd == 1:
            print(f"    R{rnd:3d} | loss={tl/nc:.4f} | acc={ta/nc:.4f} | {time.time()-t0:.2f}s")
    return results


def make_clients(n_clients, n_classes, alpha, dim, seed=42):
    """Create Non-IID clients."""
    rng = np.random.RandomState(seed)
    names = ["Suzhou", "Wuxi", "Kunshan", "Shenzhen", "Dongguan",
             "Hangzhou", "Nanjing", "Shanghai", "Chengdu", "Wuhan"]
    domains = ["PCB", "PCB", "PCB", "PCB", "PCB",
               "PCB", "PCB", "PCB", "PCB", "PCB"]
    clients = []
    for i in range(n_clients):
        n = rng.randint(300, 700)
        X, y, ld = generate_non_iid(n, n_classes, alpha, dim, seed + i)
        emb = rng.randn(32).astype(np.float32) * 0.1
        clients.append(ClientData(
            f"Factory-{chr(65+i)}", X, y, ld, n, emb,
            f"{domains[i] if i < len(domains) else 'PCB'}/{names[i] if i < len(names) else f'City-{i}'}"
        ))
    return clients


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Embodied-FL: Baseline Comparison Experiments v5")
    print("  FedAvg vs FedProx vs Ours (Task-Aware Aggregation)")
    print("  Optimizer: Adam | LR: Cosine decay | Model: 24→128→64→10")
    print("=" * 70)

    # ── Config ──
    DIM = 24
    HID = 128
    NCLS = 10
    ARCH = [DIM, HID, 64, NCLS]
    ROUNDS = 80
    EPOCHS = 5
    LR = 0.001

    # Global task embedding
    rng = np.random.RandomState(0)
    global_emb = np.ones(32, dtype=np.float32)
    global_emb[:10] = 2.0  # emphasize rare classes

    # ═══════════════════════════════════════════════════════════
    # Exp1: 5 Factories, Non-IID α=0.5
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Exp1: 5 Factories, Non-IID (α=0.5)")
    print("─" * 70)
    clients5 = make_clients(5, NCLS, 0.5, DIM, 42)
    for c in clients5:
        dom = int(np.argmax(c.label_dist))
        print(f"  {c.id:12s} | n={c.n:4d} | dominant={dom} ({c.label_dist[dom]:.0%})")

    exp1 = {}
    for method in ["FedAvg", "FedProx", "Ours"]:
        print(f"\n  Running {method}...")
        r = run_fl(clients5, method, ROUNDS, global_emb, ARCH, EPOCHS, LR)
        exp1[method] = r
        print(f"  Final: loss={r[-1].global_loss:.4f}, acc={r[-1].global_accuracy:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Exp2: 10 Factories (Scalability)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Exp2: 10 Factories (Scalability)")
    print("─" * 70)
    clients10 = make_clients(10, NCLS, 0.5, DIM, 42)
    exp2 = {}
    for method in ["FedAvg", "Ours"]:
        print(f"\n  Running {method}...")
        r = run_fl(clients10, method, ROUNDS, global_emb, ARCH, EPOCHS, LR)
        exp2[method] = r
        print(f"  Final: loss={r[-1].global_loss:.4f}, acc={r[-1].global_accuracy:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Exp3: Non-IID Severity
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Exp3: Non-IID Severity Sweep")
    print("─" * 70)
    exp3 = {}
    for alpha, label in [(5.0, "IID (α=5.0)"), (1.0, "Low (α=1.0)"),
                          (0.5, "Medium (α=0.5)"), (0.1, "High (α=0.1)")]:
        print(f"\n  {label}:")
        exp3[label] = {}
        for method in ["FedAvg", "Ours"]:
            cl = make_clients(5, NCLS, alpha, DIM, 42)
            r = run_fl(cl, method, ROUNDS, global_emb, ARCH, EPOCHS, LR)
            exp3[label][method] = r
            print(f"    {method:8s}: acc={r[-1].global_accuracy:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Exp4: Heterogeneous Tasks (Shared Backbone)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Exp4: Heterogeneous Tasks (inspection + grasping + assembly)")
    print("─" * 70)
    BB_ARCH = [DIM, 64, 32]
    BACKBONE = [DIM, 64, 32]
    tasks = [
        ("inspection", 10, 500), ("grasping", 6, 400), ("assembly", 4, 350)
    ]
    rng4 = np.random.RandomState(42)
    W_lat = rng4.randn(DIM, 16).astype(np.float32) * 0.5
    het_clients = []
    for tname, tcls, tn in tasks:
        X, y, _ = generate_non_iid(tn, tcls, 0.5, DIM, 42)
        emb = rng4.randn(32).astype(np.float32) * 0.1
        het_clients.append(ClientData(tname, X, y, _, tn, emb, tname))

    def run_hetero(method):
        bb = MLP(BACKBONE, seed=42)
        gbb = bb.get_params()
        heads = {c.id: MLP([32, 16, c.y.shape[1]], seed=42) for c in het_clients}
        results = []
        for rnd in range(1, ROUNDS + 1):
            t0 = time.time()
            lr_now = cosine_lr(0.001, rnd, ROUNDS)
            bb_ups, cw, losses, tembs = [], [], [], []
            for c in het_clients:
                lbb = MLP(BACKBONE, seed=42)
                lbb.set_params([p.copy() for p in gbb])
                head = heads[c.id]
                for _ in range(5):
                    feat = lbb.forward(c.X)
                    out = head.forward(feat)
                    m = c.n
                    loss = -np.sum(c.y * np.log(out + 1e-8)) / m
                    d = (out - c.y) / m
                    for i in range(len(head.W) - 1, -1, -1):
                        hdw = head._acts[i].T @ d
                        hdb = d.sum(axis=0)
                        if i > 0:
                            d = (d @ head.W[i].T) * (head._pre[i-1] > 0)
                        head._adam(head.W[i], hdw, head._mw[i], head._vw[i], lr_now)
                        head._adam(head.b[i], hdb, head._mb[i], head._vb[i], lr_now)
                    # Backbone grad through head
                    d_feat = d @ head.W[0].T
                    d_bb = d_feat * (lbb._pre[-1] > 0)  # ReLU grad
                    for i in range(len(lbb.W) - 1, -1, -1):
                        bdw = lbb._acts[i].T @ d_bb
                        bdb = d_bb.sum(axis=0)
                        if i > 0:
                            d_bb = (d_bb @ lbb.W[i].T) * (lbb._pre[i-1] > 0)
                        lbb._adam(lbb.W[i], bdw, lbb._mw[i], lbb._vw[i], lr_now)
                        lbb._adam(lbb.b[i], bdb, lbb._mb[i], lbb._vb[i], lr_now)
                bb_ups.append(lbb.get_params())
                cw.append(c.n)
                losses.append(loss)
                tembs.append(c.task_emb)
            if method == "Ours":
                gbb = agg_ours(bb_ups, cw, losses, tembs, global_emb)
            else:
                gbb = agg_fedavg(bb_ups, cw)
            bb.set_params(gbb)
            tl, ta, pc = 0.0, 0.0, []
            for c in het_clients:
                feat = bb.forward(c.X)
                out = heads[c.id].forward(feat)
                l = -np.sum(c.y * np.log(out + 1e-8)) / c.n
                a = float(np.mean(np.argmax(out, 1) == np.argmax(c.y, 1)))
                tl += l; ta += a
                pc.append({"id": c.id, "loss": float(l), "accuracy": float(a)})
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
        print(f"  Final: loss={r[-1].global_loss:.4f}, acc={r[-1].global_accuracy:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Exp5: Continual Learning (EWC + Replay)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Exp5: Continual Learning (EWC + Replay vs Fine-tune)")
    print("  Phase 1: classes 0-5 | Phase 2: classes 0-9 (new 6-9)")
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
            yc = np.zeros((nc, N_P2), dtype=np.float32)
            yc[:, c] = 1.0
            y.append(yc)
        return np.vstack(X).astype(np.float32), np.vstack(y)

    X_p1, y_p1 = make_phase_data(N_P1, 800, 42)
    X_p2, y_p2 = make_phase_data(N_P2, 800, 43)

    def run_ewc_exp(use_ewc, use_replay, lam=10.0, replay_ratio=0.2):
        model = MLP([DIM, HID, 64, N_P2], seed=42)
        ewc = EWC(model, lam) if use_ewc else None
        results = []
        total_steps = 50

        # Phase 1: classes 0-5
        for step in range(1, 26):
            lr_now = cosine_lr(0.001, step, total_steps, warmup=3)
            idx = np.random.permutation(len(X_p1))
            loss = model.train_step(X_p1[idx], y_p1[idx], lr=lr_now)
            # No EWC penalty in Phase 1 (not yet consolidated)
            acc_old = model.accuracy(X_p1, y_p1)
            results.append({"step": step, "phase": 1, "acc_old": acc_old, "acc_new": 0.0})

        if ewc:
            ewc.consolidate(X_p1, y_p1)

        # Phase 2: classes 0-9 with optional replay
        replay_size = int(len(X_p1) * replay_ratio) if use_replay else 0
        for step in range(26, 51):
            lr_now = cosine_lr(0.001, step, total_steps, warmup=3)
            idx = np.random.permutation(len(X_p2))
            X_batch, y_batch = X_p2[idx], y_p2[idx]

            # Mix in replay data
            if use_replay and replay_size > 0:
                ridx = np.random.choice(len(X_p1), replay_size, replace=False)
                X_batch = np.vstack([X_batch, X_p1[ridx]])
                y_batch = np.vstack([y_batch, y_p1[ridx]])

            loss = model.train_step(X_batch, y_batch, lr=lr_now)
            if ewc:
                ewc.grad_penalty(lr_now)
            acc_old = model.accuracy(X_p1, y_p1)
            acc_new = model.accuracy(X_p2, y_p2)
            results.append({"step": step, "phase": 2, "acc_old": acc_old, "acc_new": acc_new})
        return results

    exp5 = {}
    configs = [
        ("Fine-tune", False, False),
        ("EWC only", True, False),
        ("Replay only", False, True),
        ("EWC + Replay", True, True),
    ]
    for label, use_ewc, use_replay in configs:
        print(f"\n  Running {label}...")
        r = run_ewc_exp(use_ewc, use_replay)
        exp5[label] = r
        print(f"    Phase 2 end: old={r[-1]['acc_old']:.4f}, new={r[-1]['acc_new']:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Exp6: Gradient Compression
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  Exp6: Gradient Compression (50 training steps)")
    print("─" * 70)

    # Use IID data for compression experiment (focus on compression, not Non-IID)
    rng_c = np.random.RandomState(42)
    X_all = rng_c.randn(800, DIM).astype(np.float32)
    y_all = np.zeros((800, NCLS), dtype=np.float32)
    for i in range(800):
        y_all[i, i % NCLS] = 1.0
    # Add some structure so it's not trivially separable
    W_proj = rng_c.randn(DIM, 16).astype(np.float32) * 0.5
    for c in range(NCLS):
        mask = np.arange(800) % NCLS == c
        X_all[mask] += rng_c.randn(mask.sum(), DIM).astype(np.float32) * 0.5

    X_comp, y_comp = X_all[:600], y_all[:600]
    X_val, y_val = X_all[600:], y_all[600:]
    n_params = MLP(ARCH, seed=42).param_count()
    print(f"  Model: {n_params} params | Train: 600 | Val: 200 (IID)")

    def train_with_compression(compress_fn, n_steps=50, label=""):
        model = MLP(ARCH, seed=42)
        for step in range(1, n_steps + 1):
            lr_now = cosine_lr(0.001, step, n_steps, warmup=3)
            idx = np.random.permutation(len(X_comp))
            grads = model.compute_grad(X_comp[idx], y_comp[idx])
            if compress_fn:
                grads, _ = compress_fn(grads)
            model.apply_grad(grads, lr=lr_now)
        return model.accuracy(X_val, y_val)

    exp6 = {}

    # Baseline (no compression)
    acc_base = train_with_compression(None, 50)
    exp6["No compression"] = {"ratio": 1.0, "accuracy": acc_base, "bytes": n_params * 4}
    print(f"\n  No compression: acc={acc_base:.4f}")

    # Top-K
    print("\n  Top-K Sparsification:")
    for sp in [0.5, 0.7, 0.9, 0.95, 0.99]:
        _, info = topk_sparsify([np.zeros(100)], sp)  # just get ratio
        # Actual training
        def make_topk(s=sp):
            return lambda g: topk_sparsify(g, s)
        acc = train_with_compression(make_topk(), 50)
        _, stats = topk_sparsify([np.zeros(n_params)], sp)
        exp6[f"TopK-{int(sp*100)}"] = {"ratio": stats["ratio"], "accuracy": acc, "bytes": int(stats["nonzero"] * 6)}
        print(f"    {int(sp*100)}%: {stats['ratio']:.1f}x, acc={acc:.4f}")

    # Quantization
    print("\n  Quantization:")
    for bits in [16, 8, 4]:
        def make_quant(b=bits):
            return lambda g: quantize_grads(g, b)
        acc = train_with_compression(make_quant(), 50)
        _, stats = quantize_grads([np.zeros(n_params, dtype=np.float32)], bits)
        exp6[f"Quant-{bits}bit"] = {"ratio": stats["ratio"], "accuracy": acc, "bytes": stats["bytes"]}
        print(f"    {bits}-bit: {stats['ratio']:.1f}x, acc={acc:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  Exp1 (5 factories, {ROUNDS} rounds):")
    fa1 = exp1["FedAvg"][-1].global_accuracy
    for m in ["FedAvg", "FedProx", "Ours"]:
        r = exp1[m][-1]
        d = (r.global_accuracy - fa1) / max(fa1, 1e-8) * 100
        star = " ★" if d > 0.5 else ""
        print(f"    {m:10s} loss={r.global_loss:.4f} acc={r.global_accuracy:.4f} ({d:+.1f}%){star}")

    print(f"\n  Exp2 (10 factories):")
    fa2 = exp2["FedAvg"][-1].global_accuracy
    for m in ["FedAvg", "Ours"]:
        r = exp2[m][-1]
        d = (r.global_accuracy - fa2) / max(fa2, 1e-8) * 100
        star = " ★" if d > 0.5 else ""
        print(f"    {m:10s} loss={r.global_loss:.4f} acc={r.global_accuracy:.4f} ({d:+.1f}%){star}")

    print(f"\n  Exp3 (Non-IID severity):")
    for sev in exp3:
        fa = exp3[sev]["FedAvg"][-1].global_accuracy
        ou = exp3[sev]["Ours"][-1].global_accuracy
        d = (ou - fa) / max(fa, 1e-8) * 100
        print(f"    {sev:20s} FedAvg={fa:.4f} Ours={ou:.4f} ({d:+.1f}%)")

    print(f"\n  Exp4 (Heterogeneous):")
    fa4 = exp4["FedAvg"][-1].global_accuracy
    for m in ["FedAvg", "Ours"]:
        r = exp4[m][-1]
        d = (r.global_accuracy - fa4) / max(fa4, 1e-8) * 100
        star = " ★" if d > 0.5 else ""
        print(f"    {m:10s} acc={r.global_accuracy:.4f} ({d:+.1f}%){star}")

    print(f"\n  Exp5 (Continual Learning):")
    ft_old = exp5["Fine-tune"][-1]["acc_old"]
    for label in exp5:
        r = exp5[label][-1]
        d = (r["acc_old"] - ft_old) / max(ft_old, 1e-8) * 100
        star = " ★" if d > 5 else ""
        print(f"    {label:16s} old={r['acc_old']:.4f} new={r['acc_new']:.4f} (Δold={d:+.1f}%){star}")

    print(f"\n  Exp6 (Compression):")
    for label, info in exp6.items():
        print(f"    {label:18s} {info['ratio']:>6.1f}x  acc={info['accuracy']:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════
    def ser_rounds(rds):
        return [{"round": r.round_num, "loss": r.global_loss,
                 "accuracy": r.global_accuracy} for r in rds]

    output = {
        "config": {"arch": ARCH, "rounds": ROUNDS, "lr": LR, "optimizer": "Adam"},
        "exp1": {m: ser_rounds(r) for m, r in exp1.items()},
        "exp2": {m: ser_rounds(r) for m, r in exp2.items()},
        "exp3": {s: {m: ser_rounds(r) for m, r in methods.items()} for s, methods in exp3.items()},
        "exp4": {m: ser_rounds(r) for m, r in exp4.items()},
        "exp5": {l: [{"step": r["step"], "phase": r["phase"],
                       "acc_old": r["acc_old"], "acc_new": r["acc_new"]} for r in rs]
                 for l, rs in exp5.items()},
        "exp6": exp6,
    }

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/experiment_results.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o))
    print(f"\n✅ Results saved to {results_dir}/experiment_results.json")

    # ═══════════════════════════════════════════════════════════
    # Generate Plots
    # ═══════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"FedAvg": "#e74c3c", "FedProx": "#f39c12", "Ours": "#2ecc71"}
        markers = {"FedAvg": "o", "FedProx": "s", "Ours": "^"}

        # Fig 1: Convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for m in ["FedAvg", "FedProx", "Ours"]:
            rds = exp1[m]
            ax1.plot([r.round_num for r in rds], [r.global_loss for r in rds],
                    label=m, color=colors[m], marker=markers[m], markersize=2, linewidth=1.5)
            ax2.plot([r.round_num for r in rds], [r.global_accuracy for r in rds],
                    label=m, color=colors[m], marker=markers[m], markersize=2, linewidth=1.5)
        ax1.set_xlabel("Round"); ax1.set_ylabel("Loss"); ax1.set_title("(a) Loss"); ax1.legend()
        ax2.set_xlabel("Round"); ax2.set_ylabel("Accuracy"); ax2.set_title("(b) Accuracy"); ax2.legend()
        plt.tight_layout(); plt.savefig(f"{results_dir}/fig1_convergence.png"); plt.close()
        print("  ✅ fig1_convergence.png")

        # Fig 2: Non-IID severity bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        sevs = list(exp3.keys())
        x = np.arange(len(sevs))
        w = 0.35
        fa_accs = [exp3[s]["FedAvg"][-1].global_accuracy for s in sevs]
        ou_accs = [exp3[s]["Ours"][-1].global_accuracy for s in sevs]
        ax.bar(x - w/2, fa_accs, w, label="FedAvg", color="#e74c3c", alpha=0.8)
        ax.bar(x + w/2, ou_accs, w, label="Ours", color="#2ecc71", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("(")[0].strip() for s in sevs], rotation=15)
        ax.set_ylabel("Accuracy"); ax.set_title("Non-IID Severity: FedAvg vs Ours"); ax.legend()
        for i, (fa, ou) in enumerate(zip(fa_accs, ou_accs)):
            ax.text(i - w/2, fa + 0.005, f"{fa:.3f}", ha='center', fontsize=7)
            ax.text(i + w/2, ou + 0.005, f"{ou:.3f}", ha='center', fontsize=7)
        plt.tight_layout(); plt.savefig(f"{results_dir}/fig2_noniid_severity.png"); plt.close()
        print("  ✅ fig2_noniid_severity.png")

        # Fig 3: Scalability
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for m in ["FedAvg", "Ours"]:
            rds = exp2[m]
            ax1.plot([r.round_num for r in rds], [r.global_loss for r in rds],
                    label=m, color=colors[m], linewidth=1.5)
            ax2.plot([r.round_num for r in rds], [r.global_accuracy for r in rds],
                    label=m, color=colors[m], linewidth=1.5)
        ax1.set_xlabel("Round"); ax1.set_ylabel("Loss"); ax1.set_title("(a) 10 Clients Loss"); ax1.legend()
        ax2.set_xlabel("Round"); ax2.set_ylabel("Accuracy"); ax2.set_title("(b) 10 Clients Accuracy"); ax2.legend()
        plt.tight_layout(); plt.savefig(f"{results_dir}/fig3_scalability.png"); plt.close()
        print("  ✅ fig3_scalability.png")

        # Fig 4: EWC + Replay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ewc_colors = {"Fine-tune": "#e74c3c", "EWC only": "#3498db",
                      "Replay only": "#f39c12", "EWC + Replay": "#2ecc71"}
        for label, rds in exp5.items():
            steps = [r["step"] for r in rds]
            ax1.plot(steps, [r["acc_old"] for r in rds], label=label,
                    color=ewc_colors[label], linewidth=1.5)
            p2 = [r for r in rds if r["phase"] == 2]
            if p2:
                ax2.plot([r["step"] for r in p2], [r["acc_new"] for r in p2],
                        label=label, color=ewc_colors[label], linewidth=1.5)
        ax1.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel("Step"); ax1.set_ylabel("Old Classes Acc")
        ax1.set_title("(a) Forgetting Prevention"); ax1.legend(fontsize=7)
        ax2.set_xlabel("Step (Phase 2)"); ax2.set_ylabel("New Classes Acc")
        ax2.set_title("(b) New Task Learning"); ax2.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(f"{results_dir}/fig4_ewc_replay.png"); plt.close()
        print("  ✅ fig4_ewc_replay.png")

        # Fig 5: Compression trade-off
        fig, ax = plt.subplots(figsize=(8, 5))
        ms, cs, acs, cls = [], [], [], []
        for label, info in exp6.items():
            ms.append(label); cs.append(info["ratio"]); acs.append(info["accuracy"])
            cls.append("#3498db" if "TopK" in label else "#e67e22" if "Quant" in label else "#95a5a6")
        ax.scatter(cs, acs, c=cls, s=100, zorder=5, edgecolors='white')
        for i, m in enumerate(ms):
            ax.annotate(m, (cs[i], acs[i]), textcoords="offset points", xytext=(8, 5), fontsize=7)
        ax.set_xlabel("Compression Ratio (×)"); ax.set_ylabel("Accuracy")
        ax.set_title("Communication Cost vs Accuracy"); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{results_dir}/fig5_compression_tradeoff.png"); plt.close()
        print("  ✅ fig5_compression_tradeoff.png")

        print(f"\n  All plots saved to {results_dir}/")
    except ImportError:
        print("  ⚠ matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
