"""
Embodied-FL Experiment Framework v3
=====================================
Key insight: All clients share the SAME task (same output_dim) but have
different data distributions (Non-IID). This is the standard FL setting
where task-aware aggregation should shine.

Scenario: 5 factories all doing PCB defect classification (10 classes)
but with different defect distributions (Non-IID label skew).
"""

import numpy as np
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy


# ============================================================
# Neural Network (pure NumPy)
# ============================================================

class MLP:
    """Multi-layer perceptron with ReLU + softmax output."""

    def __init__(self, layer_sizes: List[int], seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self._activations = []
        self._pre_acts = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)
            self.weights.append(self.rng.randn(fan_in, fan_out).astype(np.float32) * std)
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._activations = [x]
        self._pre_acts = []
        a = x
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            self._pre_acts.append(z)
            a = np.maximum(0, z)
            self._activations.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        self._pre_acts.append(z)
        # Softmax
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        self._activations.append(probs)
        return probs

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        m = x.shape[0]
        probs = self.forward(x)
        # Cross-entropy loss
        loss = -np.sum(y * np.log(probs + 1e-8)) / m
        # Gradient of softmax + cross-entropy
        delta = (probs - y) / m
        for i in range(len(self.weights) - 1, -1, -1):
            dw = self._activations[i].T @ delta
            db = np.sum(delta, axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self._pre_acts[i - 1] > 0).astype(np.float32)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
        return loss

    def get_params(self) -> List[np.ndarray]:
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.copy())
            params.append(b.copy())
        return params

    def set_params(self, params: List[np.ndarray]):
        for i in range(len(self.weights)):
            self.weights[i] = params[2 * i].copy()
            self.biases[i] = params[2 * i + 1].copy()

    def param_count(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        probs = self.forward(x)
        return float(np.mean(np.argmax(probs, axis=1) == np.argmax(y, axis=1)))


# ============================================================
# Data Generation: Non-IID Label Skew (Dirichlet)
# ============================================================

def generate_non_iid_data(n_samples: int, n_classes: int, alpha: float,
                          input_dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Non-IID data with Dirichlet label distribution.
    
    alpha controls Non-IID severity:
      alpha → ∞: uniform (IID)
      alpha → 0: extreme Non-IID (each client has few classes)
    """
    rng = np.random.RandomState(seed)

    # Sample label distribution from Dirichlet
    label_dist = rng.dirichlet(np.ones(n_classes) * alpha)
    labels = rng.choice(n_classes, size=n_samples, p=label_dist)

    # Generate features: class-conditional Gaussians
    X = np.zeros((n_samples, input_dim), dtype=np.float32)
    # Shared basis (backbone should learn this)
    W_shared = rng.randn(input_dim, 16).astype(np.float32) * 0.5
    for c in range(n_classes):
        mask = labels == c
        n_c = mask.sum()
        if n_c == 0:
            continue
        # Class-specific latent
        latent = rng.randn(n_c, 16).astype(np.float32) + rng.randn(1, 16) * 1.5
        X[mask] = latent @ W_shared.T + rng.randn(n_c, input_dim) * 0.3

    # One-hot labels
    y = np.zeros((n_samples, n_classes), dtype=np.float32)
    y[np.arange(n_samples), labels] = 1.0

    return X, y, label_dist


# ============================================================
# Federated Aggregation
# ============================================================

def aggregate_fedavg(updates: List[Dict], client_weights: List[float]) -> List[np.ndarray]:
    total = sum(client_weights)
    aggregated = []
    for j in range(len(updates[0]["params"])):
        param = sum(w * u["params"][j] for w, u in zip(client_weights, updates))
        aggregated.append(param / total)
    return aggregated


def aggregate_fedprox(updates: List[Dict], client_weights: List[float]) -> List[np.ndarray]:
    # Same aggregation as FedAvg (proximal term applied client-side)
    return aggregate_fedavg(updates, client_weights)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def aggregate_task_aware(updates: List[Dict], client_weights: List[float],
                         task_embeddings: List[np.ndarray],
                         global_embedding: np.ndarray,
                         alpha: float = 0.3, beta: float = 0.7) -> Tuple[List[np.ndarray], Dict]:
    """Task-Aware Aggregation: weight by task similarity + sample count."""
    total_samples = sum(client_weights)
    similarities = []
    for emb in task_embeddings:
        sim = cosine_similarity(emb, global_embedding)
        similarities.append(max(0.05, sim))

    combined = []
    for i, (nw, sim) in enumerate(zip(client_weights, similarities)):
        w = alpha * (nw / total_samples) + beta * sim
        combined.append(w)

    total_w = sum(combined)
    combined = [w / total_w for w in combined]

    aggregated = []
    for j in range(len(updates[0]["params"])):
        param = sum(w * u["params"][j] for w, u in zip(combined, updates))
        aggregated.append(param)

    return aggregated, {"weights": combined, "similarities": similarities}


# ============================================================
# Experiment Data Structures
# ============================================================

@dataclass
class ClientData:
    client_id: str
    X: np.ndarray
    y: np.ndarray
    label_dist: np.ndarray
    n_samples: int
    task_embedding: np.ndarray
    factory_type: str  # "similar" or "dissimilar" to global need


@dataclass
class RoundResult:
    round_num: int
    global_loss: float
    global_accuracy: float
    per_client: List[Dict]
    communication_bytes: float
    wall_time: float
    aggregation_weights: Optional[List[float]] = None


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(clients: List[ClientData], method: str, n_rounds: int,
                   global_embedding: np.ndarray, model_arch: List[int],
                   local_epochs: int = 5, lr: float = 0.01,
                   mu: float = 0.01) -> List[RoundResult]:
    """Run a federated learning experiment."""
    n_classes = model_arch[-1]
    results = []

    # Initialize global model
    global_model = MLP(model_arch, seed=42)
    global_params = global_model.get_params()
    param_bytes = sum(p.nbytes for p in global_params)

    for rnd in range(1, n_rounds + 1):
        t0 = time.time()
        updates = []
        client_weights = []
        task_embeddings = []

        for client in clients:
            # Local model starts from global
            local_model = MLP(model_arch, seed=42)
            local_model.set_params([p.copy() for p in global_params])

            # Local training with multiple epochs
            for _ in range(local_epochs):
                # Mini-batch
                idx = np.random.permutation(client.n_samples)
                X_batch = client.X[idx]
                y_batch = client.y[idx]
                loss = local_model.backward(X_batch, y_batch, lr=lr)

                # FedProx: add proximal regularization
                if method == "FedProx":
                    for i in range(len(local_model.weights)):
                        local_model.weights[i] -= lr * mu * (local_model.weights[i] - global_params[2 * i])
                        local_model.biases[i] -= lr * mu * (local_model.biases[i] - global_params[2 * i + 1])

            # Evaluate locally
            local_acc = local_model.accuracy(client.X, client.y)

            updates.append({"params": local_model.get_params(), "loss": loss, "accuracy": local_acc})
            client_weights.append(client.n_samples)
            task_embeddings.append(client.task_embedding)

        # Aggregate
        if method == "Ours":
            new_params, agg_info = aggregate_task_aware(
                updates, client_weights, task_embeddings, global_embedding)
            agg_weights = agg_info["weights"]
        elif method == "FedProx":
            new_params = aggregate_fedprox(updates, client_weights)
            agg_weights = None
        else:
            new_params = aggregate_fedavg(updates, client_weights)
            agg_weights = None

        global_params = new_params

        # Evaluate global model on all clients
        global_model.set_params(global_params)
        total_loss, total_acc, per_client = 0.0, 0.0, []
        for client in clients:
            probs = global_model.forward(client.X)
            loss = -np.sum(client.y * np.log(probs + 1e-8)) / client.n_samples
            acc = float(np.mean(np.argmax(probs, axis=1) == np.argmax(client.y, axis=1)))
            total_loss += loss
            total_acc += acc
            per_client.append({"id": client.client_id, "loss": float(loss), "accuracy": float(acc)})

        n_clients = len(clients)
        wall_time = time.time() - t0
        results.append(RoundResult(
            round_num=rnd,
            global_loss=total_loss / n_clients,
            global_accuracy=total_acc / n_clients,
            per_client=per_client,
            communication_bytes=param_bytes * n_clients / 1024,
            wall_time=wall_time,
            aggregation_weights=agg_weights,
        ))

        if rnd % 10 == 0 or rnd == 1:
            print(f"    R{rnd:3d} | loss={results[-1].global_loss:.4f} | acc={results[-1].global_accuracy:.4f} | "
                  f"clients={n_clients} | {wall_time:.2f}s")

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  Embodied-FL: Baseline Comparison Experiments v3")
    print("  FedAvg vs FedProx vs Ours (Task-Aware Aggregation)")
    print("  Task: PCB Defect Classification (10 classes, Non-IID)")
    print("=" * 70)

    # Config
    INPUT_DIM = 24
    HIDDEN = 64
    N_CLASSES = 10
    MODEL_ARCH = [INPUT_DIM, HIDDEN, 32, N_CLASSES]
    N_ROUNDS = 50
    LOCAL_EPOCHS = 5
    LR = 0.01

    # Global task embedding: represents what the federation "needs"
    # In our scenario: the federation needs a model good at ALL defect types
    # We set higher weight on rare defects (classes 7,8,9) that few clients have
    global_embedding = np.zeros(32, dtype=np.float32)
    global_embedding[0:10] = np.array([0.3, 0.3, 0.3, 0.3, 0.3,  # common defects
                                        0.5, 0.5, 0.8, 0.8, 0.8])  # rare defects (high need)

    # ============================================================
    # Experiment 1: 5 Factories, Non-IID Label Skew
    # ============================================================
    print("\n" + "─" * 70)
    print("  Experiment 1: 5 Factories, Non-IID Label Skew (α=0.5)")
    print("─" * 70)

    exp1_configs = [
        {"id": "Factory-A (Suzhou)",   "n": 500, "alpha": 0.5, "seed": 42,
         "desc": "Electronics SMT line"},
        {"id": "Factory-B (Wuxi)",     "n": 400, "alpha": 0.3, "seed": 123,
         "desc": "Automotive PCB line"},
        {"id": "Factory-C (Kunshan)",  "n": 600, "alpha": 0.8, "seed": 456,
         "desc": "3C assembly line"},
        {"id": "Factory-D (Shenzhen)", "n": 450, "alpha": 0.2, "seed": 789,
         "desc": "Semiconductor fab"},
        {"id": "Factory-E (Dongguan)", "n": 380, "alpha": 0.4, "seed": 321,
         "desc": "Logistics sorting"},
    ]

    # Create clients
    def create_clients(configs):
        clients = []
        for cfg in configs:
            X, y, label_dist = generate_non_iid_data(
                cfg["n"], N_CLASSES, cfg["alpha"], INPUT_DIM, cfg["seed"])
            # Task embedding: based on label distribution
            task_emb = np.zeros(32, dtype=np.float32)
            task_emb[0:N_CLASSES] = label_dist.astype(np.float32)
            task_emb[16] = cfg["n"] / 1000.0
            task_emb[17] = cfg["alpha"]
            # Add some noise
            rng = np.random.RandomState(cfg["seed"] + 999)
            task_emb[20:] = rng.randn(12).astype(np.float32) * 0.05
            clients.append(ClientData(
                client_id=cfg["id"], X=X, y=y, label_dist=label_dist,
                n_samples=cfg["n"], task_embedding=task_emb,
                factory_type=cfg.get("desc", "")
            ))
        return clients

    clients = create_clients(exp1_configs)
    for c in clients:
        dominant = np.argmax(c.label_dist)
        print(f"  {c.client_id:25s} | n={c.n_samples:4d} | α={exp1_configs[clients.index(c)]['alpha']:.1f} | "
              f"dominant class={dominant} ({c.label_dist[dominant]:.1%})")

    all_results = {}
    for method in ["FedAvg", "FedProx", "Ours"]:
        print(f"\n  Running {method}...")
        fresh = create_clients(exp1_configs)
        results = run_experiment(fresh, method, N_ROUNDS, global_embedding,
                                MODEL_ARCH, LOCAL_EPOCHS, LR)
        all_results[method] = results
        print(f"    Final: loss={results[-1].global_loss:.4f}, acc={results[-1].global_accuracy:.4f}")

    # ============================================================
    # Experiment 2: Scalability (10 clients)
    # ============================================================
    print("\n" + "─" * 70)
    print("  Experiment 2: 10 Factories (Scalability)")
    print("─" * 70)

    exp2_configs = []
    cities = ["Suzhou", "Wuxi", "Kunshan", "Shenzhen", "Dongguan",
              "Nanjing", "Hangzhou", "Shanghai", "Chengdu", "Wuhan"]
    for i, city in enumerate(cities):
        exp2_configs.append({
            "id": f"Factory-{chr(65+i)} ({city})",
            "n": 300 + (i * 37) % 200,
            "alpha": 0.2 + (i * 0.13),
            "seed": 42 + i * 100,
            "desc": f"Plant {i+1}"
        })

    exp2_results = {}
    for method in ["FedAvg", "Ours"]:
        print(f"\n  Running {method}...")
        fresh = create_clients(exp2_configs)
        results = run_experiment(fresh, method, N_ROUNDS, global_embedding,
                                MODEL_ARCH, LOCAL_EPOCHS, LR)
        exp2_results[method] = results
        print(f"    Final: loss={results[-1].global_loss:.4f}, acc={results[-1].global_accuracy:.4f}")

    # ============================================================
    # Experiment 3: Non-IID Severity
    # ============================================================
    print("\n" + "─" * 70)
    print("  Experiment 3: Non-IID Severity (α varies)")
    print("─" * 70)

    base_configs = [
        {"id": "C1", "n": 500, "alpha": 1.0, "seed": 42},
        {"id": "C2", "n": 400, "alpha": 1.0, "seed": 123},
        {"id": "C3", "n": 600, "alpha": 1.0, "seed": 456},
    ]

    exp3_results = {}
    for severity, alpha_val in [("IID (α=5.0)", 5.0), ("Low (α=1.0)", 1.0),
                                 ("Medium (α=0.5)", 0.5), ("High (α=0.1)", 0.1)]:
        print(f"\n  {severity}:")
        cfgs = []
        for i, bc in enumerate(base_configs):
            cfgs.append({**bc, "alpha": alpha_val, "id": f"{bc['id']}-{severity[:4]}"})
        exp3_results[severity] = {}
        for method in ["FedAvg", "Ours"]:
            fresh = create_clients(cfgs)
            results = run_experiment(fresh, method, N_ROUNDS, global_embedding,
                                    MODEL_ARCH, LOCAL_EPOCHS, LR)
            exp3_results[severity][method] = results
            print(f"    {method:8s}: loss={results[-1].global_loss:.4f}, acc={results[-1].global_accuracy:.4f}")

    # ============================================================
    # Experiment 4: Heterogeneous Tasks (core contribution)
    # 3 task types: inspection(10cls), grasping(6cls), assembly(4cls)
    # Shared backbone, task-specific heads
    # Task-aware should outperform FedAvg because it weights by task relevance
    # ============================================================
    print("\n" + "─" * 70)
    print("  Experiment 4: Heterogeneous Tasks (Shared Backbone)")
    print("  inspection(10cls) + grasping(6cls) + assembly(4cls)")
    print("─" * 70)

    BACKBONE_ARCH = [INPUT_DIM, HIDDEN, 32]  # shared
    HEAD_ARCHS = {
        "inspection": [32, 16, 10],
        "grasping": [32, 16, 6],
        "assembly": [32, 16, 4],
    }

    exp4_configs = [
        # 2 inspection factories (similar tasks)
        {"id": "Insp-A (Suzhou)",   "task": "inspection", "domain": "electronics", "n": 500, "seed": 42},
        {"id": "Insp-B (Shenzhen)", "task": "inspection", "domain": "semiconductor", "n": 450, "seed": 789},
        # 2 grasping factories
        {"id": "Grasp-A (Wuxi)",    "task": "grasping", "domain": "automotive", "n": 400, "seed": 123},
        {"id": "Grasp-B (Dongguan)","task": "grasping", "domain": "logistics", "n": 380, "seed": 321},
        # 1 assembly factory (dissimilar)
        {"id": "Asm-A (Kunshan)",   "task": "assembly", "domain": "manufacturing", "n": 600, "seed": 456},
    ]

    # Task embeddings: same task → high similarity, different task → low similarity
    task_type_embeddings = {
        "inspection": np.array([1.0, 0.8, 0.2, 0.1, 0.0] + [0.3] * 27, dtype=np.float32),
        "grasping":   np.array([0.2, 1.0, 0.7, 0.1, 0.0] + [0.3] * 27, dtype=np.float32),
        "assembly":   np.array([0.1, 0.2, 1.0, 0.8, 0.5] + [0.3] * 27, dtype=np.float32),
    }

    # Global need: we want a backbone good at inspection + grasping (more factories)
    global_emb_hetero = np.mean(
        [task_type_embeddings[c["task"]] for c in exp4_configs], axis=0)

    def create_hetero_clients(configs):
        clients = []
        for cfg in configs:
            n_out = len(HEAD_ARCHS[cfg["task"]])
            X, y, label_dist = generate_non_iid_data(
                cfg["n"], n_out, 0.5, INPUT_DIM, cfg["seed"])
            # Task embedding = task type + domain variation
            emb = task_type_embeddings[cfg["task"]].copy()
            rng = np.random.RandomState(cfg["seed"] + 999)
            emb += rng.randn(32).astype(np.float32) * 0.1  # domain variation
            clients.append(ClientData(
                client_id=cfg["id"], X=X, y=y, label_dist=label_dist,
                n_samples=cfg["n"], task_embedding=emb,
                factory_type=f"{cfg['task']}/{cfg['domain']}"
            ))
        return clients

    def run_hetero_experiment(clients, method, n_rounds, global_emb, local_epochs=5, lr=0.01):
        """FL with shared backbone + task-specific heads."""
        results = []
        # Init backbone
        backbone = MLP(BACKBONE_ARCH, seed=42)
        global_bb_params = backbone.get_params()
        bb_bytes = sum(p.nbytes for p in global_bb_params)

        # Init heads per client
        heads = {}
        for c in clients:
            n_out = c.y.shape[1]
            heads[c.client_id] = MLP([32, 16, n_out], seed=42)

        for rnd in range(1, n_rounds + 1):
            t0 = time.time()
            bb_updates = []
            client_weights = []
            task_embs = []

            for c in clients:
                # Local backbone
                local_bb = MLP(BACKBONE_ARCH, seed=42)
                local_bb.set_params([p.copy() for p in global_bb_params])
                local_head = heads[c.client_id]

                # Train backbone + head jointly
                for _ in range(local_epochs):
                    # Forward
                    feat = local_bb.forward(c.X)
                    out = local_head.forward(feat)
                    m = c.n_samples
                    loss = -np.sum(c.y * np.log(out + 1e-8)) / m

                    # Backprop head
                    delta = (out - c.y) / m
                    for i in range(len(local_head.weights) - 1, -1, -1):
                        dw = local_head._activations[i].T @ delta
                        db = np.sum(delta, axis=0)
                        if i > 0:
                            delta = (delta @ local_head.weights[i].T) * (local_head._pre_acts[i-1] > 0).astype(np.float32)
                        else:
                            # Last iteration: compute gradient w.r.t. input (features)
                            delta = delta @ local_head.weights[i].T  # shape: (batch, 32)
                        local_head.weights[i] -= lr * dw
                        local_head.biases[i] -= lr * db

                    # Backprop backbone
                    delta_feat = delta  # shape: (batch, 32) = backbone output dim
                    for i in range(len(local_bb.weights) - 1, -1, -1):
                        dw = local_bb._activations[i].T @ delta_feat
                        db = np.sum(delta_feat, axis=0)
                        if i > 0:
                            delta_feat = (delta_feat @ local_bb.weights[i].T) * (local_bb._pre_acts[i-1] > 0).astype(np.float32)
                        local_bb.weights[i] -= lr * dw
                        local_bb.biases[i] -= lr * db

                bb_updates.append({"params": local_bb.get_params()})
                client_weights.append(c.n_samples)
                task_embs.append(c.task_embedding)

            # Aggregate backbone only
            if method == "Ours":
                new_bb, agg_info = aggregate_task_aware(
                    bb_updates, client_weights, task_embs, global_emb, alpha=0.3, beta=0.7)
                agg_weights = agg_info["weights"]
            else:
                new_bb = aggregate_fedavg(bb_updates, client_weights)
                agg_weights = None

            global_bb_params = new_bb

            # Evaluate
            eval_bb = MLP(BACKBONE_ARCH, seed=42)
            eval_bb.set_params(global_bb_params)
            total_acc, total_loss, per_client = 0.0, 0.0, []
            for c in clients:
                feat = eval_bb.forward(c.X)
                out = heads[c.client_id].forward(feat)
                loss = -np.sum(c.y * np.log(out + 1e-8)) / c.n_samples
                acc = float(np.mean(np.argmax(out, axis=1) == np.argmax(c.y, axis=1)))
                total_acc += acc
                total_loss += loss
                per_client.append({"id": c.client_id, "loss": float(loss), "accuracy": float(acc)})

            n_c = len(clients)
            wt = time.time() - t0
            results.append(RoundResult(
                round_num=rnd, global_loss=total_loss/n_c, global_accuracy=total_acc/n_c,
                per_client=per_client, communication_bytes=bb_bytes*n_c/1024,
                wall_time=wt, aggregation_weights=agg_weights))

            if rnd % 10 == 0 or rnd == 1:
                print(f"    R{rnd:3d} | loss={results[-1].global_loss:.4f} | acc={results[-1].global_accuracy:.4f} | {wt:.2f}s")

        return results

    exp4_results = {}
    for method in ["FedAvg", "Ours"]:
        print(f"\n  Running {method}...")
        fresh = create_hetero_clients(exp4_configs)
        results = run_hetero_experiment(fresh, method, N_ROUNDS, global_emb_hetero)
        exp4_results[method] = results
        print(f"    Final: loss={results[-1].global_loss:.4f}, acc={results[-1].global_accuracy:.4f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  Experiment 1 (5 factories, {N_ROUNDS} rounds, Non-IID):")
    print(f"  {'Method':<10} {'Final Loss':>12} {'Final Acc':>12} {'Best Acc':>12} {'Δ vs FedAvg':>14}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*14}")
    fa_loss = all_results["FedAvg"][-1].global_loss
    fa_acc = all_results["FedAvg"][-1].global_accuracy
    for method in ["FedAvg", "FedProx", "Ours"]:
        r = all_results[method]
        best = max(rr.global_accuracy for rr in r)
        delta_l = (fa_loss - r[-1].global_loss) / fa_loss * 100
        star = " ★" if method == "Ours" else ""
        print(f"  {method:<10} {r[-1].global_loss:>12.4f} {r[-1].global_accuracy:>12.4f} "
              f"{best:>12.4f} {delta_l:>+13.1f}%{star}")

    print(f"\n  Experiment 4 (Heterogeneous Tasks, {N_ROUNDS} rounds):")
    print(f"  {'Method':<10} {'Final Loss':>12} {'Final Acc':>12} {'Δ vs FedAvg':>14}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*14}")
    fa4_loss = exp4_results["FedAvg"][-1].global_loss
    fa4_acc = exp4_results["FedAvg"][-1].global_accuracy
    for method in ["FedAvg", "Ours"]:
        r = exp4_results[method][-1]
        delta_l = (fa4_loss - r.global_loss) / fa4_loss * 100
        delta_a = (r.global_accuracy - fa4_acc) / max(fa4_acc, 1e-8) * 100
        star = " ★" if method == "Ours" else ""
        print(f"  {method:<10} {r.global_loss:>12.4f} {r.global_accuracy:>12.4f} {delta_l:>+13.1f}%{star}")

    print(f"\n  Experiment 3 (Non-IID severity):")
    print(f"  {'Severity':<18} {'FedAvg Acc':>12} {'Ours Acc':>12} {'Improvement':>14}")
    print(f"  {'─'*18} {'─'*12} {'─'*12} {'─'*14}")
    for sev in exp3_results:
        fa = exp3_results[sev]["FedAvg"][-1].global_accuracy
        ou = exp3_results[sev]["Ours"][-1].global_accuracy
        imp = (ou - fa) / max(fa, 1e-8) * 100
        print(f"  {sev:<18} {fa:>12.4f} {ou:>12.4f} {imp:>+13.1f}%")

    # Save
    output = {
        "config": {
            "model_arch": MODEL_ARCH, "n_rounds": N_ROUNDS,
            "local_epochs": LOCAL_EPOCHS, "lr": LR, "n_classes": N_CLASSES,
        },
        "exp1": {method: [{"round": r.round_num, "loss": r.global_loss,
                            "accuracy": r.global_accuracy} for r in results]
                 for method, results in all_results.items()},
        "exp2": {method: [{"round": r.round_num, "loss": r.global_loss,
                            "accuracy": r.global_accuracy} for r in results]
                 for method, results in exp2_results.items()},
        "exp3": {sev: {method: [{"round": r.round_num, "loss": r.global_loss,
                                  "accuracy": r.global_accuracy} for r in results]
                       for method, results in methods.items()}
                 for sev, methods in exp3_results.items()},
        "exp4_hetero": {method: [{"round": r.round_num, "loss": r.global_loss,
                                   "accuracy": r.global_accuracy,
                                   "per_client": r.per_client} for r in results]
                        for method, results in exp4_results.items()},
    }

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")

    # Generate plots
    generate_plots(output, results_dir)

    return output


def generate_plots(data: dict, output_dir: str):
    """Generate publication-quality plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    colors = {'FedAvg': '#1f77b4', 'FedProx': '#ff7f0e', 'Ours': '#2ca02c'}
    markers = {'FedAvg': 'o', 'FedProx': 's', 'Ours': 'D'}
    linestyles = {'FedAvg': '-', 'FedProx': '--', 'Ours': '-'}

    # === Fig 1: Convergence curves (Exp1) ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in ["FedAvg", "FedProx", "Ours"]:
        rounds = [r["round"] for r in data["exp1"][method]]
        losses = [r["loss"] for r in data["exp1"][method]]
        accs = [r["accuracy"] for r in data["exp1"][method]]
        ax1.plot(rounds, losses, label=method, color=colors[method],
                marker=markers[method], markersize=3, linestyle=linestyles[method], linewidth=1.5)
        ax2.plot(rounds, accs, label=method, color=colors[method],
                marker=markers[method], markersize=3, linestyle=linestyles[method], linewidth=1.5)

    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Global Loss (Cross-Entropy)")
    ax1.set_title("(a) Convergence - Loss")
    ax1.legend()

    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Global Accuracy")
    ax2.set_title("(b) Convergence - Accuracy")
    ax2.legend()

    plt.savefig(f"{output_dir}/fig1_convergence.png")
    plt.close()
    print(f"  ✅ fig1_convergence.png")

    # === Fig 2: Non-IID severity (Exp3) ===
    fig, ax = plt.subplots(figsize=(8, 5))

    severities = list(data["exp3"].keys())
    x = np.arange(len(severities))
    width = 0.35

    fedavg_accs = [data["exp3"][s]["FedAvg"][-1]["accuracy"] for s in severities]
    ours_accs = [data["exp3"][s]["Ours"][-1]["accuracy"] for s in severities]

    bars1 = ax.bar(x - width/2, fedavg_accs, width, label='FedAvg', color=colors["FedAvg"], alpha=0.8)
    bars2 = ax.bar(x + width/2, ours_accs, width, label='Ours (Task-Aware)', color=colors["Ours"], alpha=0.8)

    ax.set_xlabel("Non-IID Severity")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Impact of Non-IID Severity on Model Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(severities, rotation=15, ha='right')
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.savefig(f"{output_dir}/fig2_noniid_severity.png")
    plt.close()
    print(f"  ✅ fig2_noniid_severity.png")

    # === Fig 3: Scalability (Exp2) ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in ["FedAvg", "Ours"]:
        rounds = [r["round"] for r in data["exp2"][method]]
        losses = [r["loss"] for r in data["exp2"][method]]
        accs = [r["accuracy"] for r in data["exp2"][method]]
        ax1.plot(rounds, losses, label=method, color=colors[method],
                marker=markers[method], markersize=3, linestyle=linestyles[method], linewidth=1.5)
        ax2.plot(rounds, accs, label=method, color=colors[method],
                marker=markers[method], markersize=3, linestyle=linestyles[method], linewidth=1.5)

    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Global Loss")
    ax1.set_title("(a) 10 Clients - Loss")
    ax1.legend()
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Global Accuracy")
    ax2.set_title("(b) 10 Clients - Accuracy")
    ax2.legend()

    plt.savefig(f"{output_dir}/fig3_scalability.png")
    plt.close()
    print(f"  ✅ fig3_scalability.png")

    print(f"\n  All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
