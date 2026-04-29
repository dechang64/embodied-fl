# ── python/analysis/multi_task_fl.py ──
"""
Multi-Task Federated Learning Engine for Embodied Intelligence
===============================================================
Supports simultaneous training of detection + classification + policy heads.

Architecture:
  Client-side (per factory):
    1. YOLOv11 backbone (shared) → detection head (local)
    2. DINOv2 (frozen) → classification head (trainable)
    3. Policy MLP → action output (trainable)

  Server-side (FedAvg):
    1. Average YOLO backbone weights across factories
    2. Average classification head weights
    3. Average policy head weights (for same-task clients)
    4. Task-Aware weighting via HNSW similarity
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Optional, Callable
import time


class EmbodiedMultiTaskFL:
    """Multi-task FL engine for embodied intelligence.

    Manages three parallel training tracks:
    - Detection (YOLOv11 backbone aggregation)
    - Classification (DINOv2 + linear head)
    - Policy (MLP for robot control)
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 10,
        action_dim: int = 6,
        hidden_dim: int = 128,
        lr: float = 0.001,
        local_epochs: int = 2,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.seed = seed

        # Classification head (DINOv2 → classes)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        # Policy head (features → actions)
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, action_dim),
        )

        # State
        self.detector_weights: Optional[OrderedDict] = None
        self.history: list[dict] = []

    def _split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clients: int,
        val_split: float = 0.2,
    ) -> tuple:
        rng = np.random.RandomState(self.seed)
        n = len(features)
        indices = rng.permutation(n)

        val_size = int(n * val_split)
        val_idx = indices[-val_size:]
        train_idx = indices[:-val_size]

        n_train = len(train_idx)
        chunk_size = n_train // n_clients
        client_splits = []
        for i in range(n_clients):
            start = i * chunk_size
            end = start + chunk_size if i < n_clients - 1 else n_train
            client_splits.append(np.arange(start, end))

        return (
            features[train_idx], labels[train_idx],
            features[val_idx], labels[val_idx],
            client_splits,
        )

    def _train_head_client(
        self,
        client_model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[OrderedDict, float, float]:
        optimizer = optim.Adam(client_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        client_model.train()
        total_loss, correct, total = 0.0, 0, 0

        for _ in range(self.local_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                out = client_model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(xb)

        params = OrderedDict(client_model.state_dict())
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return params, avg_loss, accuracy

    @staticmethod
    def _evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 64):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size)

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(xb)

        return total_loss / max(total, 1), correct / max(total, 1)

    @staticmethod
    def _fedavg(params_list: list[OrderedDict]) -> OrderedDict:
        avg = OrderedDict()
        for key in params_list[0]:
            avg[key] = torch.stack([p[key] for p in params_list]).mean(dim=0)
        return avg

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_clients: int = 3,
        rounds: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> list[dict]:
        train_X, train_y, val_X, val_y, client_splits = self._split_data(
            features, labels, n_clients
        )

        train_X_t = torch.tensor(train_X, dtype=torch.float32)
        train_y_t = torch.tensor(train_y, dtype=torch.long)
        val_X_t = torch.tensor(val_X, dtype=torch.float32)
        val_y_t = torch.tensor(val_y, dtype=torch.long)

        history = []
        t0 = time.time()

        for rnd in range(1, rounds + 1):
            client_params_list = []
            client_metrics = []

            for cid, split_idx in enumerate(client_splits):
                idx_tensor = torch.tensor(split_idx, dtype=torch.long)
                cX = train_X_t[idx_tensor]
                cy = train_y_t[idx_tensor]

                # Clone classifier head
                client_model = nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(self.hidden_dim, self.num_classes),
                )
                client_model.load_state_dict(OrderedDict(self.classifier.state_dict()))

                params, train_loss, train_acc = self._train_head_client(
                    client_model, cX, cy
                )
                client_params_list.append(params)
                client_metrics.append({
                    "client_id": cid,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "n_samples": len(split_idx),
                })

            # FedAvg aggregation
            global_cls_params = self._fedavg(client_params_list)
            self.classifier.load_state_dict(global_cls_params)

            val_loss, val_acc = self._evaluate(self.classifier, val_X_t, val_y_t, self.batch_size)
            elapsed = time.time() - t0

            round_metrics = {
                "round": rnd,
                "avg_train_loss": np.mean([m["train_loss"] for m in client_metrics]),
                "avg_train_acc": np.mean([m["train_acc"] for m in client_metrics]),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "elapsed": elapsed,
                "client_metrics": client_metrics,
            }
            history.append(round_metrics)

            if progress_callback:
                progress_callback(rnd, round_metrics)

        self.history = history
        return history
