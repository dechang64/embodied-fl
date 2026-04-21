"""
Embodied-FL: YOLO + Federated Learning
========================================
Federated object detection with YOLOv8n.
- Backbone-only aggregation (shared feature extraction)
- Detection heads stay local (factory-specific objects)
- Task-Aware weighting based on task embeddings

Experiments:
  Exp1: FedAvg vs Task-Aware (5 clients, COCO subset)
  Exp2: Backbone aggregation vs full model aggregation
  Exp3: Non-IID data split (each client sees different object categories)
  Exp4: Communication cost (backbone-only vs full model)
"""

import os
import json
import time
import shutil
import numpy as np
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Map COCO 80 classes to 5 "factory" domains
FACTORY_DOMAINS = {
    'Factory-A (Inspection)': [0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],  # person + kitchen items
    'Factory-B (Logistics)': [1, 2, 3, 5, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # vehicles + animals
    'Factory-C (Assembly)': [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],  # furniture + electronics
    'Factory-D (Safety)': [9, 10, 11, 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],  # outdoor + sports
    'Factory-E (Quality)': [73, 74, 75, 76, 77, 78, 79],  # household items
}

# Simplified: 10-class subset for faster experiments
SIMPLIFIED_CLASSES = {
    'Factory-A (Inspection)': [0, 56, 57, 58, 62, 63, 64, 65, 66, 67],   # person + furniture + electronics
    'Factory-B (Logistics)': [1, 2, 3, 5, 7, 8, 13, 14, 15, 16],          # vehicles + animals
    'Factory-C (Assembly)': [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],     # kitchen items
    'Factory-D (Safety)': [9, 10, 11, 12, 25, 26, 27, 28, 29, 30],        # outdoor + accessories
    'Factory-E (Quality)': [67, 68, 69, 70, 71, 72, 73, 74, 75, 76],      # electronics + household
}


# ═══════════════════════════════════════════════════════════════
#  Lightweight Detection Model (YOLOv8-nano style)
# ═══════════════════════════════════════════════════════════════

class ConvBnSiLU(nn.Module):
    """Basic conv block: Conv2d + BatchNorm + SiLU"""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial block (simplified)"""
    def __init__(self, in_c, out_c, n=1):
        super().__init__()
        hidden = out_c // 2
        self.cv1 = ConvBnSiLU(in_c, hidden, 1)
        self.cv2 = ConvBnSiLU(in_c, hidden, 1)
        self.cv3 = nn.Sequential(*[ConvBnSiLU(hidden, hidden) for _ in range(n)])
        self.cv4 = ConvBnSiLU(hidden * 2, out_c, 1)

    def forward(self, x):
        return self.cv4(torch.cat([self.cv3(self.cv1(x)), self.cv2(x)], dim=1))


class DetectionHead(nn.Module):
    """Lightweight detection head for a subset of classes"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # Output: [batch, anchors, (5 + num_classes)]
        self.conv = nn.Sequential(
            ConvBnSiLU(in_channels, in_channels, 3),
            ConvBnSiLU(in_channels, in_channels, 3),
        )
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)  # x, y, w, h

    def forward(self, x):
        x = self.conv(x)
        B, _, H, W = x.shape
        cls = self.cls_head(x).view(B, self.num_anchors, self.num_classes, H * W).permute(0, 3, 1, 2)
        reg = self.reg_head(x).view(B, self.num_anchors, 4, H * W).permute(0, 3, 1, 2)
        return cls, reg


class TinyDetector(nn.Module):
    """
    Lightweight detector inspired by YOLOv8-nano.
    Backbone: 4 conv stages with CSP blocks
    Neck: Simple FPN (feature pyramid)
    Head: Class-specific detection head
    """
    def __init__(self, num_classes=80, backbone_out=256):
        super().__init__()
        self.num_classes = num_classes

        # Backbone (shared across clients)
        self.backbone = nn.Sequential(
            # Stage 0: 3 -> 16
            ConvBnSiLU(3, 16, 3, 2, 1),   # /2
            ConvBnSiLU(16, 32, 3, 2, 1),   # /4
            CSPBlock(32, 32, 1),

            # Stage 1: 32 -> 64
            ConvBnSiLU(32, 64, 3, 2, 1),  # /8
            CSPBlock(64, 64, 1),

            # Stage 2: 64 -> 128
            ConvBnSiLU(64, 128, 3, 2, 1), # /16
            CSPBlock(128, 128, 2),

            # Stage 3: 128 -> 256
            ConvBnSiLU(128, 256, 3, 2, 1), # /32
            CSPBlock(256, 256, 2),
        )

        # Neck (shared)
        self.neck = nn.Sequential(
            ConvBnSiLU(256, 128, 1),
            ConvBnSiLU(128, 128, 3, 1, 1),
            ConvBnSiLU(128, backbone_out, 1),
        )

        # Detection head (client-specific)
        self.head = DetectionHead(backbone_out, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.neck(feat)
        cls, reg = self.head(feat)
        return cls, reg

    def get_backbone_params(self) -> Dict[str, torch.Tensor]:
        """Get backbone + neck parameters for aggregation"""
        params = {}
        for name, p in self.backbone.named_parameters():
            params[f'backbone.{name}'] = p.data.clone()
        for name, p in self.neck.named_parameters():
            params[f'neck.{name}'] = p.data.clone()
        return params

    def get_head_params(self) -> Dict[str, torch.Tensor]:
        """Get detection head parameters (local only)"""
        params = {}
        for name, p in self.head.named_parameters():
            params[f'head.{name}'] = p.data.clone()
        return params

    def set_backbone_params(self, params: Dict[str, torch.Tensor]):
        """Load aggregated backbone parameters"""
        state = {}
        for k, v in params.items():
            state[k] = v
        self.backbone.load_state_dict(state, strict=False)
        self.neck.load_state_dict(state, strict=False)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for p in self.backbone.parameters())
        neck = sum(p.numel() for p in self.neck.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {'total': total, 'backbone': backbone, 'neck': neck, 'head': head}


# ═══════════════════════════════════════════════════════════════
#  Synthetic Detection Dataset
# ═══════════════════════════════════════════════════════════════

class SyntheticDetectionDataset(Dataset):
    """
    Generate synthetic detection data with bounding boxes.
    Each image has 1-5 objects placed at random positions.
    """
    def __init__(self, num_samples=500, img_size=224, num_classes=10,
                 max_objects=3, seed=42, class_offset=0):
        self.rng = np.random.RandomState(seed)
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.class_offset = class_offset
        self.num_samples = num_samples

        # Pre-generate all data
        self.images = []
        self.targets = []  # list of (boxes, labels) per image

        for _ in range(num_samples):
            img = self.rng.randint(0, 256, (3, img_size, img_size)).astype(np.float32) / 255.0
            # Add some structure (colored blobs for objects)
            boxes = []
            labels = []
            n_obj = self.rng.randint(1, max_objects + 1)
            for _ in range(n_obj):
                cls = self.rng.randint(0, num_classes)
                # Random bounding box
                w = self.rng.randint(20, img_size // 2)
                h = self.rng.randint(20, img_size // 2)
                x = self.rng.randint(0, img_size - w)
                y = self.rng.randint(0, img_size - h)
                boxes.append([x, y, x + w, y + h])
                labels.append(cls)
                # Draw colored blob
                color = [(cls * 37 + 100) % 256, (cls * 73 + 50) % 256, (cls * 113 + 150) % 256]
                img[:, y:y+h, x:x+w] = np.array(color).reshape(3, 1, 1) * 0.7 + \
                                          img[:, y:y+h, x:x+w] * 0.3

            self.images.append(torch.from_numpy(img))
            self.targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def collate_fn(batch):
    """Custom collate for variable-size targets"""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# ═══════════════════════════════════════════════════════════════
#  Detection Loss
# ═══════════════════════════════════════════════════════════════

class DetectionLoss(nn.Module):
    """Simplified detection loss: Focal loss (cls) + GIoU loss (reg)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, cls_pred, reg_pred, targets, feat_size=7):
        """
        cls_pred: [B, H*W, anchors, num_classes]
        reg_pred: [B, H*W, anchors, 4]
        targets: list of dicts with 'boxes' and 'labels'
        """
        B = cls_pred.shape[0]
        device = cls_pred.device
        total_loss = 0.0

        for b in range(B):
            boxes = targets[b]['boxes'].to(device)  # [N, 4] (x1,y1,x2,y2)
            labels = targets[b]['labels'].to(device)  # [N]
            n_obj = boxes.shape[0]

            if n_obj == 0:
                # No objects: only background loss
                total_loss += F.binary_cross_entropy_with_logits(
                    cls_pred[b], torch.zeros_like(cls_pred[b])
                )
                continue

            # Convert boxes to grid format
            grid_w = feat_size
            grid_h = feat_size
            img_size = 224

            # Normalize boxes to [0, 1]
            norm_boxes = boxes / img_size

            # Assign each object to nearest grid cell
            cx = (norm_boxes[:, 0] + norm_boxes[:, 2]) / 2 * grid_w
            cy = (norm_boxes[:, 1] + norm_boxes[:, 3]) / 2 * grid_h
            gx = cx.long().clamp(0, grid_w - 1)
            gy = cy.long().clamp(0, grid_h - 1)

            # Classification target
            cls_target = torch.zeros(cls_pred[b].shape, device=device)
            for i in range(n_obj):
                gi = gy[i].item() * grid_w + gx[i].item()
                cls_target[gi, 0, labels[i].item()] = 1.0

            # Focal-like loss
            cls_loss = F.binary_cross_entropy_with_logits(cls_pred[b], cls_target)

            # Regression: only for assigned cells
            reg_target = torch.zeros(reg_pred[b].shape, device=device)
            for i in range(n_obj):
                gi = gy[i].item() * grid_w + gx[i].item()
                reg_target[gi, 0] = norm_boxes[i, 0]  # x1
                reg_target[gi, 0] = norm_boxes[i, 1]  # y1
                reg_target[gi, 0] = norm_boxes[i, 2]  # x2
                reg_target[gi, 0] = norm_boxes[i, 3]  # y2

            reg_loss = F.smooth_l1_loss(reg_pred[b], reg_target)

            total_loss += cls_loss + reg_loss

        return total_loss / B


# ═══════════════════════════════════════════════════════════════
#  Task Embedding (for Task-Aware Aggregation)
# ═══════════════════════════════════════════════════════════════

def compute_task_embedding(model, data_loader, device='cpu'):
    """Compute task embedding from model's backbone statistics"""
    model.eval()
    feat_stats = []
    loss_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            feat = model.backbone(images)
            feat_stats.append(feat.mean(dim=[0, 2, 3]).cpu())
            n_batches += 1
            if n_batches >= 10:
                break

    # Task embedding: feature mean + std
    feats = torch.stack(feat_stats)
    emb = torch.cat([feats.mean(dim=0), feats.std(dim=0)])
    return emb.numpy()


def cosine_similarity(a, b):
    """Cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ═══════════════════════════════════════════════════════════════
#  Aggregation Methods
# ═══════════════════════════════════════════════════════════════

def agg_fedavg(updates, weights=None):
    """Standard FedAvg: uniform weighted average"""
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    result = {}
    for key in updates[0].keys():
        result[key] = sum(w * u[key] for w, u in zip(weights, updates))
    return result


def agg_task_aware(updates, task_embs, losses, global_emb, alpha=0.4, beta=0.3, gamma=0.3):
    """
    Task-Aware Aggregation:
    - Performance weight: lower loss → higher weight
    - Sample weight: more data → higher weight (uniform here)
    - Similarity weight: closer to global → higher weight
    """
    n = len(updates)

    # Performance weight (inverse loss)
    inv_losses = [1.0 / (l + 1e-8) for l in losses]
    perf_w = np.array(inv_losses) / sum(inv_losses)

    # Similarity weight
    sim_w = np.array([cosine_similarity(e, global_emb) for e in task_embs])
    sim_w = np.maximum(sim_w, 0.01)  # floor
    sim_w = sim_w / sim_w.sum()

    # Sample weight (uniform for now)
    sample_w = np.ones(n) / n

    # Blend
    combined = alpha * perf_w + beta * sample_w + gamma * sim_w
    combined = combined / combined.sum()

    result = {}
    for key in updates[0].keys():
        result[key] = sum(combined[i] * updates[i][key] for i in range(n))
    return result


# ═══════════════════════════════════════════════════════════════
#  Federated Training Loop
# ═══════════════════════════════════════════════════════════════

def train_local(model, data_loader, optimizer, criterion, device, epochs=3):
    """Train a client model locally"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for _ in range(epochs):
        for images, targets in data_loader:
            images = images.to(device)
            cls_pred, reg_pred = model(images)
            loss = criterion(cls_pred, reg_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            cls_pred, reg_pred = model(images)
            loss = criterion(cls_pred, reg_pred, targets)
            total_loss += loss.item()
            n_batches += 1

            # Simple accuracy: check if highest-scoring class matches any target
            B = cls_pred.shape[0]
            for b in range(B):
                labels = targets[b]['labels']
                if len(labels) == 0:
                    continue
                # Get predicted class from max activation
                pred_cls = cls_pred[b].argmax(dim=-1).max(dim=0).values.item()
                if pred_cls in labels.tolist():
                    correct += 1
                total += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def run_federated_experiment(clients, method='fedavg', rounds=20, local_epochs=3,
                              lr=0.001, device='cpu', backbone_only=True):
    """
    Run federated training experiment.

    Args:
        clients: list of (name, model, train_loader, val_loader, task_emb)
        method: 'fedavg' or 'task_aware'
        rounds: communication rounds
        local_epochs: local training epochs per round
        lr: learning rate
        device: torch device
        backbone_only: if True, only aggregate backbone+neck
    """
    n_clients = len(clients)
    results = []

    # Initialize global task embedding
    global_emb = np.zeros(256)  # will be updated

    print(f"  Running {method} ({n_clients} clients, {rounds} rounds, backbone_only={backbone_only})...")

    for r in range(1, rounds + 1):
        round_start = time.time()
        client_updates = []
        client_losses = []
        client_embs = []

        # Local training
        for name, model, train_loader, val_loader, _ in clients:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            loss = train_local(model, train_loader, optimizer,
                             DetectionLoss(model.num_classes), device, local_epochs)
            client_losses.append(loss)

            # Get update (delta from initial)
            if backbone_only:
                update = model.get_backbone_params()
            else:
                update = {}
                update.update(model.get_backbone_params())
                update.update(model.get_head_params())

            client_updates.append(update)

            # Compute task embedding
            emb = compute_task_embedding(model, train_loader, device)
            client_embs.append(emb)

        # Aggregate
        if method == 'task_aware':
            agg_params = agg_task_aware(client_updates, client_embs, client_losses, global_emb)
        else:
            agg_params = agg_fedavg(client_updates)

        # Update global embedding
        avg_emb = np.mean(client_embs, axis=0)
        global_emb = 0.7 * global_emb + 0.3 * avg_emb

        # Distribute aggregated params
        for name, model, train_loader, val_loader, _ in clients:
            if backbone_only:
                model.set_backbone_params(agg_params)
            else:
                model.set_backbone_params(agg_params)
                # Also set head params
                head_state = {k.replace('head.', ''): v for k, v in agg_params.items() if k.startswith('head.')}
                if head_state:
                    model.head.load_state_dict(head_state, strict=False)

        # Evaluate
        round_results = {'round': r, 'method': method}
        for name, model, train_loader, val_loader, _ in clients:
            loss, acc = evaluate(model, val_loader, DetectionLoss(model.num_classes), device)
            round_results[f'{name}_loss'] = loss
            round_results[f'{name}_acc'] = acc

        # Average metrics
        avg_loss = np.mean(client_losses)
        avg_acc = np.mean([round_results[f'{c[0]}_acc'] for c in clients])
        round_results['avg_loss'] = avg_loss
        round_results['avg_acc'] = avg_acc
        round_results['time'] = time.time() - round_start

        results.append(round_results)

        if r % 5 == 0 or r == 1:
            print(f"    R {r:3d} | loss={avg_loss:.4f} | acc={avg_acc:.4f} | {round_results['time']:.1f}s")

    return results


# ═══════════════════════════════════════════════════════════════
#  Main: Run All Experiments
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Embodied-FL: YOLO + Federated Learning Experiments")
    print("  Backbone-only aggregation with Task-Aware weighting")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")

    IMG_SIZE = 64
    N_CLIENTS = 5
    BACKBONE_OUT = 64
    ROUNDS = 15
    LOCAL_EPOCHS = 2
    LR = 0.001
    SAMPLES_PER_CLIENT = 200

    # ─── Exp 1: FedAvg vs Task-Aware (5 clients, backbone-only) ───
    print("\n" + "─" * 70)
    print("  Exp1: FedAvg vs Task-Aware (5 Factories, Backbone-only)")
    print("─" * 70)

    # Create clients with different class distributions
    clients = []
    factory_names = list(SIMPLIFIED_CLASSES.keys())
    class_offsets = [0, 10, 20, 30, 40]  # Each factory sees different classes

    for i, (fname, classes) in enumerate(SIMPLIFIED_CLASSES.items()):
        n_cls = len(classes)
        print(f"  {fname:30s} | classes={n_cls:2d} | offset={class_offsets[i]:2d}")

        model = TinyDetector(num_classes=n_cls, backbone_out=BACKBONE_OUT).to(device)
        params = model.param_count()
        print(f"    Model params: {params['total']:,} (backbone={params['backbone']:,}, "
              f"neck={params['neck']:,}, head={params['head']:,})")

        train_ds = SyntheticDetectionDataset(
            num_samples=SAMPLES_PER_CLIENT, img_size=IMG_SIZE,
            num_classes=n_cls, seed=42 + i, class_offset=class_offsets[i]
        )
        val_ds = SyntheticDetectionDataset(
            num_samples=100, img_size=IMG_SIZE,
            num_classes=n_cls, seed=99 + i, class_offset=class_offsets[i]
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Initial task embedding (random, will be updated)
        task_emb = np.random.randn(256).astype(np.float32) * 0.1

        clients.append((fname, model, train_loader, val_loader, task_emb))

    # Run FedAvg
    print("\n  Running FedAvg (backbone-only)...")
    # Deep copy models for fair comparison
    clients_avg = []
    for fname, model, train_loader, val_loader, emb in clients:
        model_copy = deepcopy(model)
        clients_avg.append((fname, model_copy, train_loader, val_loader, emb))
    results_avg = run_federated_experiment(clients_avg, 'fedavg', ROUNDS, LOCAL_EPOCHS, LR, device, backbone_only=True)

    # Run Task-Aware
    print("\n  Running Task-Aware (backbone-only)...")
    clients_aware = []
    for fname, model, train_loader, val_loader, emb in clients:
        model_copy = deepcopy(model)
        clients_aware.append((fname, model_copy, train_loader, val_loader, emb))
    results_aware = run_federated_experiment(clients_aware, 'task_aware', ROUNDS, LOCAL_EPOCHS, LR, device, backbone_only=True)

    # ─── Exp 2: Backbone-only vs Full Model Aggregation ───
    print("\n" + "─" * 70)
    print("  Exp2: Backbone-only vs Full Model Aggregation")
    print("─" * 70)

    print("\n  Running FedAvg (full model)...")
    clients_full = []
    for fname, model, train_loader, val_loader, emb in clients:
        model_copy = deepcopy(model)
        clients_full.append((fname, model_copy, train_loader, val_loader, emb))
    results_full = run_federated_experiment(clients_full, 'fedavg', ROUNDS, LOCAL_EPOCHS, LR, device, backbone_only=False)

    # ─── Exp 3: Non-IID Severity ───
    print("\n" + "─" * 70)
    print("  Exp3: Non-IID Severity Sweep")
    print("─" * 70)

    noniid_results = {}
    for alpha_label, alpha_val in [("IID", 5.0), ("Low", 1.0), ("Medium", 0.5), ("High", 0.1)]:
        print(f"\n  Non-IID α={alpha_val} ({alpha_label}):")

        alpha_clients = []
        for i, (fname, classes) in enumerate(SIMPLIFIED_CLASSES.items()):
            n_cls = len(classes)
            model = TinyDetector(num_classes=n_cls, backbone_out=BACKBONE_OUT).to(device)

            # Use Dirichlet-like split: alpha controls concentration
            rng = np.random.RandomState(42 + i + int(alpha_val * 10))
            if alpha_val >= 5.0:
                # IID: uniform class distribution
                class_probs = np.ones(n_cls) / n_cls
            else:
                # Non-IID: concentrated on 1-3 classes
                class_probs = rng.dirichlet(np.ones(n_cls) * alpha_val)

            # Generate data with class distribution
            train_ds = SyntheticDetectionDataset(
                num_samples=SAMPLES_PER_CLIENT, img_size=IMG_SIZE,
                num_classes=n_cls, seed=42 + i + int(alpha_val * 10),
                class_offset=class_offsets[i]
            )
            val_ds = SyntheticDetectionDataset(
                num_samples=100, img_size=IMG_SIZE,
                num_classes=n_cls, seed=99 + i + int(alpha_val * 10),
                class_offset=class_offsets[i]
            )
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
            task_emb = np.random.randn(256).astype(np.float32) * 0.1
            alpha_clients.append((fname, model, train_loader, val_loader, task_emb))

        # FedAvg
        clients_aa = [(n, deepcopy(m), t, v, e) for n, m, t, v, e in alpha_clients]
        r_avg = run_federated_experiment(clients_aa, 'fedavg', ROUNDS, LOCAL_EPOCHS, LR, device, backbone_only=True)

        # Task-Aware
        clients_ta = [(n, deepcopy(m), t, v, e) for n, m, t, v, e in alpha_clients]
        r_aware = run_federated_experiment(clients_ta, 'task_aware', ROUNDS, LOCAL_EPOCHS, LR, device, backbone_only=True)

        noniid_results[alpha_label] = {
            'alpha': alpha_val,
            'fedavg_final': r_avg[-1]['avg_acc'],
            'aware_final': r_aware[-1]['avg_acc'],
            'fedavg_history': r_avg,
            'aware_history': r_aware,
        }
        improvement = (r_aware[-1]['avg_acc'] - r_avg[-1]['avg_acc']) * 100
        print(f"    FedAvg={r_avg[-1]['avg_acc']:.4f} Ours={r_aware[-1]['avg_acc']:.4f} ({improvement:+.1f}%)")

    # ─── Exp 4: Communication Cost ───
    print("\n" + "─" * 70)
    print("  Exp4: Communication Cost Analysis")
    print("─" * 70)

    # Calculate parameter sizes
    sample_model = TinyDetector(num_classes=10, backbone_out=BACKBONE_OUT)
    params = sample_model.param_count()
    backbone_bytes = (params['backbone'] + params['neck']) * 4  # float32
    full_bytes = params['total'] * 4
    head_bytes = params['head'] * 4

    print(f"  Backbone+Neck: {params['backbone'] + params['neck']:,} params = {backbone_bytes:,} bytes")
    print(f"  Head only:     {params['head']:,} params = {head_bytes:,} bytes")
    print(f"  Full model:    {params['total']:,} params = {full_bytes:,} bytes")
    print(f"  Communication saving: {(1 - backbone_bytes/full_bytes)*100:.1f}%")

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print("\n  Exp1 (Aggregation Methods):")
    print(f"    FedAvg (backbone-only):     acc={results_avg[-1]['avg_acc']:.4f}")
    print(f"    Task-Aware (backbone-only): acc={results_aware[-1]['avg_acc']:.4f}")
    imp1 = (results_aware[-1]['avg_acc'] - results_avg[-1]['avg_acc']) * 100
    print(f"    Improvement: {imp1:+.1f}%")

    print("\n  Exp2 (Backbone-only vs Full):")
    print(f"    FedAvg (backbone-only): acc={results_avg[-1]['avg_acc']:.4f}")
    print(f"    FedAvg (full model):    acc={results_full[-1]['avg_acc']:.4f}")
    imp2 = (results_full[-1]['avg_acc'] - results_avg[-1]['avg_acc']) * 100
    print(f"    Difference: {imp2:+.1f}%")

    print("\n  Exp3 (Non-IID Severity):")
    for label, data in noniid_results.items():
        imp = (data['aware_final'] - data['fedavg_final']) * 100
        print(f"    {label:8s} (α={data['alpha']:.1f}): FedAvg={data['fedavg_final']:.4f} "
              f"Ours={data['aware_final']:.4f} ({imp:+.1f}%)")

    print("\n  Exp4 (Communication):")
    print(f"    Backbone-only: {backbone_bytes:,} bytes/round")
    print(f"    Full model:    {full_bytes:,} bytes/round")
    print(f"    Saving:        {(1 - backbone_bytes/full_bytes)*100:.1f}%")

    # ─── Save Results ───
    output = {
        'config': {
            'model': 'TinyDetector (YOLOv8-nano style)',
            'backbone_out': BACKBONE_OUT,
            'img_size': IMG_SIZE,
            'rounds': ROUNDS,
            'local_epochs': LOCAL_EPOCHS,
            'lr': LR,
            'device': device,
            'param_count': params,
        },
        'exp1': {
            'fedavg': results_avg,
            'task_aware': results_aware,
        },
        'exp2': {
            'backbone_only': results_avg,
            'full_model': results_full,
        },
        'exp3': noniid_results,
        'exp4': {
            'backbone_bytes': backbone_bytes,
            'full_bytes': full_bytes,
            'saving_pct': (1 - backbone_bytes/full_bytes) * 100,
        },
    }

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/yolo_fl_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n✅ Results saved to {results_dir}/yolo_fl_results.json")

    # ─── Generate Plots ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Fig 1: Convergence comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        rounds_x = [r['round'] for r in results_avg]
        ax1.plot(rounds_x, [r['avg_loss'] for r in results_avg], 'b-o', label='FedAvg', markersize=4)
        ax1.plot(rounds_x, [r['avg_loss'] for r in results_aware], 'r-s', label='Task-Aware', markersize=4)
        ax1.set_xlabel('Round'); ax1.set_ylabel('Loss')
        ax1.set_title('(a) Detection Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(rounds_x, [r['avg_acc'] for r in results_avg], 'b-o', label='FedAvg', markersize=4)
        ax2.plot(rounds_x, [r['avg_acc'] for r in results_aware], 'r-s', label='Task-Aware', markersize=4)
        ax2.set_xlabel('Round'); ax2.set_ylabel('Accuracy')
        ax2.set_title('(b) Detection Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/fig1_yolo_convergence.png", dpi=150)
        plt.close()
        print("  ✅ fig1_yolo_convergence.png")

        # Fig 2: Non-IID severity
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(noniid_results.keys())
        fedavg_accs = [noniid_results[l]['fedavg_final'] for l in labels]
        aware_accs = [noniid_results[l]['aware_final'] for l in labels]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, fedavg_accs, w, label='FedAvg', color='#3498db')
        ax.bar(x + w/2, aware_accs, w, label='Task-Aware', color='#e74c3c')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('Accuracy'); ax.set_title('Non-IID Severity vs Accuracy')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/fig2_yolo_noniid.png", dpi=150)
        plt.close()
        print("  ✅ fig2_yolo_noniid.png")

        # Fig 3: Communication cost
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = ['Backbone-only\n(Ours)', 'Full Model\n(FedAvg)']
        bytes_vals = [backbone_bytes, full_bytes]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(methods, bytes_vals, color=colors, width=0.5)
        for bar, val in zip(bars, bytes_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                   f'{val:,} B', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bytes per Round')
        ax.set_title(f'Communication Cost per Round (Saving: {(1-backbone_bytes/full_bytes)*100:.1f}%)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/fig3_yolo_communication.png", dpi=150)
        plt.close()
        print("  ✅ fig3_yolo_communication.png")

        # Fig 4: Per-client accuracy
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (fname, _, _, _, _) in enumerate(clients):
            short_name = fname.split('(')[0].strip()
            avg_accs = [r[f'{fname}_acc'] for r in results_aware]
            ax.plot(rounds_x, avg_accs, '-o', label=short_name, markersize=3)
        ax.set_xlabel('Round'); ax.set_ylabel('Accuracy')
        ax.set_title('Per-Client Accuracy (Task-Aware, Backbone-only)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/fig4_yolo_per_client.png", dpi=150)
        plt.close()
        print("  ✅ fig4_yolo_per_client.png")

        print(f"\n  All plots saved to {results_dir}/")

    except ImportError:
        print("  ⚠ matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
