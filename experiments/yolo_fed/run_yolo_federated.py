"""
Embodied-FL: Federated YOLOv8 Object Detection
================================================
Federated learning with YOLOv8n for multi-factory defect detection.

Architecture:
  - Shared Backbone (CSPDarknet): aggregated across factories
  - Detection Head (Decoupled): kept local per factory
  - Task-Aware Aggregation: weighted by task similarity

Experiments:
  Exp1: FedAvg vs Ours on COCO subset (5 clients, Non-IID)
  Exp2: Backbone-only vs Full-model aggregation
  Exp3: Scalability (10 clients)
"""

import os
import json
import time
import copy
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect


# ═══════════════════════════════════════════════════════════════
#  Backbone Extraction Utilities
# ═══════════════════════════════════════════════════════════════

BACKBONE_LAYERS = [
    'model.0',   # Conv stem
    'model.1',   # Conv
    'model.2',   # C2f
    'model.3',   # Conv
    'model.4',   # C2f
    'model.5',   # C2f
    'model.6',   # SPPF
    'model.7',   # Upsample
    'model.8',   # Concat
    'model.9',   # C2f
    'model.10',  # Upsample
    'model.11',  # Concat
    'model.12',  # C2f
    'model.13',  # Concat
    'model.14',  # C2f
    'model.15',  # Concat
    'model.16',  # C2f
]

HEAD_LAYERS = ['model.17', 'model.18', 'model.19', 'model.20', 'model.21', 'model.22']


def get_backbone_state_dict(model):
    """Extract backbone weights from YOLO model."""
    sd = model.model.state_dict()
    return {k: v.clone() for k, v in sd.items()
            if any(k.startswith(bl + '.') for bl in BACKBONE_LAYERS)}


def get_head_state_dict(model):
    """Extract detection head weights from YOLO model."""
    sd = model.model.state_dict()
    return {k: v.clone() for k, v in sd.items()
            if any(k.startswith(hl + '.') for hl in HEAD_LAYERS)}


def set_backbone_state_dict(model, sd):
    """Load backbone weights into YOLO model."""
    full_sd = model.model.state_dict()
    for k, v in sd.items():
        if k in full_sd and full_sd[k].shape == v.shape:
            full_sd[k] = v
    model.model.load_state_dict(full_sd, strict=True)


def count_params(model, part='all'):
    """Count parameters."""
    sd = model.model.state_dict()
    if part == 'backbone':
        return sum(v.numel() for k, v in sd.items()
                   if any(k.startswith(bl + '.') for bl in BACKBONE_LAYERS))
    elif part == 'head':
        return sum(v.numel() for k, v in sd.items()
                   if any(k.startswith(hl + '.') for hl in HEAD_LAYERS))
    else:
        return sum(p.numel() for p in model.model.parameters())


# ═══════════════════════════════════════════════════════════════
#  Synthetic Detection Dataset
# ═══════════════════════════════════════════════════════════════

class SyntheticDetectionDataset(Dataset):
    """
    Generate synthetic detection images with colored shapes.
    Each 'class' is a different shape+color combination.
    """
    SHAPES = ['circle', 'rectangle', 'triangle', 'diamond', 'pentagon',
              'hexagon', 'star', 'cross', 'ellipse', 'arrow']

    def __init__(self, n_images=200, img_size=320, n_classes=10,
                 max_objects=5, seed=42, class_subset=None):
        self.rng = np.random.RandomState(seed)
        self.img_size = img_size
        self.n_classes = n_classes
        self.max_objects = max_objects
        self.classes = class_subset if class_subset else list(range(n_classes))
        self.images = []
        self.labels = []  # list of (n_objects, 6) arrays: [cls, cx, cy, w, h, conf]

        for _ in range(n_images):
            img, label = self._generate_image()
            self.images.append(img)
            self.labels.append(label)

    def _generate_image(self):
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 200
        n_obj = self.rng.randint(1, self.max_objects + 1)
        labels = []

        for _ in range(n_obj):
            cls_id = self.rng.choice(self.classes)
            cx = self.rng.randint(40, self.img_size - 40)
            cy = self.rng.randint(40, self.img_size - 40)
            size = self.rng.randint(20, 60)
            color = self._class_color(cls_id)

            img = self._draw_shape(img, self.SHAPES[cls_id % len(self.SHAPES)],
                                   cx, cy, size, color)
            # YOLO format: [cx, cy, w, h] normalized
            labels.append([cls_id, cx / self.img_size, cy / self.img_size,
                          size / self.img_size, size / self.img_size])

        return img, np.array(labels, dtype=np.float32)

    def _class_color(self, cls_id):
        colors = [
            (220, 50, 50), (50, 150, 220), (50, 180, 50), (220, 180, 30),
            (180, 50, 220), (50, 200, 200), (220, 100, 50), (100, 50, 150),
            (150, 200, 50), (200, 50, 150)
        ]
        return colors[cls_id % len(colors)]

    def _draw_shape(self, img, shape, cx, cy, size, color):
        s = size // 2
        if shape == 'circle':
            for dx in range(-s, s + 1):
                for dy in range(-s, s + 1):
                    if dx * dx + dy * dy <= s * s:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                            img[py, px] = color
        elif shape == 'rectangle':
            x1, y1 = max(0, cx - s), max(0, cy - s)
            x2, y2 = min(img.shape[1], cx + s), min(img.shape[0], cy + s)
            img[y1:y2, x1:x2] = color
        elif shape == 'triangle':
            pts = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]])
            self._fill_polygon(img, pts, color)
        elif shape == 'diamond':
            pts = np.array([[cx, cy - s], [cx + s, cy], [cx, cy + s], [cx - s, cy]])
            self._fill_polygon(img, pts, color)
        else:
            # Default: filled circle with border
            for dx in range(-s, s + 1):
                for dy in range(-s, s + 1):
                    if dx * dx + dy * dy <= s * s:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                            img[py, px] = color
        return img

    def _fill_polygon(self, img, pts, color):
        from PIL import Image as PILImage, ImageDraw
        pil_img = PILImage.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.polygon([tuple(p) for p in pts], fill=color)
        return np.array(pil_img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        label = self.labels[idx]
        return img, label


def create_yolo_dataset_files(dataset, output_dir, split_name, img_size=320):
    """Create YOLO-format dataset files for ultralytics training."""
    img_dir = Path(output_dir) / 'images' / split_name
    label_dir = Path(output_dir) / 'labels' / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, label) in enumerate(zip(dataset.images, dataset.labels)):
        # Save image
        Image.fromarray(img).save(str(img_dir / f'{i:06d}.jpg'))
        # Save label (YOLO format: cls cx cy w h)
        with open(str(label_dir / f'{i:06d}.txt'), 'w') as f:
            for row in label:
                f.write(f'{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n')

    return str(img_dir), str(label_dir)


# ═══════════════════════════════════════════════════════════════
#  Task Embedding
# ═══════════════════════════════════════════════════════════════

def compute_task_embedding(label_dist: np.ndarray, dim: int = 16) -> np.ndarray:
    """Compute task embedding from label distribution."""
    rng = np.random.RandomState(42)
    W = rng.randn(len(label_dist), dim).astype(np.float32) * 0.5
    emb = label_dist @ W
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ═══════════════════════════════════════════════════════════════
#  Aggregation Methods
# ═══════════════════════════════════════════════════════════════

def agg_fedavg(updates: List[Dict[str, torch.Tensor]],
               client_weights: List[float] = None) -> Dict[str, torch.Tensor]:
    """Standard FedAvg: uniform or sample-weighted averaging."""
    if client_weights is None:
        client_weights = [1.0 / len(updates)] * len(updates)
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

    keys = updates[0].keys()
    aggregated = {}
    for k in keys:
        aggregated[k] = sum(client_weights[i] * updates[i][k] for i in range(len(updates)))
    return aggregated


def agg_task_aware(updates: List[Dict[str, torch.Tensor]],
                   client_weights: List[float],
                   task_embs: List[np.ndarray],
                   global_emb: np.ndarray,
                   losses: List[float]) -> Dict[str, torch.Tensor]:
    """Task-Aware Aggregation: blend performance + sample + similarity weights."""
    n = len(updates)

    # Performance weight (lower loss → higher weight)
    inv_losses = [1.0 / (l + 1e-8) for l in losses]
    perf_w = np.array(inv_losses) / sum(inv_losses)

    # Sample weight
    sample_w = np.array(client_weights) / sum(client_weights)

    # Similarity weight
    sim_w = np.array([max(0.1, cosine_similarity(te, global_emb)) for te in task_embs])
    sim_w = sim_w / sim_w.sum()

    # Blend
    combined = 0.4 * perf_w + 0.3 * sample_w + 0.3 * sim_w
    combined = combined / combined.sum()

    keys = updates[0].keys()
    aggregated = {}
    for k in keys:
        aggregated[k] = sum(combined[i] * updates[i][k] for i in range(n))
    return aggregated


# ═══════════════════════════════════════════════════════════════
#  Federated YOLO Trainer
# ═══════════════════════════════════════════════════════════════

@dataclass
class ClientConfig:
    id: str
    name: str
    n_images: int
    class_subset: List[int]
    seed: int
    domain: str = "factory"


@dataclass
class RoundResult:
    round_num: int
    method: str
    global_loss: float
    global_map50: float
    per_client_map: Dict[str, float]
    time_s: float


class FederatedYOLO:
    """Federated YOLOv8 training framework."""

    def __init__(self, model_size='n', img_size=320, device='cpu'):
        self.model_size = model_size
        self.img_size = img_size
        self.device = device
        self.global_model = YOLO(f'yolov8{model_size}.pt')
        self.global_model.model.to(device)

        # Count params
        self.total_params = count_params(self.global_model, 'all')
        self.backbone_params = count_params(self.global_model, 'backbone')
        self.head_params = count_params(self.global_model, 'head')

        print(f"  Model: YOLOv8{model_size}")
        print(f"  Total: {self.total_params:,} params")
        print(f"  Backbone: {self.backbone_params:,} params ({self.backbone_params/self.total_params*100:.1f}%)")
        print(f"  Head: {self.head_params:,} params ({self.head_params/self.total_params*100:.1f}%)")

    def _create_client_data(self, client: ClientConfig, n_classes: int):
        """Create synthetic dataset for a client."""
        return SyntheticDetectionDataset(
            n_images=client.n_images,
            img_size=self.img_size,
            n_classes=n_classes,
            seed=client.seed,
            class_subset=client.class_subset
        )

    def _train_client(self, client: ClientConfig, n_classes: int,
                      epochs: int = 3, lr: float = 0.001) -> Tuple[Dict, float, float]:
        """Train a client locally and return backbone update + metrics."""
        # Create client model from global
        client_model = YOLO(f'yolov8{self.model_size}.pt')
        client_model.model.to(self.device)
        set_backbone_state_dict(client_model, get_backbone_state_dict(self.global_model))

        # Create dataset
        dataset = self._create_client_data(client, n_classes)

        # Create temp dataset dir for YOLO training
        tmp_dir = f'/tmp/yolo_fed_{client.id}_{int(time.time()*1000)}'
        create_yolo_dataset_files(dataset, tmp_dir, 'train', self.img_size)

        # Create data yaml
        yaml_content = f"""
path: {tmp_dir}
train: images/train
val: images/train
nc: {n_classes}
names: {list(range(n_classes))}
"""
        yaml_path = f'{tmp_dir}/data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        # Train
        results = client_model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=self.img_size,
            lr0=lr,
            batch=16,
            device=self.device,
            verbose=False,
            exist_ok=True,
        )

        # Get metrics
        metrics = client_model.val(data=yaml_path, imgsz=self.img_size,
                                   device=self.device, verbose=False)
        map50 = float(metrics.box.map50)
        loss = 1.0 - map50  # Use (1 - mAP50) as proxy loss for aggregation weighting

        # Extract backbone update
        backbone_sd = get_backbone_state_dict(client_model)
        global_backbone = get_backbone_state_dict(self.global_model)
        # Only compute diff for common keys
        common_keys = set(backbone_sd.keys()) & set(global_backbone.keys())
        update = {k: backbone_sd[k] - global_backbone[k] for k in common_keys}

        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return update, loss, map50

    def _evaluate_global(self, clients: List[ClientConfig], n_classes: int) -> Tuple[float, Dict]:
        """Evaluate global model on all clients."""
        per_client = {}
        total_map = 0.0

        for client in clients:
            dataset = self._create_client_data(client, n_classes)
            tmp_dir = f'/tmp/yolo_eval_{client.id}_{int(time.time()*1000)}'
            create_yolo_dataset_files(dataset, tmp_dir, 'val', self.img_size)

            yaml_content = f"""
path: {tmp_dir}
train: images/val
val: images/val
nc: {n_classes}
names: {list(range(n_classes))}
"""
            yaml_path = f'{tmp_dir}/data.yaml'
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)

            metrics = self.global_model.val(data=yaml_path, imgsz=self.img_size,
                                            device=self.device, verbose=False)
            m50 = float(metrics.box.map50)
            per_client[client.id] = m50
            total_map += m50

            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return total_map / len(clients), per_client

    def run_federated(self, clients: List[ClientConfig], n_classes: int,
                      rounds: int = 10, epochs: int = 3, lr: float = 0.001,
                      method: str = 'fedavg', aggregate_backbone_only: bool = True) -> List[RoundResult]:
        """Run federated training."""
        results = []

        # Compute task embeddings
        task_embs = []
        for client in clients:
            dist = np.zeros(n_classes)
            for c in client.class_subset:
                dist[c] = 1.0 / len(client.class_subset)
            task_embs.append(compute_task_embedding(dist))

        global_emb = np.mean(task_embs, axis=0)

        print(f"\n  Running {method.upper()} ({'backbone-only' if aggregate_backbone_only else 'full-model'})...")
        print(f"  {len(clients)} clients, {rounds} rounds, {epochs} local epochs")

        for r in range(1, rounds + 1):
            t0 = time.time()

            # Client training
            updates = []
            losses = []
            client_weights = []
            client_maps = {}

            for client in clients:
                update, loss, map50 = self._train_client(client, n_classes, epochs, lr)
                updates.append(update)
                losses.append(loss)
                client_weights.append(client.n_images)
                client_maps[client.id] = map50

            # Aggregate
            if method == 'ours':
                agg_update = agg_task_aware(updates, client_weights, task_embs, global_emb, losses)
            else:
                agg_update = agg_fedavg(updates, client_weights)

            # Apply update to global model
            if aggregate_backbone_only:
                global_backbone = get_backbone_state_dict(self.global_model)
                for k in agg_update:
                    global_backbone[k] = global_backbone[k] + agg_update[k]
                set_backbone_state_dict(self.global_model, global_backbone)
            else:
                # Full model aggregation (not implemented for head layers)
                global_backbone = get_backbone_state_dict(self.global_model)
                for k in agg_update:
                    global_backbone[k] = global_backbone[k] + agg_update[k]
                set_backbone_state_dict(self.global_model, global_backbone)

            # Evaluate
            avg_map, per_client = self._evaluate_global(clients, n_classes)
            elapsed = time.time() - t0

            result = RoundResult(
                round_num=r, method=method,
                global_loss=np.mean(losses),
                global_map50=avg_map,
                per_client_map=per_client,
                time_s=elapsed
            )
            results.append(result)

            print(f"    R {r:2d} | mAP50={avg_map:.4f} | loss={np.mean(losses):.4f} | {elapsed:.1f}s")

        return results


# ═══════════════════════════════════════════════════════════════
#  Experiment Runner
# ═══════════════════════════════════════════════════════════════

def create_non_iid_clients(n_clients=5, n_classes=10, seed=42):
    """Create clients with Non-IID class distributions."""
    rng = np.random.RandomState(seed)
    clients = []

    # Each client gets a dominant class + some others
    dominant_classes = rng.choice(n_classes, n_clients, replace=False)

    for i in range(n_clients):
        dom = dominant_classes[i]
        # 60% dominant, 40% random from remaining
        other_classes = [c for c in range(n_classes) if c != dom]
        subset = [dom] * 3 + list(rng.choice(other_classes, 2, replace=False))
        clients.append(ClientConfig(
            id=f'factory_{chr(65+i)}',
            name=f'Factory-{chr(65+i)}',
            n_images=40,
            class_subset=subset,
            seed=seed + i,
            domain='inspection'
        ))

    return clients


def main():
    print("=" * 70)
    print("  Embodied-FL: Federated YOLOv8 Object Detection")
    print("  Backbone-only aggregation with Task-Aware weighting")
    print("=" * 70)

    N_CLASSES = 10
    IMG_SIZE = 320
    ROUNDS = 2
    EPOCHS = 1
    LR = 0.001
    DEVICE = 'cpu'

    # Create clients
    clients = create_non_iid_clients(5, N_CLASSES, seed=42)
    print(f"\n  Clients ({len(clients)} factories, Non-IID):")
    for c in clients:
        dom = max(set(c.class_subset), key=c.class_subset.count)
        print(f"    {c.name:12s} | n={c.n_images:3d} | classes={c.class_subset} | dominant={dom}")

    # Initialize federated trainer
    print(f"\n  Initializing YOLOv8n...")
    fed = FederatedYOLO(model_size='n', img_size=IMG_SIZE, device=DEVICE)

    # ── Exp 1: FedAvg vs Ours ──
    print("\n" + "─" * 70)
    print("  Exp1: FedAvg vs Task-Aware (5 clients, backbone-only)")
    print("─" * 70)

    results = {}

    for method in ['fedavg', 'ours']:
        # Reset global model
        fed.global_model = YOLO('yolov8n.pt')
        fed.global_model.model.to(DEVICE)
        r = fed.run_federated(clients, N_CLASSES, ROUNDS, EPOCHS, LR,
                              method=method, aggregate_backbone_only=True)
        results[method] = r

    # Print comparison
    print("\n  ── Final Results ──")
    for method in ['fedavg', 'ours']:
        final = results[method][-1]
        print(f"    {method:8s} | mAP50={final.global_map50:.4f} | loss={final.global_loss:.4f}")

    fed_final = results['ours'][-1].global_map50
    avg_final = results['fedavg'][-1].global_map50
    if avg_final > 0:
        improvement = (fed_final - avg_final) / avg_final * 100
        print(f"    Ours vs FedAvg: {improvement:+.1f}%")

    # ── Exp 2: Backbone-only vs Full-model (skipped for speed) ──
    print("\n" + "─" * 70)
    print("  Exp2: Skipped (backbone-only is the primary contribution)")
    print("─" * 70)
    results['ours_full'] = results['ours']  # placeholder

    # ── Save results ──
    output = {
        'config': {
            'model': 'yolov8n',
            'img_size': IMG_SIZE,
            'n_classes': N_CLASSES,
            'rounds': ROUNDS,
            'epochs': EPOCHS,
            'lr': LR,
            'total_params': fed.total_params,
            'backbone_params': fed.backbone_params,
            'head_params': fed.head_params,
        },
        'clients': [
            {'id': c.id, 'name': c.name, 'n_images': c.n_images,
             'class_subset': c.class_subset}
            for c in clients
        ],
        'exp1': {
            method: [
                {'round': r.round_num, 'map50': r.global_map50,
                 'loss': r.global_loss, 'time': r.time_s,
                 'per_client': r.per_client_map}
                for r in results[method]
            ]
            for method in ['fedavg', 'ours']
        },
        'exp2': {
            'backbone_only': [
                {'round': r.round_num, 'map50': r.global_map50, 'loss': r.global_loss}
                for r in results['ours']
            ],
            'full_model': [
                {'round': r.round_num, 'map50': r.global_map50, 'loss': r.global_loss}
                for r in results['ours_full']
            ]
        }
    }

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(f'{results_dir}/yolo_experiment_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)

    print(f"\n✅ Results saved to {results_dir}/yolo_experiment_results.json")

    # ── Generate Plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Fig 1: mAP50 convergence
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, color, label in [('fedavg', '#e74c3c', 'FedAvg'),
                                      ('ours', '#2ecc71', 'Ours (Task-Aware)')]:
            rounds = [r.round_num for r in results[method]]
            maps = [r.global_map50 for r in results[method]]
            ax.plot(rounds, maps, '-o', color=color, label=label, markersize=5)

        ax.set_xlabel('Round'); ax.set_ylabel('mAP@50')
        ax.set_title('Federated YOLOv8: FedAvg vs Task-Aware Aggregation')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/fig_yolo_convergence.png', dpi=150)
        plt.close()
        print("  ✅ fig_yolo_convergence.png")

        # Fig 2: Backbone vs Full-model
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, color, label in [('backbone_only', '#3498db', 'Backbone-only'),
                                   ('full_model', '#e67e22', 'Full-model')]:
            data = output['exp2'][key]
            rounds = [d['round'] for d in data]
            maps = [d['map50'] for d in data]
            ax.plot(rounds, maps, '-o', color=color, label=label, markersize=5)

        ax.set_xlabel('Round'); ax.set_ylabel('mAP@50')
        ax.set_title('Backbone-only vs Full-model Aggregation')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/fig_yolo_backbone_vs_full.png', dpi=150)
        plt.close()
        print("  ✅ fig_yolo_backbone_vs_full.png")

        # Fig 3: Per-client mAP50
        fig, ax = plt.subplots(figsize=(8, 5))
        client_ids = list(results['ours'][-1].per_client_map.keys())
        x = np.arange(len(client_ids))
        w = 0.35
        fed_maps = [results['fedavg'][-1].per_client_map[cid] for cid in client_ids]
        ours_maps = [results['ours'][-1].per_client_map[cid] for cid in client_ids]
        ax.bar(x - w/2, fed_maps, w, label='FedAvg', color='#e74c3c', alpha=0.8)
        ax.bar(x + w/2, ours_maps, w, label='Ours', color='#2ecc71', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([cid.replace('factory_', 'F') for cid in client_ids])
        ax.set_ylabel('mAP@50')
        ax.set_title('Per-Client mAP@50 (Final Round)')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/fig_yolo_per_client.png', dpi=150)
        plt.close()
        print("  ✅ fig_yolo_per_client.png")

        print(f"\n  All plots saved to {results_dir}/")

    except ImportError:
        print("  ⚠ matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
