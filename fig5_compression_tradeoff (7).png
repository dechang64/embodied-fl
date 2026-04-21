"""
Federated learning utilities: aggregation, metrics, task embeddings.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════
#  Task Embeddings
# ═══════════════════════════════════════════════════════════════

def compute_task_embedding(class_distribution: np.ndarray, dim: int = 256) -> np.ndarray:
    """
    Compute task embedding from class distribution.
    
    Uses Fourier feature encoding to create a fixed-dimensional
    representation of the class distribution, enabling similarity
    computation between heterogeneous tasks.
    
    Args:
        class_distribution: [n_classes] array of class frequencies
        dim: embedding dimension
    
    Returns:
        [dim] normalized embedding vector
    """
    # Fourier features at random frequencies
    rng = np.random.RandomState(42)
    freqs = rng.randn(dim // 2) * 2 * np.pi
    embeddings = np.sin(freqs * class_distribution[0]) * np.cos(freqs * class_distribution[1])
    
    # Pad if needed
    if len(embeddings) < dim:
        embeddings = np.pad(embeddings, (0, dim - len(embeddings)))
    
    # Normalize
    norm = np.linalg.norm(embeddings)
    if norm > 0:
        embeddings = embeddings / norm
    
    return embeddings.astype(np.float32)


def compute_task_embeddings_for_clients(class_subsets: List[List[int]],
                                         n_classes: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Compute task embeddings for all clients and the global average.
    """
    embeddings = []
    for subset in class_subsets:
        dist = np.zeros(n_classes)
        for c in subset:
            dist[c] += 1.0
        dist = dist / max(dist.sum(), 1)
        embeddings.append(compute_task_embedding(dist))
    
    global_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(global_emb)
    if norm > 0:
        global_emb = global_emb / norm
    
    return embeddings, global_emb


# ═══════════════════════════════════════════════════════════════
#  Aggregation Methods
# ═══════════════════════════════════════════════════════════════

def aggregate_fedavg(updates: List[Dict[str, torch.Tensor]],
                     weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Standard FedAvg: weighted average of model updates.
    
    w_global = Σ (n_k / N) * w_k
    
    Args:
        updates: list of {param_name: gradient_update} per client
        weights: sample counts per client
    
    Returns:
        Aggregated update dict
    """
    total = sum(weights)
    agg = {}
    for key in updates[0]:
        agg[key] = sum(u[key] * w / total for u, w in zip(updates, weights))
    return agg


def aggregate_task_aware(updates: List[Dict[str, torch.Tensor]],
                         weights: List[float],
                         task_embs: List[np.ndarray],
                         global_emb: np.ndarray,
                         losses: List[float],
                         alpha_sim: float = 0.4,
                         alpha_perf: float = 0.3,
                         alpha_size: float = 0.3) -> Dict[str, torch.Tensor]:
    """
    Task-Aware Aggregation (our method).
    
    Combines three factors:
      1. Task similarity (cosine similarity of task embeddings)
      2. Performance (inverse loss — better clients get more weight)
      3. Sample size (proportional to data volume)
    
    w_k = α_sim * sim(k, global) + α_perf * (1/loss_k) + α_size * (n_k/N)
    
    Args:
        updates: backbone parameter updates per client
        weights: sample counts per client
        task_embs: task embedding per client
        global_emb: global task embedding (mean of all)
        losses: training loss per client
        alpha_sim/perf/size: weighting factors (sum to 1.0)
    
    Returns:
        Aggregated update dict
    """
    n = len(updates)
    total_samples = sum(weights)
    
    # Factor 1: Task similarity (cosine)
    similarities = []
    for emb in task_embs:
        sim = float(np.dot(emb, global_emb) / max(np.linalg.norm(emb) * np.linalg.norm(global_emb), 1e-8))
        similarities.append(max(sim, 0.01))  # floor to avoid zero weight
    
    # Factor 2: Performance (inverse loss)
    inv_losses = [1.0 / max(l, 0.01) for l in losses]
    
    # Factor 3: Sample size (normalized)
    size_weights = [w / total_samples for w in weights]
    
    # Normalize each factor
    sim_sum = sum(similarities)
    perf_sum = sum(inv_losses)
    norm_sim = [s / sim_sum for s in similarities]
    norm_perf = [p / perf_sum for p in inv_losses]
    
    # Combined weights
    client_weights = []
    for i in range(n):
        w = alpha_sim * norm_sim[i] + alpha_perf * norm_perf[i] + alpha_size * size_weights[i]
        client_weights.append(w)
    
    # Re-normalize
    w_sum = sum(client_weights)
    client_weights = [w / w_sum for w in client_weights]
    
    # Aggregate
    agg = {}
    for key in updates[0]:
        agg[key] = sum(u[key] * cw for u, cw in zip(updates, client_weights))
    
    return agg, client_weights


# ═══════════════════════════════════════════════════════════════
#  Evaluation Metrics
# ═══════════════════════════════════════════════════════════════

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [cx, cy, w, h] format."""
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymin = max(y1_min, y2_min)
    inter_ymax = min(y1_max, y2_max)

    inter = max(inter_xmax - inter_xmin, 0) * max(inter_ymax - inter_ymin, 0)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter

    return inter / max(union, 1e-8)


def compute_ap(pred_box, pred_cls, target, n_classes, iou_threshold=0.5,
               conf_threshold=0.3):
    """
    Compute Average Precision (AP) for one batch.
    
    Uses per-class precision-recall with IoU matching.
    
    Args:
        pred_box: [B, max_obj, 4] predicted boxes
        pred_cls: [B, max_obj, nc] predicted class logits
        target: [B, max_obj, 5] ground truth (cx, cy, w, h, class_id)
        n_classes: number of classes
        iou_threshold: IoU threshold for TP/FP
        conf_threshold: confidence threshold for predictions
    
    Returns:
        Mean AP across all classes
    """
    aps = []

    for c in range(n_classes):
        tp, fp, n_gt = 0, 0, 0
        scores = []

        B = pred_box.size(0)
        for b in range(B):
            # Count GT for this class
            for i in range(target.size(1)):
                if int(target[b, i, 4].item()) == c:
                    n_gt += 1

            # Collect predictions for this class
            for i in range(pred_box.size(1)):
                prob = F.softmax(pred_cls[b, i], dim=0)[c].item()
                if prob > conf_threshold:
                    scores.append((prob, b, i))

        if n_gt == 0:
            continue

        # Sort by confidence (descending)
        scores.sort(reverse=True)
        matched = set()

        for prob, b, i in scores:
            pred = pred_box[b, i].detach().cpu().numpy()
            best_iou, best_gt = 0, -1

            for j in range(target.size(1)):
                if int(target[b, j, 4].item()) != c:
                    continue
                gt = target[b, j, :4].detach().cpu().numpy()
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_gt = iou, j

            if best_iou >= iou_threshold and best_gt not in matched:
                tp += 1
                matched.add(best_gt)
            else:
                fp += 1

        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / max(n_gt, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            aps.append(f1)

    return np.mean(aps) if aps else 0.0


# ═══════════════════════════════════════════════════════════════
#  Parameter Utilities
# ═══════════════════════════════════════════════════════════════

def get_backbone_params(backbone) -> Dict[str, torch.Tensor]:
    """Extract backbone parameters as a dict."""
    return {k: v.clone() for k, v in backbone.state_dict().items()}


def set_backbone_params(backbone, params: Dict[str, torch.Tensor]):
    """Load parameters into backbone."""
    backbone.load_state_dict(params)


def count_parameters(module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def communication_cost(params: Dict[str, torch.Tensor]) -> int:
    """Estimate communication cost in bytes (float32)."""
    return sum(v.numel() * 4 for v in params.values())
