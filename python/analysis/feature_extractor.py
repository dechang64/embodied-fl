# ── python/analysis/feature_extractor.py ──
"""
DINOv2 Scene Feature Extractor for Embodied Intelligence
==========================================================
Self-supervised visual features for robot task understanding.

Replaces the original 32-dim hand-crafted metadata embedding
with 768-dim DINOv2 features from scene images.

Use cases:
- Task similarity: embed workspace images → find similar tasks
- Scene understanding: what does the robot's environment look like?
- Change detection: compare workspace states over time
- Zero-shot transfer: no labels needed for new robot deployments

Bridge to Rust:
  Python extracts DINOv2 features → sends to Rust server via gRPC
  → Rust stores in HNSW index for task matching
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from typing import Optional
from torchvision import transforms


class DINOv2SceneExtractor(nn.Module):
    """DINOv2-based scene feature extractor for robot environments."""

    MODEL_DIMS = {
        "vits14": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    def __init__(self, model_name: str = "facebook/dinov2-base", device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.variant = model_name.split("/")[-1] if "/" in model_name else model_name
        self.dim = self.MODEL_DIMS.get(self.variant, 768)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        return outputs.last_hidden_state[:, 0]  # CLS token [B, dim]

    def extract(self, image_path: str) -> np.ndarray:
        """Extract 768-dim feature from a single image."""
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        feat = self.forward(x)
        return feat.squeeze(0).cpu().numpy()

    def extract_batch(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Extract features for multiple images."""
        features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch.append(self.transform(img))
            x = torch.stack(batch).to(self.device)
            feats = self.forward(x)
            features.append(feats.cpu().numpy())
        return np.concatenate(features, axis=0)

    def extract_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """Extract feature from numpy array (H, W, 3) uint8."""
        img = Image.fromarray(image_array)
        x = self.transform(img).unsqueeze(0).to(self.device)
        feat = self.forward(x)
        return feat.squeeze(0).cpu().numpy()


class MetadataFallbackExtractor:
    """Fallback: 32-dim metadata embedding (original embodied-fl approach).

    Used when DINOv2 is not available or for non-visual tasks.
    """

    TASK_TYPES = [
        "grasping", "navigation", "inspection", "assembly",
        "manipulation", "welding", "custom",
    ]
    DOMAINS = [
        "electronics", "automotive", "consumer_3c", "food",
        "pharma", "logistics", "other",
    ]
    SENSORS = [
        "rgb", "depth", "force", "imu", "tactile", "thermal", "other",
    ]

    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed(self, task_type: str = "", domain: str = "",
              sensor: str = "", data_scale: float = 0.0,
              complexity: str = "medium", realtime: str = "low") -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)

        # Task type one-hot
        idx = self._one_hot_index(task_type, self.TASK_TYPES)
        if idx < self.dim: vec[idx] = 1.0

        # Domain one-hot
        idx = self._one_hot_index(domain, self.DOMAINS)
        if 7 + idx < self.dim: vec[7 + idx] = 1.0

        # Sensor one-hot
        idx = self._one_hot_index(sensor, self.SENSORS)
        if 14 + idx < self.dim: vec[14 + idx] = 1.0

        # Data scale
        if 21 < self.dim: vec[21] = min(data_scale, 1.0)

        # Complexity
        comp_idx = {"simple": 0, "medium": 1, "complex": 2}.get(complexity, 1)
        if 24 + comp_idx < self.dim: vec[24 + comp_idx] = 1.0

        # Realtime
        rt_idx = {"low": 0, "medium": 1, "high": 2}.get(realtime, 0)
        if 27 + rt_idx < self.dim: vec[27 + rt_idx] = 1.0

        return vec

    def _one_hot_index(self, value: str, categories: list[str]) -> int:
        v = value.lower()
        for i, cat in enumerate(categories):
            if cat in v or v in cat:
                return i
        return len(categories) - 1  # "other"


def get_extractor(mode: str = "dinov2", **kwargs):
    """Factory: get feature extractor by mode.

    Args:
        mode: "dinov2", "metadata", or "hybrid"
    """
    if mode == "dinov2":
        return DINOv2SceneExtractor(**kwargs)
    elif mode == "metadata":
        return MetadataFallbackExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
