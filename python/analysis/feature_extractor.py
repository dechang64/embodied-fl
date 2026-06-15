# ── python/analysis/feature_extractor.py ──
"""
DINOv2 Scene Feature Extractor for Embodied Intelligence (twc-core wrapper)
============================================================================
Delegates to twc_core.features.DINOv2Extractor.
DINOv2SceneExtractor adds batch extraction and array extraction.
MetadataFallbackExtractor is domain-specific (kept as-is).
"""

import numpy as np
from PIL import Image
from twc_core.features import DINOv2Extractor

__all__ = ["DINOv2SceneExtractor", "MetadataFallbackExtractor", "get_extractor"]


class DINOv2SceneExtractor(DINOv2Extractor):
    """DINOv2 scene feature extractor for robot environments.

    Extends twc_core.DINOv2Extractor with batch and array extraction.
    """

    def extract_batch(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """Extract features for multiple images."""
        import torch
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

        idx = self._one_hot_index(task_type, self.TASK_TYPES)
        if idx < self.dim: vec[idx] = 1.0

        idx = self._one_hot_index(domain, self.DOMAINS)
        if 7 + idx < self.dim: vec[7 + idx] = 1.0

        idx = self._one_hot_index(sensor, self.SENSORS)
        if 14 + idx < self.dim: vec[14 + idx] = 1.0

        if 21 < self.dim: vec[21] = min(data_scale, 1.0)

        comp_idx = {"simple": 0, "medium": 1, "complex": 2}.get(complexity, 1)
        if 24 + comp_idx < self.dim: vec[24 + comp_idx] = 1.0

        rt_idx = {"low": 0, "medium": 1, "high": 2}.get(realtime, 0)
        if 27 + rt_idx < self.dim: vec[27 + rt_idx] = 1.0

        return vec

    def _one_hot_index(self, value: str, categories: list) -> int:
        v = value.lower()
        for i, cat in enumerate(categories):
            if cat in v or v in cat:
                return i
        return len(categories) - 1


def get_extractor(mode: str = "dinov2", **kwargs):
    """Factory: get feature extractor by mode."""
    if mode == "dinov2":
        return DINOv2SceneExtractor(**kwargs)
    elif mode == "metadata":
        return MetadataFallbackExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
