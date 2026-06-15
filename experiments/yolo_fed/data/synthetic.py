"""
Synthetic detection dataset for federated experiments.
Generates colored rectangles on Gaussian noise with class labels.

For paper experiments, replace with:
  - COCO subset (use torchvision.datasets.CocoDetection)
  - VOC2012 (use torchvision.datasets.VOCDetection)
  - Custom industrial defect dataset
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDetectionDataset(Dataset):
    """
    Synthetic object detection dataset.
    
    Each image: Gaussian noise background + colored rectangles.
    Each rectangle: random position, size, and class from class_subset.
    
    Args:
        n_images: number of images
        img_size: image resolution (square)
        n_classes: total number of classes in the federation
        class_subset: which classes this client sees (Non-IID)
        seed: random seed for reproducibility
        max_obj: max objects per image (padded with background)
        difficulty: 0=easy (large, distinct), 1=hard (small, overlapping)
    """
    def __init__(self, n_images=200, img_size=64, n_classes=10,
                 class_subset=None, seed=0, max_obj=5, difficulty=0.0):
        self.rng = np.random.RandomState(seed)
        self.img_size = img_size
        self.n_classes = n_classes
        self.max_obj = max_obj
        self.classes = class_subset or list(range(n_classes))
        self.difficulty = difficulty

        # Pre-generate all data
        self.images = []
        self.targets = []

        for _ in range(n_images):
            img = self._generate_image()
            self.images.append(img)
            targets = self._generate_targets()  # draws on self.images[-1]
            self.targets.append(targets)

    def _generate_image(self):
        """Generate background: Gaussian noise."""
        return self.rng.randn(3, self.img_size, self.img_size).astype(np.float32) * 0.2

    def _generate_targets(self):
        """Generate random object targets."""
        sz = self.img_size
        n_obj = self.rng.randint(1, self.max_obj + 1)
        objs = []

        for _ in range(n_obj):
            cls_id = int(self.rng.choice(self.classes))

            # Size depends on difficulty
            min_size = 0.1 + 0.15 * (1 - self.difficulty)
            max_size = 0.3 + 0.2 * (1 - self.difficulty)
            w = self.rng.uniform(min_size, max_size)
            h = self.rng.uniform(min_size, max_size)
            cx = self.rng.uniform(w / 2, 1 - w / 2)
            cy = self.rng.uniform(h / 2, 1 - h / 2)
            objs.append([cx, cy, w, h, cls_id])

            # Draw rectangle on image
            x1 = max(0, int((cx - w / 2) * sz))
            y1 = max(0, int((cy - h / 2) * sz))
            x2 = min(sz, int((cx + w / 2) * sz))
            y2 = min(sz, int((cy + h / 2) * sz))
            if x2 > x1 and y2 > y1:
                color = (cls_id + 1) / self.n_classes
                noise = self.rng.randn(3, y2 - y1, x2 - x1) * 0.1
                self.images[-1][:, y1:y2, x1:x2] = color + noise

        # Pad to max_obj with background
        while len(objs) < self.max_obj:
            objs.append([0, 0, 0, 0, -1])  # -1 = background

        return objs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return img, target


def create_non_iid_split(n_clients, n_classes, seed=42):
    """
    Create Non-IID class splits for federated clients.
    
    Each client gets 3 dominant classes + 2 minority classes.
    This simulates real factory scenarios where each factory
    sees different types of defects/objects.
    
    Args:
        n_clients: number of federated clients
        n_classes: total number of classes
        seed: random seed
    
    Returns:
        List of class subsets (one per client)
    """
    rng = np.random.RandomState(seed)
    splits = []

    # Assign dominant classes (evenly distributed)
    dominant_per_client = max(1, n_classes // n_clients)
    for i in range(n_clients):
        start = (i * dominant_per_client) % n_classes
        dom_classes = [(start + j) % n_classes for j in range(dominant_per_client)]
        # Add 2 random minority classes
        others = [c for c in range(n_classes) if c not in dom_classes]
        if len(others) >= 2:
            minority = list(rng.choice(others, 2, replace=False))
        else:
            minority = others
        splits.append(dom_classes + minority)

    return splits
