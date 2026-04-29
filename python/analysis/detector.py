# ── python/analysis/detector.py ──
"""
YOLOv11 Robot Scene Detector
=============================
Object detection for embodied intelligence scenarios.

Use cases:
- Factory floor: detect workpieces, tools, safety zones
- Assembly line: detect components, fixtures, human workers
- Warehouse: detect packages, pallets, forklifts
- Quality inspection: detect defects, misalignments

Federated Learning Integration:
- Backbone (feature extractor) aggregated via FedAvg
- Detection head stays local (factory-specific objects)
- Task-Aware weighting based on task similarity
"""

import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class Detection:
    """Single object detection result."""
    bbox: list[float]          # [x1, y1, x2, y2]
    class_name: str
    class_id: int
    confidence: float
    cx: float
    cy: float
    width: float
    height: float
    area: float

    def to_dict(self) -> dict:
        return asdict(self)


class RobotSceneDetector:
    """YOLOv11-based robot scene detector.

    Default classes for factory/warehouse scenarios.
    Custom classes can be configured per deployment.
    """

    DEFAULT_CLASSES = [
        "workpiece", "tool", "fixture", "conveyor",
        "human_worker", "safety_zone", "defect",
        "package", "pallet", "forklift",
    ]

    def __init__(
        self,
        model_size: str = "n",
        classes: Optional[list[str]] = None,
        device: Optional[str] = None,
    ):
        self.model_size = model_size
        self.class_names = classes or self.DEFAULT_CLASSES
        self.device = device
        self.model = None

    def _ensure_model(self):
        if self.model is not None:
            return
        from ultralytics import YOLO
        self.model = YOLO(f"yolo11{self.model_size}.pt")
        # Replace head for custom classes
        if self.class_names != self.DEFAULT_CLASSES:
            self.model.nc = len(self.class_names)
            self.model.names = {i: name for i, name in enumerate(self.class_names)}

    def train_local(self, images_dir: str, epochs: int = 50, imgsz: int = 640):
        """Client-side local training on factory data."""
        self._ensure_model()
        results = self.model.train(
            data=f"{images_dir}/data.yaml",
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
        )
        return results

    def detect(self, image, conf_threshold: float = 0.25):
        """Detect objects in image.

        Returns:
            List of Detection objects.
        """
        self._ensure_model()
        results = self.model(image, conf=conf_threshold)
        detections = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                detections.append(Detection(
                    bbox=xyxy,
                    class_name=self.class_names[int(box.cls)],
                    class_id=int(box.cls),
                    confidence=float(box.conf),
                    cx=(x1 + x2) / 2,
                    cy=(y1 + y2) / 2,
                    width=x2 - x1,
                    height=y2 - y1,
                    area=(x2 - x1) * (y2 - y1),
                ))
        return detections

    def export_backbone_weights(self) -> dict:
        """Export backbone weights for FedAvg aggregation.

        Only the shared feature extractor is shared —
        detection head stays local (factory-specific).
        """
        self._ensure_model()
        backbone_params = {}
        for name, param in self.model.model.state_dict().items():
            # Only share backbone layers (not detection head)
            if any(prefix in name for prefix in ["model.0", "model.1", "model.2", "model.3",
                                                   "model.4", "model.5", "model.6", "model.7",
                                                   "model.8", "model.9"]):
                backbone_params[name] = param.clone()
        return backbone_params

    def load_backbone_weights(self, state_dict: dict):
        """Load aggregated backbone weights from server."""
        self._ensure_model()
        current = self.model.model.state_dict()
        for name, param in state_dict.items():
            if name in current:
                current[name] = param
        self.model.model.load_state_dict(current)

    def count_by_class(self, detections: list[Detection]) -> dict[str, int]:
        counts = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def summary(self, detections: list[Detection]) -> dict:
        if not detections:
            return {"total": 0, "classes": {}, "avg_confidence": 0, "avg_area": 0}
        return {
            "total": len(detections),
            "classes": self.count_by_class(detections),
            "avg_confidence": float(np.mean([d.confidence for d in detections])),
            "avg_area": float(np.mean([d.area for d in detections])),
        }
