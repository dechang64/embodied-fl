# ── python/analysis/detector.py ──
"""
YOLOv11 Robot Scene Detector (twc-core wrapper)
================================================
Delegates to twc_core.detector.Detector.
RobotSceneDetector adds embodied-intelligence-specific defaults.
"""

from twc_core.detector import Detection, Detector

__all__ = ["Detection", "RobotSceneDetector"]


class RobotSceneDetector(Detector):
    """YOLOv11-based robot scene detector.

    Default classes for factory/warehouse scenarios.
    Extends twc_core.Detector with embodied-intelligence defaults.
    """

    DEFAULT_CLASSES = [
        "workpiece", "tool", "fixture", "conveyor",
        "human_worker", "safety_zone", "defect",
        "package", "pallet", "forklift",
    ]

    def __init__(self, model_size: str = "n", classes=None, device=None):
        super().__init__(
            model_name=f"yolo11{model_size}.pt",
            classes=classes or self.DEFAULT_CLASSES,
        )
        self.model_size = model_size
        self.class_names = self.classes
