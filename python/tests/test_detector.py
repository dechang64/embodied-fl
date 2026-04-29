# ── python/tests/test_detector.py ──
"""Tests for RobotSceneDetector."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from analysis.detector import RobotSceneDetector, Detection


class TestDetection:
    def test_creation(self):
        d = Detection(
            bbox=[10, 20, 100, 200], class_name="workpiece",
            class_id=0, confidence=0.95, cx=55, cy=110,
            width=90, height=180, area=16200,
        )
        assert d.class_name == "workpiece"
        assert d.confidence == 0.95

    def test_to_dict(self):
        d = Detection(
            bbox=[0, 0, 50, 50], class_name="defect",
            class_id=6, confidence=0.88, cx=25, cy=25,
            width=50, height=50, area=2500,
        )
        d_dict = d.to_dict()
        assert d_dict["class_name"] == "defect"
        assert "bbox" in d_dict


class TestRobotSceneDetector:
    def test_init(self):
        det = RobotSceneDetector(model_size="n")
        assert det.model_size == "n"
        assert len(det.DEFAULT_CLASSES) == 10

    def test_count_by_class(self):
        det = RobotSceneDetector(model_size="n")
        detections = [
            Detection([0,0,50,50], "workpiece", 0, 0.9, 25, 25, 50, 50, 2500),
            Detection([0,0,50,50], "workpiece", 0, 0.8, 25, 25, 50, 50, 2500),
            Detection([0,0,50,50], "defect", 6, 0.7, 25, 25, 50, 50, 2500),
        ]
        counts = det.count_by_class(detections)
        assert counts == {"workpiece": 2, "defect": 1}

    def test_summary_empty(self):
        det = RobotSceneDetector(model_size="n")
        summary = det.summary([])
        assert summary["total"] == 0

    def test_summary_nonempty(self):
        det = RobotSceneDetector(model_size="n")
        detections = [
            Detection([0,0,50,50], "workpiece", 0, 0.9, 25, 25, 50, 50, 2500),
            Detection([0,0,100,100], "tool", 1, 0.7, 50, 50, 100, 100, 10000),
        ]
        summary = det.summary(detections)
        assert summary["total"] == 2
        assert summary["avg_confidence"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
