# ── python/tests/test_gradcam.py ──
"""Tests for Grad-CAM explainability."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import torch
import torch.nn as nn
from analysis.gradcam import GradCAM, generate_robot_explanation


class DummyConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 6)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestGradCAM:
    def test_init(self):
        model = DummyConvModel()
        cam = GradCAM(model, target_layer=model.conv2)
        assert cam.target_layer == model.conv2

    def test_generate_heatmap(self):
        model = DummyConvModel()
        model.eval()
        cam = GradCAM(model, target_layer=model.conv2)
        x = torch.randn(1, 3, 224, 224)
        heatmap = cam.generate(x, target_class=0)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1


class TestRobotExplanation:
    def test_basic(self):
        heatmap = np.random.rand(224, 224).astype(np.float32)
        report = generate_robot_explanation(
            heatmap=heatmap, action="grasp", confidence=0.92,
        )
        assert "grasp" in report
        assert "92.0%" in report

    def test_with_scene(self):
        heatmap = np.random.rand(224, 224).astype(np.float32)
        report = generate_robot_explanation(
            heatmap=heatmap, action="navigate", confidence=0.85,
            scene_description={"obstacles": 3, "target_distance": "2.1m"},
        )
        assert "navigate" in report
        assert "Scene Context" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
