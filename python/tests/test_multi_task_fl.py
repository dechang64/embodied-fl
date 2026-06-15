# ── python/tests/test_multi_task_fl.py ──
"""Tests for EmbodiedMultiTaskFL engine."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import torch
from analysis.multi_task_fl import EmbodiedMultiTaskFL


class TestEmbodiedMultiTaskFL:
    def test_init(self):
        engine = EmbodiedMultiTaskFL(
            input_dim=64, num_classes=5, action_dim=4,
            hidden_dim=16, lr=0.01, local_epochs=1,
        )
        assert engine.input_dim == 64
        assert engine.num_classes == 5

    def test_basic_training(self):
        engine = EmbodiedMultiTaskFL(
            input_dim=64, num_classes=3, action_dim=4,
            hidden_dim=16, lr=0.01, local_epochs=1,
        )
        features = np.random.randn(300, 64).astype(np.float32)
        labels = np.random.randint(0, 3, 300).astype(np.int64)

        history = engine.run(features, labels, n_clients=3, rounds=3)

        assert len(history) == 3
        assert all("val_acc" in h for h in history)
        assert history[-1]["val_acc"] > 0.1

    def test_classifier_dimensions(self):
        engine = EmbodiedMultiTaskFL(input_dim=128, num_classes=10, action_dim=6)
        x = torch.randn(2, 128)
        out = engine.classifier(x)
        assert out.shape == (2, 10)

    def test_policy_dimensions(self):
        engine = EmbodiedMultiTaskFL(input_dim=64, num_classes=3, action_dim=6)
        x = torch.randn(2, 64)
        out = engine.policy_head(x)
        assert out.shape == (2, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
