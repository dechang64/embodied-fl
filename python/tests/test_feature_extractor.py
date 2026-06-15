# ── python/tests/test_feature_extractor.py ──
"""Tests for DINOv2 Scene Feature Extractor."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from analysis.feature_extractor import (
    DINOv2SceneExtractor, MetadataFallbackExtractor, get_extractor,
)


class TestDINOv2SceneExtractor:
    def test_model_dims(self):
        assert DINOv2SceneExtractor.MODEL_DIMS["vits14"] == 384
        assert DINOv2SceneExtractor.MODEL_DIMS["base"] == 768
        assert DINOv2SceneExtractor.MODEL_DIMS["large"] == 1024

    def test_dim_property(self):
        ext = DINOv2SceneExtractor.__new__(DINOv2SceneExtractor)
        ext.variant = "base"
        ext.dim = 768
        assert ext.dim == 768


class TestMetadataFallbackExtractor:
    def test_basic(self):
        ext = MetadataFallbackExtractor(dim=32)
        vec = ext.embed("grasping", "electronics", "rgb", 0.5, "medium", "low")
        assert len(vec) == 32
        assert vec.dtype == np.float32

    def test_different_tasks(self):
        ext = MetadataFallbackExtractor(dim=32)
        v1 = ext.embed("grasping", "electronics", "rgb", 0.5, "medium", "low")
        v2 = ext.embed("navigation", "logistics", "depth", 0.8, "complex", "high")
        assert not np.allclose(v1, v2)

    def test_consistency(self):
        ext = MetadataFallbackExtractor(dim=32)
        v1 = ext.embed("grasping", "electronics", "rgb", 0.5, "medium", "low")
        v2 = ext.embed("grasping", "electronics", "rgb", 0.5, "medium", "low")
        assert np.allclose(v1, v2)


class TestGetExtractor:
    def test_factory_invalid(self):
        with pytest.raises(ValueError):
            get_extractor("invalid_mode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
