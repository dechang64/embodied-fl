# ── python/tests/test_instruction_embedding.py ──
"""Tests for InstructionEmbedder module."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np

from analysis.instruction_embedding import InstructionEmbedder, EmbeddingConfig


class TestInstructionEmbedder:
    def setup_method(self):
        self.embedder = InstructionEmbedder(EmbeddingConfig(
            mode="hash", dimension=64,
        ))

    def test_embed_single(self):
        """Single instruction should produce embedding of correct dimension."""
        emb = self.embedder.embed("pick up the red cup")
        assert emb.shape == (64,)
        assert emb.dtype == np.float32

    def test_embed_batch(self):
        """Batch embedding should produce (N, dim) array."""
        instructions = ["pick up the cup", "go to the door", "inspect the PCB"]
        embs = self.embedder.embed_batch(instructions)
        assert embs.shape == (3, 64)

    def test_deterministic(self):
        """Same instruction should produce same embedding."""
        e1 = self.embedder.embed("pick up the cup")
        e2 = self.embedder.embed("pick up the cup")
        np.testing.assert_array_equal(e1, e2)

    def test_different_instructions_different_embeddings(self):
        """Different instructions should produce different embeddings."""
        e1 = self.embedder.embed("pick up the cup")
        e2 = self.embedder.embed("go to the door")
        assert not np.allclose(e1, e2)

    def test_similarity(self):
        """Similarity should be in [-1, 1]."""
        e1 = self.embedder.embed("pick up the cup")
        e2 = self.embedder.embed("grab the cup")
        sim = self.embedder.similarity(e1, e2)
        assert -1.0 <= sim <= 1.0

    def test_identical_similarity(self):
        """Identical embeddings should have similarity 1.0."""
        e = self.embedder.embed("pick up the cup")
        sim = self.embedder.similarity(e, e)
        assert abs(sim - 1.0) < 1e-5

    def test_find_most_similar(self):
        """find_most_similar should rank correctly."""
        query = "pick up the red cup"
        candidates = [
            "grab the red cup",
            "go to the kitchen",
            "inspect the PCB",
            "pick up the blue plate",
        ]
        results = self.embedder.find_most_similar(query, candidates, top_k=2)
        assert len(results) == 2
        assert results[0][2] >= results[1][2]  # Sorted by similarity

    def test_empty_batch(self):
        """Empty batch should return empty array."""
        embs = self.embedder.embed_batch([])
        assert embs.shape == (0, 64)

    def test_empty_find_similar(self):
        """Empty candidates should return empty list."""
        results = self.embedder.find_most_similar("test", [], top_k=5)
        assert results == []

    def test_config_dimension(self):
        """Different dimensions should work."""
        embedder_128 = InstructionEmbedder(EmbeddingConfig(mode="hash", dimension=128))
        emb = embedder_128.embed("test")
        assert emb.shape == (128,)


class TestEmbeddingConfig:
    def test_defaults(self):
        config = EmbeddingConfig()
        assert config.mode == "hash"
        assert config.dimension == 384
        assert config.normalize is True

    def test_custom(self):
        config = EmbeddingConfig(mode="sentence_transformer", dimension=768, device="cuda")
        assert config.mode == "sentence_transformer"
        assert config.dimension == 768
        assert config.device == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
