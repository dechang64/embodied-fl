# ── python/tests/test_vla.py ──
"""Tests for VLA (Vision-Language-Action) modules."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
import torch

from analysis.vla_collector import (
    BaseCollector, SyntheticCollector, Episode, Step,
    Observation, Action, compute_episode_statistics,
)
from analysis.vla_dataset import VLADataset, VLASample
from analysis.action_tokenizer import (
    ActionTokenizer, DeltaActionTokenizer, TokenizerConfig,
)
from analysis.instruction_parser import InstructionParser, ParsedInstruction
from analysis.vla_model import VLAFLModel, VLAFLTrainer, VLAConfig


# ═══════════════════════════════════════════════════════════════
# Action Tokenizer Tests
# ═══════════════════════════════════════════════════════════════

class TestActionTokenizer:
    def test_encode_decode_roundtrip(self):
        """Continuous action → tokens → continuous action should be close."""
        tokenizer = ActionTokenizer(TokenizerConfig(action_dim=8, num_bins=256))
        action = np.array([0.5, -0.3, 0.0, 0.8, -0.5, 0.1, -0.9, 0.2], dtype=np.float32)
        tokens = tokenizer.encode(action)
        assert tokens.shape == (8,)
        assert tokens.dtype == np.int64

        decoded = tokenizer.decode(tokens)
        assert decoded.shape == (8,)
        max_error = np.max(np.abs(decoded - action))
        assert max_error < 1.0 / 256 * 2, f"Quantization error too large: {max_error}"

    def test_batch_encode_decode(self):
        """Batch encode/decode should work."""
        tokenizer = ActionTokenizer(TokenizerConfig(action_dim=4, num_bins=128))
        actions = np.random.randn(10, 4).astype(np.float32) * 0.5
        tokens = tokenizer.encode_batch(actions)
        assert tokens.shape == (10, 4)

        decoded = tokenizer.decode_batch(tokens)
        assert decoded.shape == (10, 4)

    def test_out_of_range_clipping(self):
        """Values outside [low, high] should be clipped."""
        tokenizer = ActionTokenizer(TokenizerConfig(action_dim=3, num_bins=256))
        action = np.array([5.0, -5.0, 0.0])  # Way out of range
        tokens = tokenizer.encode(action)
        decoded = tokenizer.decode(tokens)
        assert np.all(decoded >= -1.0)
        assert np.all(decoded <= 1.0)

    def test_per_dim_bounds(self):
        """Per-dimension bounds should override global bounds."""
        config = TokenizerConfig(
            action_dim=3,
            num_bins=256,
            per_dim_low=[0.0, -1.0, 0.0],
            per_dim_high=[1.0, 1.0, 2.0],
        )
        tokenizer = ActionTokenizer(config)
        action = np.array([0.0, 0.0, 1.0])
        tokens = tokenizer.encode(action)
        decoded = tokenizer.decode(tokens)
        assert np.allclose(decoded, action, atol=0.1)

    def test_config_serialization(self):
        """Tokenizer config should serialize to dict."""
        config = TokenizerConfig(action_dim=6, num_bins=256)
        d = config.to_dict()
        assert d["action_dim"] == 6
        assert d["num_bins"] == 256

        restored = TokenizerConfig.from_dict(d)
        assert restored.action_dim == config.action_dim
        assert restored.num_bins == config.num_bins

    def test_vocab_size(self):
        """Vocab size should be num_bins + special tokens."""
        config = TokenizerConfig(action_dim=4, num_bins=256)
        tokenizer = ActionTokenizer(config)
        assert tokenizer.vocab_size == 256 + 3  # pad + eos + sos


class TestDeltaActionTokenizer:
    def test_delta_encode_decode(self):
        """Delta tokenizer should encode relative actions."""
        tokenizer = DeltaActionTokenizer(TokenizerConfig(action_dim=4))
        current = np.array([0.1, 0.2, 0.3, 0.4])
        target = np.array([0.15, 0.25, 0.35, 0.45])
        delta = target - current

        tokens = tokenizer.encode_delta(current, target)
        decoded_delta = tokenizer.decode(tokens)
        assert np.allclose(decoded_delta, delta, atol=0.05)

        abs_action = tokenizer.decode_delta(tokens, current)
        assert np.allclose(abs_action, target, atol=0.05)


# ═══════════════════════════════════════════════════════════════
# Instruction Parser Tests
# ═══════════════════════════════════════════════════════════════

class TestInstructionParser:
    def setup_method(self):
        self.parser = InstructionParser()

    def test_english_grasping(self):
        result = self.parser.parse("pick up the red cup")
        assert result.task_type == "grasping"
        assert result.language == "en"
        assert result.task_confidence > 0

    def test_chinese_grasping(self):
        result = self.parser.parse("抓起红色杯子")
        assert result.task_type == "grasping"
        assert result.language == "zh"

    def test_english_navigation(self):
        result = self.parser.parse("go to the kitchen")
        assert result.task_type == "navigation"

    def test_chinese_assembly(self):
        result = self.parser.parse("把螺丝装配到孔里")
        assert result.task_type == "assembly"

    def test_spatial_relation(self):
        result = self.parser.parse("put the cup on the table")
        assert result.task_type == "placing"
        assert len(result.spatial_relations) > 0

    def test_object_extraction(self):
        result = self.parser.parse("pick up the red cup and place it on the blue plate")
        assert "red cup" in result.objects
        assert "blue plate" in result.objects

    def test_mixed_language(self):
        result = self.parser.parse("pick up the 杯子")
        assert result.language == "mixed"

    def test_empty_instruction(self):
        result = self.parser.parse("")
        assert result.task_type == "custom"
        assert result.task_confidence == 0.0

    def test_batch_parse(self):
        instructions = [
            "pick up the cup",
            "go to the door",
            "inspect the PCB",
        ]
        results = self.parser.parse_batch(instructions)
        assert len(results) == 3
        assert results[0].task_type == "grasping"
        assert results[1].task_type == "navigation"
        assert results[2].task_type == "inspection"

    def test_task_distribution(self):
        instructions = [
            "pick up the cup", "grab the tool", "grasp the handle",
            "go to the door", "move to the table",
        ]
        dist = self.parser.get_task_distribution(instructions)
        assert dist["grasping"] == 3
        assert dist["navigation"] == 2

    def test_embedding_hash_deterministic(self):
        r1 = self.parser.parse("pick up the cup")
        r2 = self.parser.parse("pick up the cup")
        assert r1.embedding_hash == r2.embedding_hash

    def test_to_dict(self):
        result = self.parser.parse("pick up the red cup")
        d = result.to_dict()
        assert "task_type" in d
        assert "objects" in d
        assert d["task_type"] == "grasping"


# ═══════════════════════════════════════════════════════════════
# VLA Collector Tests
# ═══════════════════════════════════════════════════════════════

class TestVLACollector:
    def test_synthetic_collector(self):
        """SyntheticCollector should generate valid episodes."""
        collector = SyntheticCollector(robot_type="franka_panda")
        episodes = collector.collect(num_episodes=3, steps_per_episode=10)

        assert len(episodes) == 3
        for ep in episodes:
            assert isinstance(ep, Episode)
            assert ep.num_steps == 10
            assert ep.episode_id.startswith("synth_ep_")
            assert ep.metadata["robot_type"] == "franka_panda"
            for step in ep.steps:
                assert isinstance(step, Step)
                assert isinstance(step.observation, Observation)
                assert isinstance(step.action, Action)
                assert len(step.observation.robot_state) == 7

    def test_episode_serialization(self):
        """Episode.to_dict() should produce valid dict."""
        collector = SyntheticCollector()
        episodes = collector.collect(num_episodes=1, steps_per_episode=5)
        data = episodes[0].to_dict()
        assert "episode_id" in data
        assert "steps" in data
        assert "metadata" in data
        assert len(data["steps"]) == 5

    def test_episode_rlds_format(self):
        """Episode.to_rlds() should produce RLDS-compatible format."""
        collector = SyntheticCollector()
        episodes = collector.collect(num_episodes=1, steps_per_episode=3)
        rlds = episodes[0].to_rlds()
        assert "steps" in rlds
        assert rlds["steps"][0]["is_first"] is True
        assert rlds["steps"][-1]["is_last"] is True

    def test_episode_statistics(self):
        """Episode should compute basic statistics."""
        collector = SyntheticCollector(robot_type="franka_panda")
        episodes = collector.collect(num_episodes=5, steps_per_episode=20)

        for ep in episodes:
            assert ep.num_steps == 20
            assert 0 <= ep.get_success_rate() <= 1.0
            assert isinstance(ep.get_avg_reward(), float)

    def test_episode_dimensions(self):
        """Episode should infer action_dim and state_dim."""
        collector = SyntheticCollector(state_dim=6, action_dim=7)
        episodes = collector.collect(num_episodes=1, steps_per_episode=5)
        ep = episodes[0]
        assert ep.state_dim == 6
        assert ep.action_dim == 7

    def test_compute_episode_statistics(self):
        """compute_episode_statistics should aggregate over episodes."""
        collector = SyntheticCollector()
        episodes = collector.collect(num_episodes=10, steps_per_episode=20)
        stats = compute_episode_statistics(episodes)
        assert stats["num_episodes"] == 10
        assert stats["total_steps"] == 200
        assert "success_rate" in stats
        assert "avg_reward" in stats


# ═══════════════════════════════════════════════════════════════
# VLA Dataset Tests
# ═══════════════════════════════════════════════════════════════

class TestVLADataset:
    def test_from_episodes(self):
        """VLADataset should load from Episode objects."""
        collector = SyntheticCollector()
        episodes = collector.collect(num_episodes=3, steps_per_episode=10)
        dataset = VLADataset.from_episodes(episodes)
        assert len(dataset) == 30

    def test_from_dicts(self):
        """VLADataset should load from list of episode dicts."""
        collector = SyntheticCollector()
        episodes = collector.collect(num_episodes=2, steps_per_episode=5)
        dicts = [ep.to_dict() for ep in episodes]
        dataset = VLADataset.from_dicts(dicts)
        assert len(dataset) == 10

    def test_summary(self):
        """Dataset summary should compute statistics."""
        collector = SyntheticCollector()
        episodes = collector.collect(num_episodes=5, steps_per_episode=20)
        dataset = VLADataset.from_episodes(episodes)
        summary = dataset.summary()
        assert summary["num_samples"] == 100
        assert summary["num_episodes"] == 5

    def test_episode_boundaries(self):
        """Episode boundaries should be correct."""
        collector = SyntheticCollector()
        episodes = collector.collect(num_episodes=3, steps_per_episode=10)
        dataset = VLADataset.from_episodes(episodes)
        boundaries = dataset.get_episode_boundaries()
        assert len(boundaries) == 3
        assert boundaries[0] == (0, 10)
        assert boundaries[1] == (10, 20)
        assert boundaries[2] == (20, 30)

    def test_normalize_robot_state(self):
        """Normalization should standardize robot states."""
        collector = SyntheticCollector()
        episodes = collector.collect(num_episodes=2, steps_per_episode=50)
        dataset = VLADataset.from_episodes(episodes)
        dataset.normalize_robot_state()
        states = np.array([s.robot_state for s in dataset.samples])
        mean = np.mean(states, axis=0)
        assert np.allclose(mean, 0, atol=0.1)


# ═══════════════════════════════════════════════════════════════
# VLA Model Tests
# ═══════════════════════════════════════════════════════════════

class TestVLAFLModel:
    def test_forward_pass(self):
        """Model should produce correct output shape."""
        config = VLAConfig(
            vision_dim=768, lang_dim=384, state_dim=8,
            d_model=256, action_dim=4, num_action_bins=32,
            n_heads=4, n_fusion_layers=2,
        )
        model = VLAFLModel(config)

        B = 4
        vision = torch.randn(B, 768)
        lang = torch.randn(B, 8, 384)
        state = torch.randn(B, 8)

        logits = model(vision, lang, state)
        assert logits.shape == (B, 4, 32)

    def test_with_attention_mask(self):
        """Model should handle attention masks."""
        config = VLAConfig(
            vision_dim=128, lang_dim=64, state_dim=4,
            d_model=64, action_dim=3, num_action_bins=16,
            n_heads=2, n_fusion_layers=1,
        )
        model = VLAFLModel(config)

        B = 2
        vision = torch.randn(B, 128)
        lang = torch.randn(B, 10, 64)
        mask = torch.ones(B, 10)
        mask[:, 7:] = 0  # Mask last 3 tokens
        state = torch.randn(B, 4)

        logits = model(vision, lang, state, attention_mask=mask)
        assert logits.shape == (2, 3, 16)

    def test_compute_loss(self):
        """Loss computation should work."""
        config = VLAConfig(
            vision_dim=64, lang_dim=32, state_dim=4,
            d_model=32, action_dim=3, num_action_bins=16,
            n_heads=2, n_fusion_layers=1,
        )
        model = VLAFLModel(config)

        B = 8
        vision = torch.randn(B, 64)
        lang = torch.randn(B, 6, 32)
        state = torch.randn(B, 4)
        targets = torch.randint(0, 16, (B, 3))

        logits = model(vision, lang, state)
        loss = model.compute_loss(logits, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_shared_vs_local_params(self):
        """Model should distinguish shared and local parameters."""
        config = VLAConfig(
            vision_dim=128, lang_dim=64, state_dim=8,
            d_model=64, action_dim=4, num_action_bins=32,
            n_heads=2, n_fusion_layers=1,
        )
        model = VLAFLModel(config)
        counts = model.count_parameters()

        assert counts["total"] > 0
        assert counts["shared"] > 0
        assert counts["local"] > 0
        assert counts["total"] == counts["shared"] + counts["local"]

    def test_load_shared_params(self):
        """Loading shared params should preserve them exactly."""
        config = VLAConfig(d_model=64, n_heads=2, n_fusion_layers=1)
        model = VLAFLModel(config)

        shared = model.get_shared_state_dict()

        model2 = VLAFLModel(config)
        model2.load_shared_params(shared)

        for name, param in shared.items():
            assert torch.equal(param, dict(model2.named_parameters())[name])

    def test_predict(self):
        """Predict should return token IDs."""
        config = VLAConfig(
            d_model=64, n_heads=2, n_fusion_layers=1,
            action_dim=4, num_action_bins=32,
        )
        model = VLAFLModel(config)

        vision = torch.randn(2, 768)
        lang = torch.randn(2, 8, 384)
        state = torch.randn(2, 8)

        tokens = model.predict(vision, lang, state)
        assert tokens.shape == (2, 4)
        assert tokens.dtype == np.int64

    def test_predict_actions(self):
        """Predict actions should return continuous actions."""
        config = VLAConfig(
            d_model=64, n_heads=2, n_fusion_layers=1,
            action_dim=4, num_action_bins=32,
        )
        model = VLAFLModel(config)

        vision = torch.randn(2, 768)
        lang = torch.randn(2, 8, 384)
        state = torch.randn(2, 8)

        actions = model.predict_actions(vision, lang, state)
        assert actions.shape == (2, 4)
        assert actions.dtype == np.float32
        assert np.all(actions >= -1.0)
        assert np.all(actions <= 1.0)


class TestVLAFLTrainer:
    def test_local_training(self):
        """Local training should decrease loss."""
        config = VLAConfig(
            vision_dim=128, lang_dim=64, state_dim=8,
            d_model=64, action_dim=4, num_action_bins=32,
            n_heads=2, n_fusion_layers=1,
            lr=0.01, local_epochs=5,
        )
        trainer = VLAFLTrainer(config)

        N = 50
        vision = torch.randn(N, 128)
        lang = torch.randn(N, 8, 64)
        state = torch.randn(N, 8)
        targets = torch.randint(0, 32, (N, 4))

        result = trainer.train_local(vision, lang, state, targets)
        assert "final_loss" in result
        assert "accuracy" in result
        # Loss should generally decrease over epochs
        assert result["losses"][-1] <= result["losses"][0] + 0.1

    def test_upload_payload(self):
        """Upload payload should contain shared params."""
        config = VLAConfig(d_model=64, n_heads=2, n_fusion_layers=1)
        trainer = VLAFLTrainer(config)

        payload = trainer.get_upload_payload()
        assert "shared_params" in payload
        assert "param_count" in payload

    def test_fedavg_simulation(self):
        """Simulate 3-client FedAvg on VLA model."""
        config = VLAConfig(
            vision_dim=128, lang_dim=64, state_dim=8,
            d_model=64, action_dim=4, num_action_bins=32,
            n_heads=2, n_fusion_layers=1,
            lr=0.01, local_epochs=2,
        )

        clients = [VLAFLTrainer(config) for _ in range(3)]

        N = 30
        for trainer in clients:
            vision = torch.randn(N, 128)
            lang = torch.randn(N, 8, 64)
            state = torch.randn(N, 8)
            targets = torch.randint(0, 32, (N, 4))
            trainer.train_local(vision, lang, state, targets)

        # FedAvg: average shared params
        shared_params = {}
        all_shared = [c.model.get_shared_state_dict() for c in clients]
        for key in all_shared[0]:
            shared_params[key] = torch.stack(
                [s[key] for s in all_shared]
            ).mean(dim=0)

        global_model = VLAFLModel(config)
        global_model.load_shared_params(shared_params)

        vision = torch.randn(5, 128)
        lang = torch.randn(5, 8, 64)
        state = torch.randn(5, 8)
        logits = global_model(vision, lang, state)
        assert logits.shape == (5, 4, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
