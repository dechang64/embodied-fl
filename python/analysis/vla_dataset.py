# ── python/analysis/vla_dataset.py ──
"""
VLA Dataset for Federated Training
====================================
PyTorch Dataset that loads VLA episodes and produces training samples
compatible with federated learning pipelines.

Each sample:
  {
    "image": Tensor (C, H, W),          // from DINOv2 preprocessing
    "robot_state": Tensor (state_dim),   // joint angles
    "gripper": Tensor (1),               // gripper aperture
    "instruction_tokens": Tensor (L),    // tokenized language instruction
    "instruction_embedding": Tensor (d), // sentence-transformer embedding
    "action": Tensor (action_dim),       // target action (for supervised learning)
    "attention_mask": Tensor (L),        // for language padding
  }

Federated Learning Integration:
  - Each client (robot/factory) gets its own VLA dataset
  - Shared Backbone (DINOv2 + Lang Encoder) aggregated via FedAvg
  - Action Head stays local (different action spaces)
  - Task matching via HNSW on instruction embeddings
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Optional, Union, Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class VLASample:
    """Single training sample for VLA model."""
    image: Optional[np.ndarray] = None          # (C, H, W) float32
    robot_state: np.ndarray = None              # (state_dim,) float32
    gripper: float = 0.5
    instruction: str = ""
    instruction_tokens: Optional[np.ndarray] = None   # (L,) int64
    instruction_embedding: Optional[np.ndarray] = None  # (d,) float32
    action: Optional[np.ndarray] = None         # (action_dim,) float32
    attention_mask: Optional[np.ndarray] = None  # (L,) int64
    reward: float = 0.0
    episode_id: str = ""
    step_idx: int = 0

    def __post_init__(self):
        if self.robot_state is None:
            self.robot_state = np.array([], dtype=np.float32)
        if self.action is None:
            self.action = np.array([], dtype=np.float32)
        if self.attention_mask is None and self.instruction_tokens is not None:
            self.attention_mask = np.ones_like(self.instruction_tokens)


class VLADataset:
    """In-memory VLA dataset for federated training.

    Loads episodes from JSON/RLDS files and converts to training samples.
    Supports optional image loading and instruction tokenization.
    """

    def __init__(
        self,
        samples: Optional[list[VLASample]] = None,
        max_seq_len: int = 32,
        normalize_actions: bool = True,
        normalize_states: bool = True,
    ):
        self.samples = samples or []
        self.max_seq_len = max_seq_len
        self.normalize_actions = normalize_actions
        self.normalize_states = normalize_states

        # Statistics for normalization (computed on first access)
        self._state_stats = None
        self._action_stats = None

    @classmethod
    def from_episodes(
        cls,
        episodes: list,
        max_seq_len: int = 32,
        skip_no_action: bool = True,
        instruction_mode: str = "raw",
    ) -> "VLADataset":
        """Create dataset from Episode objects.

        Args:
            episodes: List of Episode objects (from vla_collector).
            max_seq_len: Maximum instruction token length.
            skip_no_action: Skip steps without actions.
            instruction_mode: "raw" (keep string), "hash" (hash to int tokens).
        """
        samples = []
        for ep in episodes:
            ep_dict = ep.to_dict() if hasattr(ep, "to_dict") else ep
            ep_id = ep_dict.get("episode_id", "unknown")
            steps = ep_dict.get("steps", [])

            for step_idx, step in enumerate(steps):
                obs = step.get("observation", {})
                action = step.get("action")

                if skip_no_action and action is None:
                    continue

                # Robot state
                state = np.array(
                    obs.get("robot_state", []),
                    dtype=np.float32,
                )

                # Gripper
                gripper = float(obs.get("gripper", 0.5))

                # Instruction
                instruction = step.get("instruction", "")

                # Instruction tokens (simple hash-based tokenization)
                inst_tokens = None
                if instruction and instruction_mode == "hash":
                    inst_tokens = cls._tokenize_instruction(
                        instruction, max_seq_len
                    )

                # Action
                act = None
                if action is not None:
                    target_pos = action.get("target_joint_pos", [])
                    grip_cmd = action.get("gripper_command", gripper)
                    act_vec = list(target_pos) + [grip_cmd]
                    act = np.array(act_vec, dtype=np.float32)

                sample = VLASample(
                    robot_state=state,
                    gripper=gripper,
                    instruction=instruction,
                    instruction_tokens=inst_tokens,
                    action=act,
                    reward=float(step.get("reward", 0.0)),
                    episode_id=ep_id,
                    step_idx=step_idx,
                )
                samples.append(sample)

        return cls(samples=samples, max_seq_len=max_seq_len)

    @classmethod
    def from_json_dir(
        cls,
        json_dir: Union[str, Path],
        max_seq_len: int = 32,
    ) -> "VLADataset":
        """Load dataset from a directory of JSON episode files."""
        from .vla_collector import JSONLogCollector

        collector = JSONLogCollector()
        episodes = collector.collect(json_dir)
        return cls.from_episodes(episodes, max_seq_len=max_seq_len)

    @classmethod
    def from_rlds_bundle(
        cls,
        rlds_path: Union[str, Path],
        max_seq_len: int = 32,
    ) -> "VLADataset":
        """Load dataset from an RLDS bundle JSON file."""
        rlds_path = Path(rlds_path)
        with open(rlds_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        episodes = bundle.get("episodes", [])
        return cls.from_episodes(episodes, max_seq_len=max_seq_len)

    @classmethod
    def from_dicts(
        cls,
        episode_dicts: list[dict],
        max_seq_len: int = 32,
        skip_no_action: bool = True,
    ) -> "VLADataset":
        """Create dataset from episode dictionaries.

        Args:
            episode_dicts: List of episode dicts (as produced by Episode.to_dict()).
            max_seq_len: Maximum instruction token length.
            skip_no_action: Skip steps without actions.
        """
        return cls.from_episodes(
            episode_dicts,
            max_seq_len=max_seq_len,
            skip_no_action=skip_no_action,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VLASample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[VLASample]:
        return iter(self.samples)

    def split_by_episode(
        self, n_clients: int, method: str = "round_robin"
    ) -> list["VLADataset"]:
        """Split dataset into client subsets for federated learning.

        Args:
            n_clients: Number of federated clients.
            method: "round_robin" or "episode" (each episode to one client).

        Returns:
            List of VLADataset, one per client.
        """
        if method == "round_robin":
            client_samples = [[] for _ in range(n_clients)]
            for i, sample in enumerate(self.samples):
                client_samples[i % n_clients].append(sample)
        elif method == "episode":
            # Group by episode, then distribute
            episode_groups: dict[str, list] = {}
            for sample in self.samples:
                eid = sample.episode_id
                if eid not in episode_groups:
                    episode_groups[eid] = []
                episode_groups[eid].append(sample)

            episode_list = list(episode_groups.values())
            client_samples = [[] for _ in range(n_clients)]
            for i, ep_samples in enumerate(episode_list):
                client_samples[i % n_clients].extend(ep_samples)
        else:
            raise ValueError(f"Unknown split method: {method}")

        return [
            VLADataset(
                samples=samples,
                max_seq_len=self.max_seq_len,
                normalize_actions=self.normalize_actions,
                normalize_states=self.normalize_states,
            )
            for samples in client_samples
            if samples  # skip empty clients
        ]

    def get_instruction_embeddings(
        self, model=None, batch_size: int = 32
    ) -> np.ndarray:
        """Compute instruction embeddings for all samples.

        Args:
            model: Optional sentence-transformer model. If None, uses hash-based.
            batch_size: Batch size for encoding.

        Returns:
            (N, d) numpy array of instruction embeddings.
        """
        instructions = [s.instruction for s in self.samples]

        if model is not None:
            # Use sentence-transformer
            embeddings = model.encode(
                instructions,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.array(embeddings, dtype=np.float32)
        else:
            # Hash-based fallback (deterministic, no external dependency)
            dim = 128
            embeddings = np.zeros((len(instructions), dim), dtype=np.float32)
            for i, inst in enumerate(instructions):
                h = hashlib.sha256(inst.encode("utf-8")).digest()
                # Convert 32 bytes to 128 floats
                for j in range(dim):
                    byte_idx = j % 32
                    embeddings[i, j] = h[byte_idx] / 255.0
            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            embeddings /= norms
            return embeddings

    def compute_normalization_stats(self):
        """Compute mean/std for states and actions for normalization."""
        states = np.array([s.robot_state for s in self.samples if len(s.robot_state) > 0])
        actions = np.array([s.action for s in self.samples if s.action is not None and len(s.action) > 0])

        self._state_stats = {
            "mean": states.mean(axis=0) if len(states) > 0 else np.array([]),
            "std": states.std(axis=0) if len(states) > 0 else np.array([]),
        }
        self._action_stats = {
            "mean": actions.mean(axis=0) if len(actions) > 0 else np.array([]),
            "std": actions.std(axis=0) if len(actions) > 0 else np.array([]),
        }
        return self._state_stats, self._action_stats

    def normalize(self):
        """Apply normalization to all samples in-place."""
        if self._state_stats is None:
            self.compute_normalization_stats()

        if self.normalize_states and self._state_stats["mean"].size > 0:
            for s in self.samples:
                if len(s.robot_state) > 0:
                    s.robot_state = (
                        (s.robot_state - self._state_stats["mean"])
                        / (self._state_stats["std"] + 1e-8)
                    )

        if self.normalize_actions and self._action_stats["mean"].size > 0:
            for s in self.samples:
                if s.action is not None and len(s.action) > 0:
                    s.action = (
                        (s.action - self._action_stats["mean"])
                        / (self._action_stats["std"] + 1e-8)
                    )

    def normalize_robot_state(self):
        """Normalize only robot states (alias for normalize with states only)."""
        if self._state_stats is None:
            self.compute_normalization_stats()

        if self._state_stats["mean"].size > 0:
            for s in self.samples:
                if len(s.robot_state) > 0:
                    s.robot_state = (
                        (s.robot_state - self._state_stats["mean"])
                        / (self._state_stats["std"] + 1e-8)
                    )

    def to_tensors(self) -> dict:
        """Convert all samples to PyTorch tensors (for batch training).

        Returns:
            Dict of tensors: states, actions, grippers, etc.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for to_tensors()")

        states = np.array(
            [s.robot_state for s in self.samples],
            dtype=np.float32,
        )
        actions = np.array(
            [s.action if s.action is not None else np.zeros(self.samples[0].action.shape if self.samples[0].action is not None else 0, dtype=np.float32) for s in self.samples],
            dtype=np.float32,
        )
        grippers = np.array(
            [s.gripper for s in self.samples],
            dtype=np.float32,
        ).reshape(-1, 1)
        rewards = np.array(
            [s.reward for s in self.samples],
            dtype=np.float32,
        ).reshape(-1, 1)

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "grippers": torch.from_numpy(grippers),
            "rewards": torch.from_numpy(rewards),
        }

    def get_episode_boundaries(self) -> list[tuple[int, int]]:
        """Return (start, end) indices for each episode."""
        boundaries = []
        current_ep = None
        start = 0
        for i, s in enumerate(self.samples):
            if s.episode_id != current_ep:
                if current_ep is not None:
                    boundaries.append((start, i))
                current_ep = s.episode_id
                start = i
        if current_ep is not None:
            boundaries.append((start, len(self.samples)))
        return boundaries

    @staticmethod
    def _tokenize_instruction(instruction: str, max_len: int) -> np.ndarray:
        """Simple hash-based tokenization (no external tokenizer needed).

        Maps each character to a token ID via hash, truncated to max_len.
        For production, replace with BPE tokenizer (e.g., from transformers).
        """
        token_ids = []
        for char in instruction[:max_len]:
            token_ids.append(hash(char) % 10000)
        # Pad to max_len
        while len(token_ids) < max_len:
            token_ids.append(0)
        return np.array(token_ids, dtype=np.int64)

    def summary(self) -> dict:
        """Return dataset summary statistics."""
        n = len(self.samples)
        if n == 0:
            return {"num_samples": 0}

        state_dims = set(len(s.robot_state) for s in self.samples)
        action_dims = set(len(s.action) for s in self.samples if s.action is not None)
        episodes = set(s.episode_id for s in self.samples)
        instructions = set(s.instruction for s in self.samples)

        return {
            "num_samples": n,
            "num_episodes": len(episodes),
            "num_unique_instructions": len(instructions),
            "state_dims": list(state_dims),
            "action_dims": list(action_dims),
            "avg_reward": float(np.mean([s.reward for s in self.samples])),
        }
