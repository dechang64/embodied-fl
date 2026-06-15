# ── python/analysis/action_tokenizer.py ──
"""
Action Tokenizer for VLA Models
=================================
Converts continuous robot actions into discrete tokens and back.

This is the key bridge between continuous control and language-model-style
autoregressive generation. Following RT-2 / Octo / π0 approach:

  Continuous action (e.g., 7 joint angles + 1 gripper = 8D)
    → Quantize each dimension to N bins (default 256)
    → Each bin becomes a token ID
    → Action sequence = sequence of token IDs
    → Can be processed by Transformer / LLM architecture

Federated Learning Integration:
  - Tokenizer is shared across all clients (same bin boundaries)
  - Enables FedAvg on the action prediction head
  - Different robots can have different action_dim but same tokenizer config
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    """Configuration for action tokenizer."""
    action_dim: int = 8              # Total action dimensions (joints + gripper)
    num_bins: int = 256              # Quantization bins per dimension
    low: float = -1.0                # Lower bound for all dimensions
    high: float = 1.0                # Upper bound for all dimensions
    # Per-dimension bounds (optional, overrides low/high)
    per_dim_low: Optional[list[float]] = None
    per_dim_high: Optional[list[float]] = None
    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1            # End of sequence
    sos_token_id: int = 2            # Start of sequence

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self.num_bins + 3  # bins + pad + eos + sos

    @property
    def max_action_tokens(self) -> int:
        """Maximum tokens per action (one per dimension)."""
        return self.action_dim

    def to_dict(self) -> dict:
        """Serialize config to dict."""
        return {
            "action_dim": self.action_dim,
            "num_bins": self.num_bins,
            "low": self.low,
            "high": self.high,
            "per_dim_low": self.per_dim_low,
            "per_dim_high": self.per_dim_high,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "sos_token_id": self.sos_token_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenizerConfig":
        """Deserialize config from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ActionTokenizer:
    """Bidirectional tokenizer: continuous actions ↔ discrete tokens.

    Usage:
        tokenizer = ActionTokenizer(TokenizerConfig(action_dim=8, num_bins=256))

        # Encode
        action = np.array([0.1, -0.3, 0.5, 0.0, -0.2, 0.8, -0.1, 0.5])
        tokens = tokenizer.encode(action)  # [103, 51, 179, 128, 77, 230, 102, 128]

        # Decode
        recovered = tokenizer.decode(tokens)  # approximately [0.1, -0.3, ...]

        # Encode batch
        actions = np.random.randn(32, 8).astype(np.float32)
        tokens_batch = tokenizer.encode_batch(actions)  # (32, 8)
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._validate()

        # Precompute bin edges for each dimension
        self._bin_edges = []
        for d in range(self.config.action_dim):
            lo = self.config.per_dim_low[d] if self.config.per_dim_low else self.config.low
            hi = self.config.per_dim_high[d] if self.config.per_dim_high else self.config.high
            edges = np.linspace(lo, hi, self.config.num_bins + 1, dtype=np.float32)
            self._bin_edges.append(edges)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self.config.vocab_size

    def _validate(self):
        cfg = self.config
        if cfg.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {cfg.action_dim}")
        if cfg.num_bins <= 0:
            raise ValueError(f"num_bins must be > 0, got {cfg.num_bins}")
        if cfg.per_dim_low and len(cfg.per_dim_low) != cfg.action_dim:
            raise ValueError(
                f"per_dim_low length {len(cfg.per_dim_low)} != action_dim {cfg.action_dim}"
            )
        if cfg.per_dim_high and len(cfg.per_dim_high) != cfg.action_dim:
            raise ValueError(
                f"per_dim_high length {len(cfg.per_dim_high)} != action_dim {cfg.action_dim}"
            )

    def encode(self, action: np.ndarray) -> np.ndarray:
        """Encode a single continuous action to discrete tokens.

        Args:
            action: (action_dim,) float32 array.

        Returns:
            (action_dim,) int64 array of token IDs.
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.config.action_dim,):
            raise ValueError(
                f"Expected shape ({self.config.action_dim},), got {action.shape}"
            )

        tokens = np.zeros(self.config.action_dim, dtype=np.int64)
        for d in range(self.config.action_dim):
            # Clip to bin range
            clipped = np.clip(action[d], self._bin_edges[d][0], self._bin_edges[d][-1])
            # Find bin index
            bin_idx = np.searchsorted(self._bin_edges[d], clipped, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, self.config.num_bins - 1)
            tokens[d] = int(bin_idx) + 3  # +3 for special tokens offset

        return tokens

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode discrete tokens back to continuous action.

        Args:
            tokens: (action_dim,) int64 array of token IDs.

        Returns:
            (action_dim,) float32 array.
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.shape != (self.config.action_dim,):
            raise ValueError(
                f"Expected shape ({self.config.action_dim},), got {tokens.shape}"
            )

        action = np.zeros(self.config.action_dim, dtype=np.float32)
        for d in range(self.config.action_dim):
            bin_idx = int(tokens[d]) - 3  # Remove special token offset
            bin_idx = np.clip(bin_idx, 0, self.config.num_bins - 1)
            # Use bin center as the continuous value
            action[d] = (self._bin_edges[d][bin_idx] + self._bin_edges[d][bin_idx + 1]) / 2.0

        return action

    def encode_batch(self, actions: np.ndarray) -> np.ndarray:
        """Encode a batch of actions.

        Args:
            actions: (batch, action_dim) float32 array.

        Returns:
            (batch, action_dim) int64 array.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.config.action_dim:
            raise ValueError(
                f"Expected shape (N, {self.config.action_dim}), got {actions.shape}"
            )
        return np.array([self.encode(a) for a in actions], dtype=np.int64)

    def decode_batch(self, tokens: np.ndarray) -> np.ndarray:
        """Decode a batch of tokens.

        Args:
            tokens: (batch, action_dim) int64 array.

        Returns:
            (batch, action_dim) float32 array.
        """
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 2 or tokens.shape[1] != self.config.action_dim:
            raise ValueError(
                f"Expected shape (N, {self.config.action_dim}), got {tokens.shape}"
            )
        return np.array([self.decode(t) for t in tokens], dtype=np.float32)

    def encode_trajectory(
        self, trajectory: np.ndarray, add_sos_eos: bool = True
    ) -> np.ndarray:
        """Encode a full action trajectory (sequence of actions).

        Args:
            trajectory: (T, action_dim) float32 array.
            add_sos_eos: Prepend SOS and append EOS tokens.

        Returns:
            (T * action_dim + 2,) int64 array if add_sos_eos, else (T * action_dim,).
        """
        tokens_per_step = self.encode_batch(trajectory)  # (T, action_dim)
        flat = tokens_per_step.reshape(-1)  # (T * action_dim,)

        if add_sos_eos:
            flat = np.concatenate([
                [self.config.sos_token_id],
                flat,
                [self.config.eos_token_id],
            ])
        return flat

    def decode_trajectory(
        self, tokens: np.ndarray, has_sos_eos: bool = True
    ) -> np.ndarray:
        """Decode a token sequence back to action trajectory.

        Args:
            tokens: 1D int64 array.
            has_sos_eos: Whether tokens include SOS/EOS.

        Returns:
            (T, action_dim) float32 array.
        """
        tokens = np.asarray(tokens, dtype=np.int64)

        if has_sos_eos:
            # Strip SOS and everything after EOS
            if tokens[0] == self.config.sos_token_id:
                tokens = tokens[1:]
            eos_mask = tokens == self.config.eos_token_id
            if eos_mask.any():
                tokens = tokens[:eos_mask.argmax()]

        # Reshape into (T, action_dim)
        total_action_tokens = len(tokens)
        if total_action_tokens % self.config.action_dim != 0:
            # Truncate to nearest multiple
            total_action_tokens = (total_action_tokens // self.config.action_dim) * self.config.action_dim
            tokens = tokens[:total_action_tokens]

        T = total_action_tokens // self.config.action_dim
        if T == 0:
            return np.zeros((0, self.config.action_dim), dtype=np.float32)

        tokens_2d = tokens.reshape(T, self.config.action_dim)
        return self.decode_batch(tokens_2d)

    def quantization_error(self, actions: np.ndarray) -> dict:
        """Compute quantization error for a batch of actions.

        Args:
            actions: (N, action_dim) float32 array.

        Returns:
            Dict with MSE, MAE, max_error per dimension.
        """
        tokens = self.encode_batch(actions)
        recovered = self.decode_batch(tokens)
        diff = actions - recovered

        return {
            "mse": float(np.mean(diff ** 2)),
            "mae": float(np.mean(np.abs(diff))),
            "max_error": float(np.max(np.abs(diff))),
            "per_dim_mse": [float(np.mean(diff[:, d] ** 2)) for d in range(self.config.action_dim)],
        }

    def get_config_dict(self) -> dict:
        """Serialize config to dict (for saving/loading)."""
        cfg = self.config
        return {
            "action_dim": cfg.action_dim,
            "num_bins": cfg.num_bins,
            "low": cfg.low,
            "high": cfg.high,
            "per_dim_low": cfg.per_dim_low,
            "per_dim_high": cfg.per_dim_high,
            "pad_token_id": cfg.pad_token_id,
            "eos_token_id": cfg.eos_token_id,
            "sos_token_id": cfg.sos_token_id,
        }

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "ActionTokenizer":
        """Create tokenizer from serialized config dict."""
        config = TokenizerConfig(**{k: v for k, v in config_dict.items() if v is not None})
        return cls(config)


class DeltaActionTokenizer(ActionTokenizer):
    """Extended tokenizer that encodes delta (relative) actions.

    Instead of absolute joint positions, encodes the change from current state.
    This often gives better generalization in VLA models.

    Usage:
        tokenizer = DeltaActionTokenizer(TokenizerConfig(action_dim=8))
        current_state = np.array([0.1, 0.2, ...])
        target_action = np.array([0.15, 0.25, ...])
        delta = target_action - current_state
        tokens = tokenizer.encode(delta)
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        # Delta actions typically have smaller range
        if config is None:
            config = TokenizerConfig(
                action_dim=8,
                num_bins=256,
                low=-0.5,
                high=0.5,
            )
        super().__init__(config)

    def encode_delta(
        self, current_state: np.ndarray, target_action: np.ndarray
    ) -> np.ndarray:
        """Encode the delta between current state and target action.

        Args:
            current_state: (action_dim,) current robot state.
            target_action: (action_dim,) target action.

        Returns:
            (action_dim,) int64 token array.
        """
        delta = target_action - current_state
        return self.encode(delta)

    def decode_delta(
        self, tokens: np.ndarray, current_state: np.ndarray
    ) -> np.ndarray:
        """Decode tokens to absolute action given current state.

        Args:
            tokens: (action_dim,) int64 token array.
            current_state: (action_dim,) current robot state.

        Returns:
            (action_dim,) float32 absolute action.
        """
        delta = self.decode(tokens)
        return current_state + delta
