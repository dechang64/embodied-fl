# ── python/analysis/vla_model.py ──
"""
VLA (Vision-Language-Action) Federated Model for Embodied Intelligence
======================================================================
Multi-modal transformer that fuses vision, language, and robot state
to predict discrete action tokens. Designed for federated training
across heterogeneous robot platforms.

Architecture:
  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
  │  DINOv2     │   │  Lang Encoder │   │ Robot State  │
  │  (frozen)   │   │  (trainable)  │   │  Projector   │
  │  768-dim    │   │  d_model-dim  │   │  state_dim   │
  └──────┬──────┘   └──────┬───────┘   └──────┬───────┘
         │                 │                   │
    Linear(768,d)     Linear(d,d)         Linear(s,d)
         │                 │                   │
         └────────────┬────┴───────────────────┘
                      │
              Cross-Attention Fusion
                      │
              ┌───────┴───────┐
              │  Action Head  │
              │  (autoreg.)   │
              │  → token IDs  │
              └───────────────┘

Federated Learning Strategy:
  - Shared (aggregated via FedAvg):
      * Vision Projector (DINOv2 → d_model)
      * Language Projector (if lightweight encoder)
      * Fusion layers (cross-attention)
  - Local (stays on each client):
      * Action Head (different action spaces per robot)
      * Language backbone (if large LLM, too expensive to aggregate)
  - Task matching via HNSW on instruction embeddings

Bridge to Rust:
  Python trains locally → uploads projector + fusion weights via gRPC
  → Rust aggregates via TaskAware FedAvg → distributes global model
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from .action_tokenizer import ActionTokenizer, TokenizerConfig


@dataclass
class VLAConfig:
    """Configuration for VLA model."""
    # Dimensions
    vision_dim: int = 768           # DINOv2 output dim
    lang_dim: int = 384             # Language encoder dim (e.g., all-MiniLM-L6-v2)
    state_dim: int = 8              # Robot state dim (7 joints + 1 gripper)
    d_model: int = 256              # Internal fusion dimension
    n_heads: int = 4                # Attention heads
    n_fusion_layers: int = 2        # Cross-attention fusion layers
    action_dim: int = 8             # Action output dim (joints + gripper)
    num_action_bins: int = 256      # Quantization bins per action dim
    max_instruction_len: int = 32   # Max instruction token length
    # Training
    lr: float = 1e-4
    local_epochs: int = 3
    batch_size: int = 16
    # Federated
    freeze_vision: bool = True      # DINOv2 is frozen
    freeze_lang_backbone: bool = True  # Freeze large language backbone
    share_fusion: bool = True       # Aggregate fusion layers


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class VisionProjector(nn.Module):
    """Project DINOv2 features to fusion dimension."""

    def __init__(self, vision_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, vision_dim) or (B, T, vision_dim)
        Returns:
            (B, 1, d_model) or (B, T, d_model)
        """
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(1)
        return self.proj(vision_features)


class LanguageProjector(nn.Module):
    """Project language embeddings to fusion dimension."""

    def __init__(self, lang_dim: int, d_model: int, max_len: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(lang_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len)

    def forward(
        self,
        lang_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            lang_embeddings: (B, L, lang_dim)
            attention_mask: (B, L) bool, True = valid token
        Returns:
            (B, L, d_model)
        """
        projected = self.proj(lang_embeddings)
        projected = self.pos_enc(projected)

        if attention_mask is not None:
            # Zero out padding positions
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            projected = projected * mask

        return projected


class StateProjector(nn.Module):
    """Project robot state to fusion dimension."""

    def __init__(self, state_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, robot_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            robot_state: (B, state_dim)
        Returns:
            (B, 1, d_model)
        """
        return self.proj(robot_state).unsqueeze(1)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion of vision, language, and state.

    Vision acts as query, language+state as key/value.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        vision: torch.Tensor,
        lang: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision: (B, 1, d_model)
            lang: (B, L, d_model)
            state: (B, 1, d_model)
        Returns:
            (B, d_model) — fused representation
        """
        # Concatenate lang + state as context
        context = torch.cat([lang, state], dim=1)  # (B, L+1, d_model)

        # Use vision as query, context as memory
        x = vision  # (B, 1, d_model)
        for layer in self.layers:
            # Self-attention on x, then cross-attention with context
            x = layer(x)  # Self-attention
            # Manual cross-attention
            attn_out = F.scaled_dot_product_attention(
                x, context, context,
            )
            x = x + attn_out
            x = F.layer_norm(x, x.shape[-1:])

        return x.squeeze(1)  # (B, d_model)


class ActionHead(nn.Module):
    """Predict discrete action tokens from fused representation.

    Outputs logits for each action dimension independently.
    """

    def __init__(self, d_model: int, action_dim: int, num_bins: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        # Shared trunk + per-dimension heads
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # Per-dimension prediction heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model, num_bins)
            for _ in range(action_dim)
        ])

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, d_model)
        Returns:
            (B, action_dim, num_bins) logits
        """
        trunk_out = self.trunk(fused)  # (B, d_model)
        logits = torch.stack(
            [head(trunk_out) for head in self.heads],
            dim=1,
        )  # (B, action_dim, num_bins)
        return logits

    def predict_tokens(self, fused: torch.Tensor) -> torch.Tensor:
        """Predict most likely action tokens.

        Args:
            fused: (B, d_model)
        Returns:
            (B, action_dim) int64 token IDs
        """
        logits = self.forward(fused)  # (B, action_dim, num_bins)
        return logits.argmax(dim=-1)  # (B, action_dim)


class VLAFLModel(nn.Module):
    """Vision-Language-Action model for federated training.

    Designed to be split into shared (aggregated) and local (per-client)
    components for FedAvg.

    Usage:
        config = VLAConfig()
        model = VLAFLModel(config)

        # Forward pass
        vision_feats = torch.randn(4, 768)       # DINOv2 features
        lang_embeds = torch.randn(4, 16, 384)    # Language embeddings
        robot_state = torch.randn(4, 8)           # Joint angles
        action_tokens = torch.randint(0, 256, (4, 8))  # Target actions

        logits = model(vision_feats, lang_embeds, robot_state)
        loss = model.compute_loss(logits, action_tokens)

        # Get shared parameters for FedAvg
        shared_params = model.get_shared_params()
        local_params = model.get_local_params()
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        super().__init__()
        self.config = config or VLAConfig()
        cfg = self.config

        # Projectors (shared — aggregated via FedAvg)
        self.vision_projector = VisionProjector(cfg.vision_dim, cfg.d_model)
        self.lang_projector = LanguageProjector(
            cfg.lang_dim, cfg.d_model, cfg.max_instruction_len
        )
        self.state_projector = StateProjector(cfg.state_dim, cfg.d_model)

        # Fusion layers (shared — aggregated via FedAvg)
        self.fusion = CrossAttentionFusion(
            cfg.d_model, cfg.n_heads, cfg.n_fusion_layers
        )

        # Action head (local — stays on each client)
        self.action_head = ActionHead(
            cfg.d_model, cfg.action_dim, cfg.num_action_bins
        )

        # Action tokenizer (shared config, no learnable params)
        self.tokenizer = ActionTokenizer(
            TokenizerConfig(
                action_dim=cfg.action_dim,
                num_bins=cfg.num_action_bins,
            )
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        lang_embeddings: torch.Tensor,
        robot_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            vision_features: (B, vision_dim) DINOv2 features.
            lang_embeddings: (B, L, lang_dim) Language embeddings.
            robot_state: (B, state_dim) Robot joint state.
            attention_mask: (B, L) bool, True = valid token.

        Returns:
            (B, action_dim, num_bins) action logits.
        """
        # Project to common dimension
        vision_proj = self.vision_projector(vision_features)    # (B, 1, d_model)
        lang_proj = self.lang_projector(lang_embeddings, attention_mask)  # (B, L, d_model)
        state_proj = self.state_projector(robot_state)         # (B, 1, d_model)

        # Fuse
        fused = self.fusion(vision_proj, lang_proj, state_proj)  # (B, d_model)

        # Predict actions
        logits = self.action_head(fused)  # (B, action_dim, num_bins)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for action prediction.

        Args:
            logits: (B, action_dim, num_bins) predicted logits.
            target_tokens: (B, action_dim) ground truth token IDs.

        Returns:
            Scalar loss.
        """
        # Cross-entropy per action dimension, then average
        # logits: (B, action_dim, num_bins) → (B * action_dim, num_bins)
        # targets: (B, action_dim) → (B * action_dim,)
        B, A, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * A, C),
            target_tokens.reshape(B * A),
        )
        return loss

    def predict(
        self,
        vision_features: torch.Tensor,
        lang_embeddings: torch.Tensor,
        robot_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Predict action tokens (inference mode).

        Args:
            Same as forward().

        Returns:
            (B, action_dim) int64 numpy array of action tokens.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                vision_features, lang_embeddings, robot_state, attention_mask
            )
            tokens = logits.argmax(dim=-1)  # (B, action_dim)
        return tokens.cpu().numpy()

    def predict_actions(
        self,
        vision_features: torch.Tensor,
        lang_embeddings: torch.Tensor,
        robot_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Predict continuous actions (decoded from tokens).

        Args:
            Same as forward().

        Returns:
            (B, action_dim) float32 numpy array of continuous actions.
        """
        tokens = self.predict(
            vision_features, lang_embeddings, robot_state, attention_mask
        )
        return self.tokenizer.decode_batch(tokens)

    def get_shared_params(self) -> dict[str, nn.Parameter]:
        """Get parameters that should be aggregated via FedAvg.

        Returns:
            Dict of {name: parameter} for shared components.
        """
        shared = {}
        for name, param in self.vision_projector.named_parameters():
            shared[f"vision_projector.{name}"] = param
        for name, param in self.lang_projector.named_parameters():
            shared[f"lang_projector.{name}"] = param
        for name, param in self.state_projector.named_parameters():
            shared[f"state_projector.{name}"] = param
        for name, param in self.fusion.named_parameters():
            shared[f"fusion.{name}"] = param
        return shared

    def get_local_params(self) -> dict[str, nn.Parameter]:
        """Get parameters that stay local (per-client).

        Returns:
            Dict of {name: parameter} for local components.
        """
        local = {}
        for name, param in self.action_head.named_parameters():
            local[f"action_head.{name}"] = param
        return local

    def load_shared_params(self, state_dict: dict):
        """Load shared parameters from aggregated global model.

        Args:
            state_dict: Dict of {name: parameter} from server.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)

    def get_shared_state_dict(self) -> dict:
        """Get shared parameters as state dict for upload.

        Returns:
            Dict of {name: tensor} for shared components.
        """
        shared_names = set(self.get_shared_params().keys())
        return {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if name in shared_names
        }

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        for name, module in [
            ("vision_projector", self.vision_projector),
            ("lang_projector", self.lang_projector),
            ("state_projector", self.state_projector),
            ("fusion", self.fusion),
            ("action_head", self.action_head),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        counts["shared"] = sum(
            p.numel() for p in self.get_shared_params().values()
        )
        counts["local"] = sum(
            p.numel() for p in self.get_local_params().values()
        )
        return counts


class VLAFLTrainer:
    """Federated trainer for VLA model.

    Manages local training loop and FedAvg-compatible parameter exchange.

    Usage:
        config = VLAConfig(action_dim=8, state_dim=8)
        trainer = VLAFLTrainer(config)

        # Local training
        metrics = trainer.train_local(
            vision_feats, lang_embeds, robot_states, action_tokens,
            n_epochs=3,
        )

        # Get shared params for upload
        shared = trainer.model.get_shared_state_dict()

        # Load aggregated global model
        trainer.model.load_shared_params(global_shared_params)
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        self.config = config or VLAConfig()
        self.model = VLAFLModel(self.config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=1e-4,
        )

    def train_local(
        self,
        vision_features: torch.Tensor,
        lang_embeddings: torch.Tensor,
        robot_states: torch.Tensor,
        action_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_epochs: Optional[int] = None,
    ) -> dict:
        """Run local training epochs.

        Args:
            vision_features: (N, vision_dim)
            lang_embeddings: (N, L, lang_dim)
            robot_states: (N, state_dim)
            action_tokens: (N, action_dim)
            attention_mask: (N, L) optional
            n_epochs: Override config.local_epochs

        Returns:
            Dict with loss history and final metrics.
        """
        epochs = n_epochs or self.config.local_epochs
        self.model.train()

        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            logits = self.model(
                vision_features, lang_embeddings, robot_states, attention_mask
            )
            loss = self.model.compute_loss(logits, action_tokens)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            losses.append(loss.item())

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                vision_features, lang_embeddings, robot_states, attention_mask
            )
            pred_tokens = logits.argmax(dim=-1)
            accuracy = (pred_tokens == action_tokens).float().mean().item()

        return {
            "losses": losses,
            "final_loss": losses[-1],
            "accuracy": accuracy,
            "epochs": epochs,
        }

    def get_upload_payload(self) -> dict:
        """Get payload for uploading to federated server.

        Returns:
            Dict with shared_params and metadata.
        """
        return {
            "shared_params": {
                k: v.numpy() for k, v in self.model.get_shared_state_dict().items()
            },
            "param_count": self.model.count_parameters(),
        }
