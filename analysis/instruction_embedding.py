# ── python/analysis/instruction_embedding.py ──
"""
Instruction Embedding for VLA Task Matching
=============================================
Generates dense embeddings from natural language instructions for
HNSW-based task matching in federated learning.

Supports three modes:
  1. sentence-transformer (recommended): all-MiniLM-L6-v2, 384-dim
  2. DINOv2 visual: encode instruction as image (OCR screenshot)
  3. hash fallback: deterministic hash → pseudo-embedding (no GPU needed)

Federated Learning Integration:
  - Instruction embeddings sent to Rust server via gRPC
  - Indexed in HNSW for task similarity matching
  - Used for task-aware aggregation weighting
  - Shared across all clients for consistent task understanding

Bridge to Rust:
  Python computes embeddings → gRPC ReportTask(task_embedding=[...])
  → Rust stores in HNSW → TaskMatcher finds similar tasks
"""

import numpy as np
import hashlib
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for instruction embedding."""
    mode: str = "hash"           # "sentence_transformer", "dinov2_visual", "hash"
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384         # Output embedding dimension
    normalize: bool = True       # L2 normalize output
    batch_size: int = 32
    device: str = "cpu"          # "cpu" or "cuda"


class InstructionEmbedder:
    """Generate embeddings from natural language robot instructions.

    Usage:
        embedder = InstructionEmbedder(EmbeddingConfig(mode="hash"))
        vec = embedder.embed("pick up the red cup")  # (384,) float32

        # Batch
        vecs = embedder.embed_batch(["pick up cup", "go to door"])
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None

    def embed(self, instruction: str) -> np.ndarray:
        """Embed a single instruction.

        Args:
            instruction: Natural language instruction string.

        Returns:
            (dimension,) float32 embedding vector.
        """
        if not instruction or not instruction.strip():
            return np.zeros(self.config.dimension, dtype=np.float32)

        if self.config.mode == "sentence_transformer":
            return self._embed_sentence_transformer(instruction)
        elif self.config.mode == "dinov2_visual":
            return self._embed_dinov2_visual(instruction)
        elif self.config.mode == "hash":
            return self._embed_hash(instruction)
        else:
            raise ValueError(f"Unknown embedding mode: {self.config.mode}")

    def embed_batch(self, instructions: list[str]) -> np.ndarray:
        """Embed multiple instructions.

        Args:
            instructions: List of instruction strings.

        Returns:
            (N, dimension) float32 embedding matrix.
        """
        if not instructions:
            return np.zeros((0, self.config.dimension), dtype=np.float32)

        if self.config.mode == "sentence_transformer":
            return self._embed_batch_sentence_transformer(instructions)
        elif self.config.mode == "hash":
            return np.array(
                [self._embed_hash(inst) for inst in instructions],
                dtype=np.float32,
            )
        else:
            # Fallback to per-item
            return np.array(
                [self.embed(inst) for inst in instructions],
                dtype=np.float32,
            )

    def _embed_sentence_transformer(self, instruction: str) -> np.ndarray:
        """Embed using sentence-transformers (all-MiniLM-L6-v2).

        Lazy-loads the model on first call.
        """
        self._ensure_sentence_transformer()
        from sentence_transformers import SentenceTransformer

        embedding = self._model.encode(
            [instruction],
            batch_size=1,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return embedding[0].astype(np.float32)

    def _embed_batch_sentence_transformer(self, instructions: list[str]) -> np.ndarray:
        """Batch embed using sentence-transformers."""
        self._ensure_sentence_transformer()
        from sentence_transformers import SentenceTransformer

        embeddings = self._model.encode(
            instructions,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def _ensure_sentence_transformer(self):
        """Lazy-load sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                )
                # Update dimension from model
                self.config.dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def _embed_dinov2_visual(self, instruction: str) -> np.ndarray:
        """Embed instruction by rendering it as text image and using DINOv2.

        This is a creative fallback: render instruction as an image,
        then extract DINOv2 features. Useful when no text encoder is available
        but DINOv2 is already loaded for visual features.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import torch
        except ImportError:
            raise ImportError("PIL and torch required for visual embedding mode")

        # Render instruction as white-on-black image
        img_size = (384, 64)
        img = Image.new("RGB", img_size, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Try to use a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Word wrap
        words = instruction.split()
        lines = []
        current_line = ""
        for word in words:
            test = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] > img_size[0] - 10:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test
        if current_line:
            lines.append(current_line)

        y = 5
        for line in lines[:3]:  # Max 3 lines
            draw.text((5, y), line, fill=(255, 255, 255), font=font)
            y += 20

        # Extract DINOv2 features
        from torchvision import transforms
        from .feature_extractor import DINOv2SceneExtractor

        extractor = DINOv2SceneExtractor(model_name="vits14")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = extractor(tensor).squeeze(0).numpy()

        # Pad or truncate to target dimension
        dim = self.config.dimension
        if features.shape[0] >= dim:
            features = features[:dim]
        else:
            padded = np.zeros(dim, dtype=np.float32)
            padded[:features.shape[0]] = features
            features = padded

        if self.config.normalize:
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features /= norm

        return features.astype(np.float32)

    def _embed_hash(self, instruction: str) -> np.ndarray:
        """Deterministic hash-based pseudo-embedding.

        No external dependencies needed. Produces a fixed-dimension vector
        from the instruction text using SHA-256 hashing.

        Quality: Low semantic quality, but deterministic and reproducible.
        Use for testing or when no GPU/model is available.
        """
        dim = self.config.dimension
        normalized = instruction.strip().lower()
        embedding = np.zeros(dim, dtype=np.float32)

        # Generate multiple hash chunks to fill the dimension
        chunk_size = 4  # Each hash produces 4 floats
        for i in range(0, dim, chunk_size):
            # Vary the seed per chunk for different values
            chunk_text = f"{normalized}__chunk_{i // chunk_size}"
            hash_bytes = hashlib.sha256(chunk_text.encode("utf-8")).digest()

            for j in range(min(chunk_size, dim - i)):
                # Convert 2 bytes to float in [-1, 1]
                val = (hash_bytes[j * 2] + hash_bytes[j * 2 + 1] * 256) / 65535.0 * 2.0 - 1.0
                embedding[i + j] = val

        if self.config.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding /= norm

        return embedding

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(dot / (norm1 * norm2))

    def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """Find the most similar instructions to a query.

        Args:
            query: Query instruction.
            candidates: List of candidate instructions.
            top_k: Number of results to return.

        Returns:
            List of (index, instruction, similarity) tuples, sorted by similarity.
        """
        if not candidates:
            return []

        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)

        similarities = [
            self.similarity(query_emb, candidate_embs[i])
            for i in range(len(candidates))
        ]

        ranked = sorted(
            enumerate(zip(candidates, similarities)),
            key=lambda x: x[1][1],
            reverse=True,
        )[:top_k]

        return [(idx, inst, sim) for idx, (inst, sim) in ranked]


def get_embedder(mode: str = "hash", **kwargs) -> InstructionEmbedder:
    """Factory: get instruction embedder by mode.

    Args:
        mode: "hash", "sentence_transformer", or "dinov2_visual"
        **kwargs: Additional config parameters.

    Returns:
        InstructionEmbedder instance.
    """
    config = EmbeddingConfig(mode=mode, **kwargs)
    return InstructionEmbedder(config)
