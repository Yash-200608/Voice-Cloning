"""Speaker embedding generation, validation, and caching."""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path

import numpy as np

from ..core.exceptions import EmbeddingGenerationError, InvalidVoiceIdentity, MissingEmbedding
from ..core.models import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

EMBEDDING_FILENAME = "speaker.npy"
EXPECTED_EMBEDDING_DIM = 256


def get_embedding_version() -> str:
    try:
        return importlib.metadata.version("resemblyzer")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class EmbeddingStore:
    """Manages persisted speaker embeddings for voice identities."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model = embedding_model

    def embedding_path_for(self, identity_dir: Path) -> Path:
        return identity_dir / "embeddings" / EMBEDDING_FILENAME

    def generate(self, processed_audio: Path, output_path: Path) -> np.ndarray:
        """Generate and persist embedding from processed reference audio."""
        from ..similarity import embed

        try:
            vector = embed(str(processed_audio))
        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to generate embedding: {e}",
                user_message="Could not create speaker embedding from reference audio.",
            ) from e

        self._validate_vector(vector)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".tmp.npy")
        np.save(tmp_path, vector, allow_pickle=False)
        tmp_path.replace(output_path)
        return vector

    def load(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise MissingEmbedding(f"Embedding not found: {path}")
        try:
            vector = np.load(path, allow_pickle=False)
        except Exception as e:
            raise InvalidVoiceIdentity(f"Corrupt embedding file: {e}") from e
        self._validate_vector(vector)
        return vector

    def validate(
        self,
        path: Path,
        *,
        embedding_model: str | None = None,
        embedding_version: str | None = None,
    ) -> np.ndarray:
        """Load and validate an embedding against expected model/version."""
        vector = self.load(path)
        if embedding_model and embedding_model != self.embedding_model:
            raise InvalidVoiceIdentity(
                f"Embedding model mismatch: expected {self.embedding_model}, got {embedding_model}"
            )
        if embedding_version and embedding_version != get_embedding_version():
            raise InvalidVoiceIdentity(
                f"Embedding version mismatch: expected {get_embedding_version()}, got {embedding_version}"
            )
        return vector

    def _validate_vector(self, vector: np.ndarray) -> None:
        if not isinstance(vector, np.ndarray):
            raise InvalidVoiceIdentity("Embedding must be a numpy array")
        if vector.ndim != 1:
            raise InvalidVoiceIdentity(f"Embedding must be 1-D, got shape {vector.shape}")
        if vector.shape[0] != EXPECTED_EMBEDDING_DIM:
            raise InvalidVoiceIdentity(
                f"Unexpected embedding dimension: {vector.shape[0]} (expected {EXPECTED_EMBEDDING_DIM})"
            )
        if not np.all(np.isfinite(vector)):
            raise InvalidVoiceIdentity("Embedding contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise InvalidVoiceIdentity("Embedding has zero norm")
