"""Voice identity persistence repository."""

from __future__ import annotations

import logging
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..Config import OUTPUTS_DIR, VOICES_DIR
from ..core.exceptions import (
    InvalidVoiceIdentity,
    VoiceIdentityNotFound,
    VoiceRepositoryError,
)
from ..core.models import (
    EMBEDDING_MODEL,
    RENDERER_MODEL,
    VoiceIdentity,
    format_timestamp,
    utc_now,
)
from .embeddings import EmbeddingStore, get_embedding_version
from .metadata import load_metadata, save_metadata

logger = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"
RAW_REFERENCE = Path("raw/reference.wav")
PROCESSED_REFERENCE = Path("processed/reference.wav")
EMBEDDING_REL = Path("embeddings/speaker.npy")


def generate_voice_id() -> str:
    return f"voice_{uuid.uuid4()}"


def get_renderer_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("chatterbox-tts")
    except Exception:
        return "unknown"


class VoiceRepository:
    """Persists and manages voice identity directories."""

    def __init__(
        self,
        voices_dir: Path | None = None,
        outputs_dir: Path | None = None,
        embedding_store: EmbeddingStore | None = None,
    ):
        self.voices_dir = Path(voices_dir or VOICES_DIR)
        self.outputs_dir = Path(outputs_dir or OUTPUTS_DIR)
        self.embedding_store = embedding_store or EmbeddingStore()
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def identity_dir(self, identity_id: str) -> Path:
        return self.voices_dir / identity_id

    def output_dir(self, identity_id: str) -> Path:
        return self.outputs_dir / identity_id

    def list_identity_dirs(self) -> list[Path]:
        return sorted(
            p for p in self.voices_dir.iterdir()
            if p.is_dir() and (p / METADATA_FILENAME).exists()
        )

    def list_identities(self) -> list[VoiceIdentity]:
        identities = []
        for identity_dir in self.list_identity_dirs():
            try:
                identities.append(self._load_from_dir(identity_dir))
            except InvalidVoiceIdentity as e:
                logger.warning("Skipping invalid identity %s: %s", identity_dir.name, e)
        return sorted(identities, key=lambda i: i.name.lower())

    def get(self, identity_id: str) -> VoiceIdentity:
        identity_dir = self.identity_dir(identity_id)
        if not identity_dir.exists():
            raise VoiceIdentityNotFound(
                f"Voice identity not found: {identity_id}",
                user_message="That voice identity was not found.",
            )
        return self._load_from_dir(identity_dir)

    def get_by_name(self, name: str) -> VoiceIdentity:
        matches = [i for i in self.list_identities() if i.name == name]
        if not matches:
            raise VoiceIdentityNotFound(
                f"No voice identity named '{name}'",
                user_message=f"No voice named '{name}' was found.",
            )
        if len(matches) > 1:
            raise VoiceRepositoryError(
                f"Multiple identities named '{name}'",
                user_message=f"Multiple voices named '{name}' exist. Please rename one.",
            )
        return matches[0]

    def rename(self, identity_id: str, new_name: str) -> VoiceIdentity:
        new_name = new_name.strip()
        if not new_name:
            raise InvalidVoiceIdentity("Voice identity name cannot be empty")

        identity = self.get(identity_id)
        identity.name = new_name
        identity.updated_at = utc_now()
        save_metadata(self.identity_dir(identity_id) / METADATA_FILENAME, identity)
        return identity

    def delete(self, identity_id: str) -> bool:
        identity_dir = self.identity_dir(identity_id)
        if not identity_dir.exists():
            return False
        shutil.rmtree(identity_dir)
        output_dir = self.output_dir(identity_id)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        return True

    @contextmanager
    def staged_creation(self, name: str):
        """Create identity in a staging directory; publish on success."""
        identity_id = generate_voice_id()
        identity_dir = self.identity_dir(identity_id)
        staging_dir = self.voices_dir / f".staging_{identity_id}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)

        try:
            yield StagingContext(
                identity_id=identity_id,
                name=name.strip(),
                staging_dir=staging_dir,
                repository=self,
            )
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def publish_staged(self, staged: "StagingContext", identity: VoiceIdentity) -> VoiceIdentity:
        """Move staged identity to final location."""
        final_dir = self.identity_dir(staged.identity_id)
        if final_dir.exists():
            raise VoiceRepositoryError(f"Identity already exists: {staged.identity_id}")
        staged.staging_dir.rename(final_dir)
        save_metadata(final_dir / METADATA_FILENAME, identity)
        return identity

    def _load_from_dir(self, identity_dir: Path) -> VoiceIdentity:
        identity = load_metadata(identity_dir / METADATA_FILENAME)
        if identity.id != identity_dir.name:
            raise InvalidVoiceIdentity(
                f"Identity ID mismatch: metadata={identity.id}, directory={identity_dir.name}"
            )
        return identity

    def resolve_path(self, identity: VoiceIdentity, rel_path: str) -> Path:
        return self.identity_dir(identity.id) / rel_path

    def ensure_files_exist(self, identity: VoiceIdentity) -> None:
        for rel in (identity.reference_audio, identity.processed_audio, identity.embedding_path):
            path = self.resolve_path(identity, rel)
            if not path.exists():
                raise InvalidVoiceIdentity(f"Missing identity file: {rel}")


class StagingContext:
    """Context for staged identity creation."""

    def __init__(self, identity_id: str, name: str, staging_dir: Path, repository: VoiceRepository):
        self.identity_id = identity_id
        self.name = name
        self.staging_dir = staging_dir
        self.repository = repository

    @property
    def raw_path(self) -> Path:
        return self.staging_dir / RAW_REFERENCE

    @property
    def processed_path(self) -> Path:
        return self.staging_dir / PROCESSED_REFERENCE

    @property
    def embedding_path(self) -> Path:
        return self.staging_dir / EMBEDDING_REL

    def build_identity(
        self,
        sample_rate: int,
        duration_seconds: float,
        quality_metrics: dict | None = None,
    ) -> VoiceIdentity:
        now = utc_now()
        return VoiceIdentity(
            id=self.identity_id,
            name=self.name,
            created_at=now,
            updated_at=now,
            reference_audio=str(RAW_REFERENCE),
            processed_audio=str(PROCESSED_REFERENCE),
            embedding_path=str(EMBEDDING_REL),
            embedding_model=EMBEDDING_MODEL,
            embedding_version=get_embedding_version(),
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            renderer_model=RENDERER_MODEL,
            renderer_version=get_renderer_version(),
            quality_metrics=quality_metrics or {},
        )
