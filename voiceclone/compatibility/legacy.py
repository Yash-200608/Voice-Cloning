"""Legacy flat WAV voice migration."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from ..audio_utils import audio_info, preprocess_reference
from ..core.exceptions import VoiceRepositoryError
from ..identity.embeddings import EmbeddingStore
from ..identity.metadata import save_metadata
from ..identity.repository import (
    METADATA_FILENAME,
    PROCESSED_REFERENCE,
    RAW_REFERENCE,
    StagingContext,
    VoiceRepository,
    generate_voice_id,
)

logger = logging.getLogger(__name__)

MIGRATION_INDEX = ".legacy_migration.json"


def _migration_index_path(voices_dir: Path) -> Path:
    return voices_dir / MIGRATION_INDEX


def _load_migration_index(voices_dir: Path) -> dict[str, str]:
    path = _migration_index_path(voices_dir)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_migration_index(voices_dir: Path, index: dict[str, str]) -> None:
    path = _migration_index_path(voices_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def _legacy_wav_files(voices_dir: Path) -> list[Path]:
    return sorted(voices_dir.glob("*.wav"))


def migrate_legacy_voices(repository: VoiceRepository) -> list[str]:
    """
    Lazily migrate flat legacy .wav voices to identity directories.

    Returns list of migrated identity IDs. Never modifies or deletes legacy
    source files on failure.
    """
    voices_dir = repository.voices_dir
    migration_index = _load_migration_index(voices_dir)
    migrated_ids: list[str] = []

    for legacy_path in _legacy_wav_files(voices_dir):
        legacy_key = legacy_path.name
        if legacy_key in migration_index:
            continue

        name = legacy_path.stem
        identity_id = generate_voice_id()
        staging_dir = voices_dir / f".staging_{identity_id}"

        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True)

        try:
            staged = StagingContext(
                identity_id=identity_id,
                name=name,
                staging_dir=staging_dir,
                repository=repository,
            )
            staged.raw_path.parent.mkdir(parents=True, exist_ok=True)
            staged.processed_path.parent.mkdir(parents=True, exist_ok=True)
            staged.embedding_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(legacy_path, staged.raw_path)
            preprocess_reference(staged.raw_path, out_path=staged.processed_path)

            info = audio_info(staged.processed_path)
            embedding_store = repository.embedding_store
            embedding_store.generate(staged.processed_path, staged.embedding_path)

            identity = staged.build_identity(
                sample_rate=info["sample_rate"],
                duration_seconds=info["duration_seconds"],
                quality_metrics={"migrated_from": legacy_key},
            )
            save_metadata(staging_dir / METADATA_FILENAME, identity)

            final_dir = repository.identity_dir(identity_id)
            if final_dir.exists():
                raise VoiceRepositoryError(f"Identity directory already exists: {identity_id}")
            staging_dir.rename(final_dir)

            migration_index[legacy_key] = identity_id
            _save_migration_index(voices_dir, migration_index)
            migrated_ids.append(identity_id)
            logger.info("Migrated legacy voice %s -> %s", legacy_key, identity_id)

        except Exception as e:
            logger.warning("Failed to migrate legacy voice %s: %s", legacy_key, e)
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            # Legacy source file is never modified or deleted on failure.

    return migrated_ids
