"""Metadata serialization and validation for voice identities."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..core.exceptions import InvalidVoiceIdentity, UnsupportedIdentityVersion
from ..core.models import SCHEMA_VERSION, VOICE_ID_PATTERN, VoiceIdentity

REQUIRED_FIELDS = {
    "id",
    "name",
    "created_at",
    "updated_at",
    "reference_audio",
    "processed_audio",
    "embedding",
    "embedding_model",
    "embedding_version",
    "sample_rate",
    "duration_seconds",
    "renderer_model",
    "renderer_version",
    "schema_version",
}


def validate_metadata_dict(data: dict[str, Any]) -> None:
    """Validate metadata dictionary structure and values."""
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise InvalidVoiceIdentity(f"Metadata missing required fields: {sorted(missing)}")

    schema_version = int(data["schema_version"])
    if schema_version > SCHEMA_VERSION:
        raise UnsupportedIdentityVersion(
            f"Unsupported identity schema version: {schema_version}",
            user_message="This voice identity was created with a newer version of the app.",
        )
    if schema_version < 1:
        raise InvalidVoiceIdentity(f"Invalid schema version: {schema_version}")

    voice_id = data["id"]
    if not VOICE_ID_PATTERN.match(voice_id):
        raise InvalidVoiceIdentity(f"Invalid voice identity ID format: {voice_id}")

    if not data["name"].strip():
        raise InvalidVoiceIdentity("Voice identity name cannot be empty")

    for path_key in ("reference_audio", "processed_audio", "embedding"):
        value = data[path_key]
        if not isinstance(value, str) or not value.strip():
            raise InvalidVoiceIdentity(f"Invalid path for {path_key}: {value!r}")
        if Path(value).is_absolute():
            raise InvalidVoiceIdentity(
                f"Metadata path must be relative, got absolute path for {path_key}"
            )


def validate_identity_paths(identity_root: Path, data: dict[str, Any]) -> None:
    """Ensure metadata paths resolve inside the identity directory."""
    for path_key in ("reference_audio", "processed_audio", "embedding"):
        rel = data[path_key]
        resolved = (identity_root / rel).resolve()
        if identity_root.resolve() not in resolved.parents and resolved != identity_root.resolve():
            raise InvalidVoiceIdentity(
                f"Metadata path escapes identity directory: {path_key}={rel}"
            )


def load_metadata(path: Path) -> VoiceIdentity:
    """Load and validate metadata from disk."""
    if not path.exists():
        raise InvalidVoiceIdentity(f"Metadata file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise InvalidVoiceIdentity(f"Corrupt metadata JSON: {e}") from e

    if not isinstance(data, dict):
        raise InvalidVoiceIdentity("Metadata must be a JSON object")

    validate_metadata_dict(data)
    validate_identity_paths(path.parent, data)
    return VoiceIdentity.from_metadata_dict(data)


def save_metadata(path: Path, identity: VoiceIdentity) -> None:
    """Atomically write metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = identity.to_metadata_dict()
    validate_metadata_dict(payload)
    validate_identity_paths(path.parent, payload)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".metadata.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
