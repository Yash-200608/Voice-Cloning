"""Generation event metadata for rendered audio outputs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.expression import ExpressionProfile
from ..core.models import format_timestamp


def metadata_path_for_audio(audio_path: str | Path) -> Path:
    path = Path(audio_path)
    return path.with_suffix(path.suffix + ".meta.json")


def save_render_metadata(
    audio_path: str | Path,
    *,
    identity_id: str,
    identity_name: str,
    expression: ExpressionProfile,
    renderer: str,
    renderer_version: str,
    generation_parameters: dict[str, Any],
    similarity: float | None = None,
    duration_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically write rendering event metadata alongside generated audio."""
    meta_path = metadata_path_for_audio(audio_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "identity_id": identity_id,
        "identity_name": identity_name,
        "expression_name": expression.versioned_name,
        "expression_version": expression.version,
        "expression_profile": expression.to_dict(),
        "renderer": renderer,
        "renderer_version": renderer_version,
        "generation_parameters": generation_parameters,
        "timestamp": format_timestamp(datetime.now(timezone.utc)),
    }
    if similarity is not None:
        payload["speaker_similarity"] = round(similarity, 4)
    if duration_seconds is not None:
        payload["duration_seconds"] = round(duration_seconds, 3)
    if extra:
        payload.update(extra)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(meta_path.parent),
        prefix=".render_meta.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, meta_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return meta_path


def load_render_metadata(audio_path: str | Path) -> dict[str, Any]:
    meta_path = metadata_path_for_audio(audio_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Render metadata not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)
