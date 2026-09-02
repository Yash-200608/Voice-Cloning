"""Voice identity data model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1
VOICE_ID_PATTERN = re.compile(r"^voice_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

EMBEDDING_MODEL = "resemblyzer"
RENDERER_MODEL = "chatterbox"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_timestamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass
class VoiceIdentity:
    """Persistent voice identity representing WHO the voice is."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    reference_audio: str
    processed_audio: str
    embedding_path: str
    embedding_model: str
    embedding_version: str
    sample_rate: int
    duration_seconds: float
    renderer_model: str
    renderer_version: str
    schema_version: int = SCHEMA_VERSION
    quality_metrics: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": format_timestamp(self.created_at),
            "updated_at": format_timestamp(self.updated_at),
            "reference_audio": self.reference_audio,
            "processed_audio": self.processed_audio,
            "embedding": self.embedding_path,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "renderer_model": self.renderer_model,
            "renderer_version": self.renderer_version,
            "schema_version": self.schema_version,
            "quality_metrics": self.quality_metrics,
        }

    @classmethod
    def from_metadata_dict(cls, data: dict[str, Any]) -> VoiceIdentity:
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=parse_timestamp(data["created_at"]),
            updated_at=parse_timestamp(data["updated_at"]),
            reference_audio=data["reference_audio"],
            processed_audio=data["processed_audio"],
            embedding_path=data.get("embedding", data.get("embedding_path", "")),
            embedding_model=data["embedding_model"],
            embedding_version=data["embedding_version"],
            sample_rate=int(data["sample_rate"]),
            duration_seconds=float(data["duration_seconds"]),
            renderer_model=data["renderer_model"],
            renderer_version=data["renderer_version"],
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            quality_metrics=dict(data.get("quality_metrics", {})),
        )
