"""Voice memory domain models."""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..core.exceptions import InvalidVoiceMemory
from ..core.expression import DIMENSION_NAMES
from ..core.models import VOICE_ID_PATTERN, format_timestamp, parse_timestamp, utc_now

MEMORY_SCHEMA_VERSION = 1

MEMORY_CATEGORIES = frozenset(
    {
        "pronunciation",
        "name",
        "vocabulary",
        "style_preference",
        "delivery_preference",
    }
)

MEMORY_SOURCES = frozenset({"explicit_user", "system_observation", "imported"})

PRONUNCIATION_TYPES = frozenset({"ipa", "phoneme", "custom_text", "renderer_hint"})

MEMORY_ID_PATTERN = re.compile(
    r"^memory_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def generate_memory_id() -> str:
    return f"memory_{uuid.uuid4()}"


def _clamp01(value: float, *, field_name: str = "value") -> float:
    if not isinstance(value, (int, float)):
        raise InvalidVoiceMemory(f"Expected numeric {field_name}, got {type(value).__name__}")
    if math.isnan(value) or math.isinf(value):
        raise InvalidVoiceMemory(f"Invalid {field_name}: {value}")
    if value < 0.0 or value > 1.0:
        raise InvalidVoiceMemory(f"{field_name} {value} out of range [0.0, 1.0]")
    return float(value)


def _validate_identity_id(identity_id: str) -> str:
    identity_id = str(identity_id).strip()
    if not VOICE_ID_PATTERN.match(identity_id):
        raise InvalidVoiceMemory(f"Invalid identity_id: {identity_id}")
    return identity_id


def _validate_memory_id(memory_id: str) -> str:
    memory_id = str(memory_id).strip()
    if not MEMORY_ID_PATTERN.match(memory_id):
        raise InvalidVoiceMemory(f"Invalid memory_id: {memory_id}")
    return memory_id


def _validate_category(category: str) -> str:
    key = str(category).strip().lower()
    if key not in MEMORY_CATEGORIES:
        raise InvalidVoiceMemory(
            f"Invalid memory category: {category}. Allowed: {sorted(MEMORY_CATEGORIES)}"
        )
    return key


def _validate_source(source: str) -> str:
    key = str(source).strip().lower()
    if key not in MEMORY_SOURCES:
        raise InvalidVoiceMemory(
            f"Invalid memory source: {source}. Allowed: {sorted(MEMORY_SOURCES)}"
        )
    return key


def _validate_key(key: str) -> str:
    key = str(key).strip()
    if not key:
        raise InvalidVoiceMemory("Memory key cannot be empty")
    return key


@dataclass
class VoiceMemoryItem:
    """Persistent vocal-interaction memory for one identity.

    Categories:
      pronunciation / name / vocabulary — preferred spoken form for a term
      style_preference — persistent preferred expression dimension (e.g. speaking_rate)
      delivery_preference — free-text or optional dimension preference for delivery

    Memory never mutates VoiceIdentity acoustic data.
    """

    id: str
    identity_id: str
    category: str
    key: str
    value: dict[str, Any]
    confidence: float = 1.0
    source: str = "explicit_user"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        self.id = _validate_memory_id(self.id)
        self.identity_id = _validate_identity_id(self.identity_id)
        self.category = _validate_category(self.category)
        self.key = _validate_key(self.key)
        self.source = _validate_source(self.source)
        self.confidence = _clamp01(self.confidence, field_name="confidence")
        if self.version < 1:
            raise InvalidVoiceMemory(f"Invalid memory version: {self.version}")
        if not isinstance(self.enabled, bool):
            raise InvalidVoiceMemory("enabled must be a boolean")
        if not isinstance(self.value, dict) or not self.value:
            raise InvalidVoiceMemory("Memory value must be a non-empty dictionary")
        if self.created_at is None:
            self.created_at = utc_now()
        if self.updated_at is None:
            self.updated_at = utc_now()
        if not isinstance(self.created_at, datetime) or not isinstance(self.updated_at, datetime):
            raise InvalidVoiceMemory("created_at and updated_at must be datetime values")
        self._validate_value()

    def _validate_value(self) -> None:
        if self.category in {"pronunciation", "name", "vocabulary"}:
            ptype = str(self.value.get("pronunciation_type", "custom_text")).strip().lower()
            if ptype not in PRONUNCIATION_TYPES:
                raise InvalidVoiceMemory(f"Invalid pronunciation_type: {ptype}")
            spoken = self.value.get("pronunciation_value")
            if not spoken or not str(spoken).strip():
                raise InvalidVoiceMemory("pronunciation_value is required")
            self.value = {
                "pronunciation_type": ptype,
                "pronunciation_value": str(spoken).strip(),
            }
        elif self.category == "style_preference":
            dim = str(self.value.get("dimension", "")).strip()
            if dim not in DIMENSION_NAMES:
                raise InvalidVoiceMemory(f"Invalid style dimension: {dim}")
            if "preference" not in self.value:
                raise InvalidVoiceMemory("style_preference requires preference")
            preference = _clamp01(float(self.value["preference"]), field_name="preference")
            self.value = {"dimension": dim, "preference": preference}
        elif self.category == "delivery_preference":
            description = str(self.value.get("description", "")).strip()
            if not description:
                raise InvalidVoiceMemory("delivery_preference requires description")
            payload: dict[str, Any] = {"description": description}
            dim = self.value.get("dimension")
            if dim is not None:
                dim = str(dim).strip()
                if dim not in DIMENSION_NAMES:
                    raise InvalidVoiceMemory(f"Invalid delivery dimension: {dim}")
                payload["dimension"] = dim
            if "preference" in self.value:
                payload["preference"] = _clamp01(
                    float(self.value["preference"]),
                    field_name="preference",
                )
            self.value = payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "identity_id": self.identity_id,
            "category": self.category,
            "key": self.key,
            "value": dict(self.value),
            "confidence": self.confidence,
            "source": self.source,
            "created_at": format_timestamp(self.created_at),
            "updated_at": format_timestamp(self.updated_at),
            "version": self.version,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceMemoryItem:
        if not isinstance(data, dict):
            raise InvalidVoiceMemory("Memory data must be a dictionary")
        required = ("id", "identity_id", "category", "key", "value")
        missing = [name for name in required if name not in data]
        if missing:
            raise InvalidVoiceMemory(f"Memory data missing fields: {missing}")
        created = data.get("created_at")
        updated = data.get("updated_at")
        return cls(
            id=data["id"],
            identity_id=data["identity_id"],
            category=data["category"],
            key=data["key"],
            value=dict(data["value"]),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "explicit_user")),
            created_at=parse_timestamp(created) if isinstance(created, str) else utc_now(),
            updated_at=parse_timestamp(updated) if isinstance(updated, str) else utc_now(),
            version=int(data.get("version", 1)),
            enabled=bool(data.get("enabled", True)),
        )

    def with_updates(self, **changes: Any) -> VoiceMemoryItem:
        data = self.to_dict()
        for key, value in changes.items():
            if key in {"id", "identity_id"} and value != data[key]:
                raise InvalidVoiceMemory(f"Cannot change {key}")
            data[key] = value
        data["updated_at"] = format_timestamp(utc_now())
        data["version"] = int(data.get("version", self.version)) + 1
        return VoiceMemoryItem.from_dict(data)
