"""Voice memory application service."""

from __future__ import annotations

from typing import Any

from ..core.exceptions import VoiceIdentityNotFound
from ..identity.repository import VoiceRepository
from .models import VoiceMemoryItem, generate_memory_id
from .repository import VoiceMemoryRepository


class VoiceMemoryService:
    """Business logic for per-identity voice memory CRUD and export/import."""

    def __init__(
        self,
        repository: VoiceMemoryRepository | None = None,
        voice_repository: VoiceRepository | None = None,
    ):
        self.voice_repository = voice_repository or VoiceRepository()
        self.repository = repository or VoiceMemoryRepository(self.voice_repository)

    def list_memories(
        self,
        identity_id: str,
        *,
        enabled_only: bool = False,
    ) -> list[VoiceMemoryItem]:
        items = self.repository.load_all(identity_id)
        if enabled_only:
            return [item for item in items if item.enabled]
        return items

    def get_memory(self, identity_id: str, memory_id: str) -> VoiceMemoryItem:
        return self.repository.get(identity_id, memory_id)

    def find_memory(
        self,
        identity_id: str,
        category: str,
        key: str,
    ) -> VoiceMemoryItem | None:
        key_norm = key.strip().lower()
        for item in self.list_memories(identity_id):
            if item.category == category and item.key.lower() == key_norm:
                return item
        return None

    def add_memory(
        self,
        identity_id: str,
        category: str,
        key: str,
        value: dict[str, Any],
        *,
        confidence: float = 1.0,
        source: str = "explicit_user",
        enabled: bool = True,
    ) -> VoiceMemoryItem:
        # Persistent memory creation is explicit/confirmed only — no autonomous learning.
        item = VoiceMemoryItem(
            id=generate_memory_id(),
            identity_id=identity_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            enabled=enabled,
        )
        return self.repository.add(item)

    def add_pronunciation_memory(
        self,
        identity_id: str,
        term: str,
        pronunciation_value: str,
        *,
        pronunciation_type: str = "custom_text",
        confidence: float = 1.0,
        source: str = "explicit_user",
        category: str = "pronunciation",
    ) -> VoiceMemoryItem:
        return self.add_memory(
            identity_id,
            category,
            term,
            {
                "pronunciation_type": pronunciation_type,
                "pronunciation_value": pronunciation_value,
            },
            confidence=confidence,
            source=source,
        )

    def add_style_preference(
        self,
        identity_id: str,
        dimension: str,
        preference: float,
        *,
        confidence: float = 1.0,
        source: str = "explicit_user",
    ) -> VoiceMemoryItem:
        return self.add_memory(
            identity_id,
            "style_preference",
            dimension,
            {"dimension": dimension, "preference": preference},
            confidence=confidence,
            source=source,
        )

    def update_memory(
        self,
        identity_id: str,
        memory_id: str,
        *,
        key: str | None = None,
        value: dict[str, Any] | None = None,
        confidence: float | None = None,
        enabled: bool | None = None,
        source: str | None = None,
    ) -> VoiceMemoryItem:
        current = self.repository.get(identity_id, memory_id)
        changes: dict[str, Any] = {}
        if key is not None:
            changes["key"] = key
        if value is not None:
            changes["value"] = value
        if confidence is not None:
            changes["confidence"] = confidence
        if enabled is not None:
            changes["enabled"] = enabled
        if source is not None:
            changes["source"] = source
        updated = current.with_updates(**changes)
        return self.repository.update(identity_id, memory_id, updated)

    def delete_memory(self, identity_id: str, memory_id: str) -> bool:
        return self.repository.delete(identity_id, memory_id)

    def enable_memory(self, identity_id: str, memory_id: str) -> VoiceMemoryItem:
        return self.update_memory(identity_id, memory_id, enabled=True)

    def disable_memory(self, identity_id: str, memory_id: str) -> VoiceMemoryItem:
        return self.update_memory(identity_id, memory_id, enabled=False)

    def clear_memories(self, identity_id: str) -> None:
        self.repository.clear(identity_id)

    def export_memories(self, identity_id: str) -> dict[str, Any]:
        return self.repository.export_store(identity_id)

    def import_memories(
        self,
        identity_id: str,
        payload: dict[str, Any],
        *,
        replace: bool = False,
    ) -> list[VoiceMemoryItem]:
        return self.repository.import_store(identity_id, payload, replace=replace)

    def resolve_memories(self, identity_id: str) -> list[VoiceMemoryItem]:
        if not self.voice_repository.identity_dir(identity_id).exists():
            raise VoiceIdentityNotFound(
                f"Voice identity not found: {identity_id}",
                user_message="That voice identity was not found.",
            )
        return self.repository.load_all(identity_id)
