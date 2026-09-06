"""Atomic per-identity voice memory persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..core.exceptions import (
    InvalidVoiceMemory,
    VoiceIdentityNotFound,
    VoiceMemoryNotFound,
    VoiceMemoryStoreError,
)
from ..identity.repository import VoiceRepository
from .models import MEMORY_SCHEMA_VERSION, VoiceMemoryItem

MEMORIES_FILENAME = "memories.json"


class VoiceMemoryRepository:
    """Persists voice memory stores under each identity directory.

    Layout:
      <identity_dir>/memory/memories.json

    Writes use temp file → validate → atomic replace.
    """

    def __init__(self, voice_repository: VoiceRepository | None = None):
        self.voice_repository = voice_repository or VoiceRepository()

    def memory_dir(self, identity_id: str) -> Path:
        return self.voice_repository.identity_dir(identity_id) / "memory"

    def memory_file(self, identity_id: str) -> Path:
        return self.memory_dir(identity_id) / MEMORIES_FILENAME

    def _ensure_identity_exists(self, identity_id: str) -> None:
        if not self.voice_repository.identity_dir(identity_id).exists():
            raise VoiceIdentityNotFound(
                f"Voice identity not found: {identity_id}",
                user_message="That voice identity was not found.",
            )

    def load_all(self, identity_id: str) -> list[VoiceMemoryItem]:
        self._ensure_identity_exists(identity_id)
        path = self.memory_file(identity_id)
        if not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise VoiceMemoryStoreError(
                f"Corrupt memory store for {identity_id}: {e}",
                user_message="Voice memory store is corrupt and cannot be loaded.",
            ) from e
        return self._parse_store(payload, identity_id)

    def save_all(self, identity_id: str, memories: list[VoiceMemoryItem]) -> None:
        self._ensure_identity_exists(identity_id)
        for item in memories:
            if item.identity_id != identity_id:
                raise InvalidVoiceMemory(
                    f"Memory {item.id} belongs to {item.identity_id}, not {identity_id}"
                )
        ids = [item.id for item in memories]
        if len(ids) != len(set(ids)):
            raise InvalidVoiceMemory("Duplicate memory IDs in store")
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "identity_id": identity_id,
            "memories": [item.to_dict() for item in memories],
        }
        self._atomic_write(self.memory_file(identity_id), payload)

    def get(self, identity_id: str, memory_id: str) -> VoiceMemoryItem:
        for item in self.load_all(identity_id):
            if item.id == memory_id:
                return item
        raise VoiceMemoryNotFound(
            f"Memory not found: {memory_id}",
            user_message="That voice memory item was not found.",
        )

    def add(self, item: VoiceMemoryItem) -> VoiceMemoryItem:
        memories = self.load_all(item.identity_id)
        if any(existing.id == item.id for existing in memories):
            raise InvalidVoiceMemory(f"Memory already exists: {item.id}")
        if any(
            existing.category == item.category and existing.key.lower() == item.key.lower()
            for existing in memories
        ):
            raise InvalidVoiceMemory(
                f"Memory key already exists for category {item.category}: {item.key}"
            )
        memories.append(item)
        self.save_all(item.identity_id, memories)
        return item

    def update(self, identity_id: str, memory_id: str, updated: VoiceMemoryItem) -> VoiceMemoryItem:
        if updated.identity_id != identity_id:
            raise InvalidVoiceMemory("Cannot change memory identity_id")
        if updated.id != memory_id:
            raise InvalidVoiceMemory("Cannot change memory id")
        memories = self.load_all(identity_id)
        found = False
        new_list: list[VoiceMemoryItem] = []
        for item in memories:
            if item.id == memory_id:
                new_list.append(updated)
                found = True
            else:
                if (
                    item.category == updated.category
                    and item.key.lower() == updated.key.lower()
                ):
                    raise InvalidVoiceMemory(
                        f"Memory key already exists for category {updated.category}: {updated.key}"
                    )
                new_list.append(item)
        if not found:
            raise VoiceMemoryNotFound(
                f"Memory not found: {memory_id}",
                user_message="That voice memory item was not found.",
            )
        self.save_all(identity_id, new_list)
        return updated

    def delete(self, identity_id: str, memory_id: str) -> bool:
        memories = self.load_all(identity_id)
        new_list = [item for item in memories if item.id != memory_id]
        if len(new_list) == len(memories):
            return False
        self.save_all(identity_id, new_list)
        return True

    def clear(self, identity_id: str) -> None:
        self.save_all(identity_id, [])

    def export_store(self, identity_id: str) -> dict[str, Any]:
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "identity_id": identity_id,
            "memories": [item.to_dict() for item in self.load_all(identity_id)],
        }

    def import_store(
        self,
        identity_id: str,
        payload: dict[str, Any],
        *,
        replace: bool = False,
    ) -> list[VoiceMemoryItem]:
        imported = self._parse_store(payload, identity_id, remap_identity=True)
        if replace:
            self.save_all(identity_id, imported)
            return imported
        existing = self.load_all(identity_id)
        existing_keys = {(item.category, item.key.lower()) for item in existing}
        merged = list(existing)
        for item in imported:
            if (item.category, item.key.lower()) in existing_keys:
                continue
            merged.append(item)
        self.save_all(identity_id, merged)
        return merged

    def _parse_store(
        self,
        payload: dict[str, Any],
        identity_id: str,
        *,
        remap_identity: bool = False,
    ) -> list[VoiceMemoryItem]:
        if not isinstance(payload, dict):
            raise InvalidVoiceMemory("Memory store must be a JSON object")
        if payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            raise InvalidVoiceMemory(
                f"Unsupported memory schema version: {payload.get('schema_version')}"
            )
        store_identity = str(payload.get("identity_id", identity_id))
        if not remap_identity and store_identity != identity_id:
            raise InvalidVoiceMemory("Memory store identity_id mismatch")
        raw = payload.get("memories")
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise InvalidVoiceMemory("Memory store memories must be a list")
        items: list[VoiceMemoryItem] = []
        for entry in raw:
            item = VoiceMemoryItem.from_dict(entry)
            if remap_identity and item.identity_id != identity_id:
                data = item.to_dict()
                data["identity_id"] = identity_id
                item = VoiceMemoryItem.from_dict(data)
            if item.identity_id != identity_id:
                raise InvalidVoiceMemory("Memory item identity_id mismatch")
            items.append(item)
        return items

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".memories.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            # Validate before publish.
            VoiceMemoryItem  # noqa: B018 — keep import warm
            with open(tmp_name, encoding="utf-8") as f:
                parsed = json.load(f)
            if parsed.get("schema_version") != MEMORY_SCHEMA_VERSION:
                raise InvalidVoiceMemory("Validated temp store has invalid schema")
            for entry in parsed.get("memories", []):
                VoiceMemoryItem.from_dict(entry)
            os.replace(tmp_name, path)
        except Exception as e:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            if isinstance(e, (InvalidVoiceMemory, VoiceMemoryStoreError)):
                raise
            raise VoiceMemoryStoreError(
                f"Failed to persist memory store: {e}",
                user_message="Failed to save voice memory.",
            ) from e
