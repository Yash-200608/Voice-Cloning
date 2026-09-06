"""Tests for atomic voice memory repository."""

import uuid

import pytest

from voiceclone.core.exceptions import InvalidVoiceMemory, VoiceMemoryStoreError
from voiceclone.core.service import VoiceIdentityService
from voiceclone.identity.repository import VoiceRepository
from voiceclone.memory.models import VoiceMemoryItem


def test_save_load_update_delete(service, synthetic_wav):
    identity = service.create_from_file("MemRepo", synthetic_wav)
    item = service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    loaded = service.get_memory(identity.id, item.id)
    assert loaded.key == "Minitorch"

    updated = service.update_memory(
        identity.id,
        item.id,
        value={"pronunciation_type": "custom_text", "pronunciation_value": "MINI-torch"},
    )
    assert updated.version == 2
    assert updated.value["pronunciation_value"] == "MINI-torch"

    service.disable_memory(identity.id, item.id)
    assert service.get_memory(identity.id, item.id).enabled is False
    service.enable_memory(identity.id, item.id)
    assert service.get_memory(identity.id, item.id).enabled is True

    assert service.delete_memory(identity.id, item.id) is True
    assert service.list_memories(identity.id) == []


def test_persistence_after_restart(service, synthetic_wav, voiceclone_home):
    identity = service.create_from_file("PersistMem", synthetic_wav)
    service.add_pronunciation_memory(identity.id, "Chatterbox", "chatter-box")

    repo = VoiceRepository(
        voices_dir=voiceclone_home / "voices",
        outputs_dir=voiceclone_home / "outputs",
        embedding_store=service.embedding_store,
    )
    restarted = VoiceIdentityService(
        repository=repo,
        renderer=service.renderer,
        embedding_store=service.embedding_store,
        auto_migrate=False,
    )
    items = restarted.list_memories(identity.id)
    assert len(items) == 1
    assert items[0].key == "Chatterbox"


def test_corrupt_memory_file_detected(service, synthetic_wav):
    identity = service.create_from_file("CorruptMem", synthetic_wav)
    path = service.memory.repository.memory_file(identity.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(VoiceMemoryStoreError):
        service.list_memories(identity.id)


def test_identity_mismatch_rejected(service, synthetic_wav):
    identity = service.create_from_file("AtomicMem", synthetic_wav)
    data = service.add_pronunciation_memory(identity.id, "Term", "turm").to_dict()
    data["identity_id"] = f"voice_{uuid.uuid4()}"
    mismatched = VoiceMemoryItem.from_dict(data)
    with pytest.raises(InvalidVoiceMemory):
        service.memory.repository.save_all(identity.id, [mismatched])
