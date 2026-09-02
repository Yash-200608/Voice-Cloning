"""Tests for rename and delete operations."""

from pathlib import Path


def test_rename_changes_name_not_id(service, synthetic_wav):
    identity = service.create_from_file("Original", synthetic_wav)
    identity_id = identity.id
    renamed = service.rename_identity(identity_id, "Renamed")
    assert renamed.name == "Renamed"
    assert renamed.id == identity_id
    assert service.repository.identity_dir(identity_id).exists()


def test_delete_removes_identity_only(service, synthetic_wav):
    a = service.create_from_file("Keep", synthetic_wav)
    b = service.create_from_file("Remove", synthetic_wav)

    assert service.delete_identity(b.id) is True
    assert not service.repository.identity_dir(b.id).exists()
    assert service.repository.identity_dir(a.id).exists()

    remaining = {i.id for i in service.list_identities()}
    assert a.id in remaining
    assert b.id not in remaining
