"""Tests for voice memory service resolution and synthesis integration."""

from pathlib import Path

from voiceclone.evaluation.render_metadata import load_render_metadata


def test_identity_isolation(service, synthetic_wav):
    a = service.create_from_file("MemA", synthetic_wav)
    b = service.create_from_file("MemB", synthetic_wav)
    service.add_pronunciation_memory(a.id, "Minitorch", "mini-torch-a")
    service.add_pronunciation_memory(b.id, "Minitorch", "mini-torch-b")

    plan_a = service.resolve_render_plan(
        identity_id=a.id, text="Minitorch is great.", use_memory=True
    )
    plan_b = service.resolve_render_plan(
        identity_id=b.id, text="Minitorch is great.", use_memory=True
    )
    assert "mini-torch-a" in plan_a.render_text
    assert "mini-torch-b" in plan_b.render_text
    assert "mini-torch-b" not in plan_a.render_text
    assert "mini-torch-a" not in plan_b.render_text


def test_precedence_explicit_expression_overrides_style_memory(service, synthetic_wav):
    identity = service.create_from_file("MemPrec", synthetic_wav)
    service.add_style_preference(identity.id, "speaking_rate", 0.9)
    # No explicit expression -> memory can adjust defaults
    plan_default = service.resolve_render_plan(
        identity_id=identity.id, text="Hello.", use_memory=True
    )
    assert plan_default.memory.memories_applied
    # Explicit expression wins for style preferences
    plan_explicit = service.resolve_render_plan(
        expression="calm",
        identity_id=identity.id,
        text="Hello.",
        use_memory=True,
    )
    ignored_reasons = dict(plan_explicit.memory.memories_ignored)
    assert any(reason == "explicitly_overridden" for reason in ignored_reasons.values())
    assert plan_explicit.base_expression.name == "calm"


def test_disabled_memory_not_applied(service, synthetic_wav):
    identity = service.create_from_file("MemDis", synthetic_wav)
    item = service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    service.disable_memory(identity.id, item.id)
    plan = service.resolve_render_plan(
        identity_id=identity.id, text="Minitorch works.", use_memory=True
    )
    assert "mini-torch" not in plan.render_text
    assert item.id not in plan.memory.memories_applied


def test_use_memory_false_bypass(service, synthetic_wav):
    identity = service.create_from_file("MemBypass", synthetic_wav)
    service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    service.add_style_preference(identity.id, "speaking_rate", 0.2)
    plan = service.resolve_render_plan(
        identity_id=identity.id, text="Minitorch works.", use_memory=False
    )
    assert "mini-torch" not in plan.render_text
    assert plan.memory.memories_applied == ()


def test_synthesize_records_memory_provenance(service, synthetic_wav):
    identity = service.create_from_file("MemSynth", synthetic_wav)
    item = service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    output = service.synthesize(identity.id, "Minitorch is ready.")
    meta = load_render_metadata(output)
    assert item.id in meta["memory_items_used"]
    assert meta["use_memory"] is True
    assert "mini-torch" in meta["render_text"]
    assert meta["memory_resolution_version"]


def test_synthesize_with_expression_and_context_and_memory(service, synthetic_wav):
    identity = service.create_from_file("MemCombo", synthetic_wav)
    service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    output = service.synthesize(
        identity.id,
        "Minitorch briefing.",
        expression="calm",
        context="car",
        use_memory=True,
    )
    meta = load_render_metadata(output)
    assert meta["context_name"].startswith("car")
    assert meta["base_expression_name"].startswith("calm")
    assert meta["memory_items_used"]
    assert "mini-torch" in meta["render_text"]


def test_best_of_with_memory(service, synthetic_wav):
    identity = service.create_from_file("MemBest", synthetic_wav)
    service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    best, score = service.synthesize_best_of(
        identity.id, "Minitorch check.", n=3, use_memory=True
    )
    assert Path(best).exists()
    assert 0.0 <= score <= 1.0
    meta = load_render_metadata(best)
    assert "mini-torch" in meta["render_text"]


def test_delete_removes_effect(service, synthetic_wav):
    identity = service.create_from_file("MemDel", synthetic_wav)
    item = service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    service.delete_memory(identity.id, item.id)
    plan = service.resolve_render_plan(
        identity_id=identity.id, text="Minitorch remains.", use_memory=True
    )
    assert "mini-torch" not in plan.render_text


def test_memory_does_not_mutate_identity(service, synthetic_wav):
    identity = service.create_from_file("MemIso", synthetic_wav)
    before = service.get_identity(identity.id)
    service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    service.add_style_preference(identity.id, "speaking_rate", 0.3)
    service.synthesize(identity.id, "Minitorch line.", use_memory=True)
    after = service.get_identity(identity.id)
    assert after.id == before.id
    assert after.embedding_path == before.embedding_path
    assert after.name == before.name
