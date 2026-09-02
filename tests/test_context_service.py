"""Tests for context-aware service synthesis."""

from pathlib import Path

from voiceclone.core.context import ContextProfile
from voiceclone.core.expression import ExpressionProfile
from voiceclone.evaluation.render_metadata import load_render_metadata, metadata_path_for_audio


def test_synthesize_without_context_unchanged(service, synthetic_wav):
    identity = service.create_from_file("NoCtx", synthetic_wav)
    before = identity.updated_at
    output = service.synthesize(identity.id, "Hello world.")
    after = service.get_identity(identity.id)
    assert Path(output).exists()
    assert after.id == identity.id
    assert after.updated_at == before
    meta = load_render_metadata(output)
    assert meta["expression_name"] == "neutral@1"
    assert meta["context_name"] == "default@1"


def test_synthesize_with_context_preset(service, synthetic_wav):
    identity = service.create_from_file("CarCtx", synthetic_wav)
    output = service.synthesize(identity.id, "Please turn left.", context="car")
    meta = load_render_metadata(output)
    assert meta["context_name"] == "car@1"
    assert meta["identity_id"] == identity.id
    assert meta["base_expression_name"] == "neutral@1"
    assert meta["resolved_expression_name"]
    assert meta["context_policy_version"]


def test_synthesize_expression_plus_context(service, synthetic_wav):
    identity = service.create_from_file("ExprCtx", synthetic_wav)
    output = service.synthesize(
        identity.id,
        "We need to leave now.",
        expression="calm",
        context="car",
    )
    meta = load_render_metadata(output)
    assert meta["base_expression_name"] == "calm@1"
    assert meta["context_name"] == "car@1"
    assert meta["applied_context_rules"]


def test_context_does_not_modify_identity(service, synthetic_wav):
    identity = service.create_from_file("IsoCtx", synthetic_wav)
    before = service.get_identity(identity.id)
    before_id = before.id
    before_embedding = before.embedding_path
    service.synthesize(identity.id, "Context test.", context="warning")
    after = service.get_identity(identity.id)
    assert after.id == before_id
    assert after.embedding_path == before_embedding
    assert after.name == before.name


def test_best_of_three_with_context(service, synthetic_wav):
    identity = service.create_from_file("BestCtx", synthetic_wav)
    best, score = service.synthesize_best_of(
        identity.id,
        "Best context test.",
        n=3,
        expression="calm",
        context="car",
    )
    assert Path(best).exists()
    meta = load_render_metadata(best)
    assert meta["context_name"] == "car@1"
    assert meta["base_expression_name"] == "calm@1"
    assert 0.0 <= score <= 1.0
    wav_count = len(list(service.repository.output_dir(identity.id).glob("*.wav")))
    assert wav_count == 1


def test_resolve_render_plan(service):
    plan = service.resolve_render_plan(expression="warm", context="presentation")
    assert plan.base_expression.name == "warm"
    assert plan.context.name == "presentation"
    assert plan.resolved_expression.name.startswith("warm+")


def test_list_context_presets_service(service):
    presets = service.list_context_presets()
    assert "car" in presets
    assert "default" in presets


def test_custom_context_profile(service, synthetic_wav):
    identity = service.create_from_file("CustomCtx", synthetic_wav)
    ctx = ContextProfile(
        name="custom",
        device="car",
        environment="noisy",
        activity="driving",
        noise_level=0.8,
        urgency=0.7,
    )
    output = service.synthesize(identity.id, "Custom context.", context=ctx)
    meta = load_render_metadata(output)
    assert meta["context_profile"]["device"] == "car"
