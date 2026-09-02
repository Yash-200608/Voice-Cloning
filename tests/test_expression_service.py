"""Tests for expression-aware service synthesis."""

from pathlib import Path

from voiceclone.core.expression import ExpressionProfile
from voiceclone.evaluation.render_metadata import load_render_metadata, metadata_path_for_audio


def test_synthesize_without_expression_uses_neutral(service, synthetic_wav):
    identity = service.create_from_file("NeutralCompat", synthetic_wav)
    before = identity.updated_at
    output = service.synthesize(identity.id, "Hello world.")
    after = service.get_identity(identity.id)
    assert Path(output).exists()
    assert after.id == identity.id
    assert after.updated_at == before
    meta = load_render_metadata(output)
    assert meta["expression_name"] == "neutral@1"


def test_synthesize_with_preset(service, synthetic_wav):
    identity = service.create_from_file("CalmVoice", synthetic_wav)
    output = service.synthesize(identity.id, "Stay calm.", expression="calm")
    meta = load_render_metadata(output)
    assert meta["expression_name"] == "calm@1"
    assert meta["identity_id"] == identity.id


def test_synthesize_with_custom_profile(service, synthetic_wav):
    identity = service.create_from_file("CustomVoice", synthetic_wav)
    profile = ExpressionProfile(name="custom", energy=0.8, warmth=0.7)
    output = service.synthesize(identity.id, "Custom delivery.", expression=profile)
    meta = load_render_metadata(output)
    assert meta["expression_profile"]["energy"] == 0.8
    assert meta["generation_parameters"]["exaggeration"]


def test_best_of_three_with_expression(service, synthetic_wav):
    identity = service.create_from_file("BestExpr", synthetic_wav)
    best, score = service.synthesize_best_of(
        identity.id, "Best expression test.", n=3, expression="excited",
    )
    assert Path(best).exists()
    assert metadata_path_for_audio(best).exists()
    meta = load_render_metadata(best)
    assert meta["expression_name"] == "excited@1"
    assert 0.0 <= score <= 1.0
    wav_count = len(list(service.repository.output_dir(identity.id).glob("*.wav")))
    assert wav_count == 1


def test_list_expression_presets_service(service):
    presets = service.list_expression_presets()
    assert "calm" in presets
    assert "urgent" in presets
