"""Tests for Chatterbox expression mapping."""

from voiceclone.core.expression import get_expression_preset
from voiceclone.inference.chatterbox_mapping import (
    MAPPING_VERSION,
    map_expression_to_chatterbox,
)


def test_mapping_is_deterministic():
    profile = get_expression_preset("calm")
    a = map_expression_to_chatterbox(profile)
    b = map_expression_to_chatterbox(profile)
    assert a == b


def test_different_presets_produce_different_settings():
    calm = map_expression_to_chatterbox(get_expression_preset("calm"))
    excited = map_expression_to_chatterbox(get_expression_preset("excited"))
    assert calm != excited


def test_mapping_values_in_valid_ranges():
    for name in ("neutral", "warm", "urgent", "professional"):
        settings = map_expression_to_chatterbox(get_expression_preset(name))
        assert 0.25 <= settings.exaggeration <= 1.0
        assert 0.2 <= settings.cfg_weight <= 1.0
        assert settings.mapping_version == MAPPING_VERSION
