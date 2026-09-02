"""Tests for ExpressionProfile validation and presets."""

import math

import pytest

from voiceclone.core.expression import (
    BUILTIN_PRESETS,
    ExpressionProfile,
    get_expression_preset,
    list_expression_presets,
    merge_expression,
    resolve_expression,
)
from voiceclone.core.exceptions import InvalidExpression, UnknownExpressionPreset


def test_valid_profile_defaults():
    profile = ExpressionProfile()
    assert profile.name == "neutral"
    assert profile.energy == 0.5


def test_invalid_range_raises():
    with pytest.raises(InvalidExpression):
        ExpressionProfile(energy=1.5)


def test_nan_raises():
    with pytest.raises(InvalidExpression):
        ExpressionProfile(warmth=float("nan"))


def test_infinity_raises():
    with pytest.raises(InvalidExpression):
        ExpressionProfile(seriousness=float("inf"))


def test_negative_raises():
    with pytest.raises(InvalidExpression):
        ExpressionProfile(confidence=-0.1)


def test_unknown_preset_raises():
    with pytest.raises(UnknownExpressionPreset):
        get_expression_preset("nonexistent")


def test_all_builtin_presets_resolve():
    for name in list_expression_presets():
        preset = get_expression_preset(name)
        assert preset.name == name
        assert preset.version == 1


def test_resolve_none_is_neutral():
    profile = resolve_expression(None)
    assert profile.name == "neutral"


def test_merge_expression_overrides():
    base = get_expression_preset("calm")
    merged = merge_expression(base, {"energy": 0.9, "name": "custom"})
    assert merged.energy == 0.9
    assert merged.name == "custom"
    assert merged.warmth == base.warmth


def test_versioned_preset_name():
    preset = get_expression_preset("calm@1")
    assert preset.versioned_name == "calm@1"


def test_preset_version_mismatch():
    with pytest.raises(UnknownExpressionPreset):
        get_expression_preset("calm@99")
