"""Tests for ContextProfile validation and presets."""

import math

import pytest

from voiceclone.core.context import (
    BUILTIN_PRESETS,
    ContextProfile,
    DEFAULT_CONTEXT,
    get_context_preset,
    list_context_presets,
    resolve_context,
)
from voiceclone.core.exceptions import InvalidContext, UnknownContextPreset


def test_valid_context_defaults():
    ctx = ContextProfile()
    assert ctx.name == "default"
    assert ctx.noise_level == 0.0


def test_invalid_noise_level_raises():
    with pytest.raises(InvalidContext):
        ContextProfile(noise_level=1.5)


def test_nan_raises():
    with pytest.raises(InvalidContext):
        ContextProfile(urgency=float("nan"))


def test_infinity_raises():
    with pytest.raises(InvalidContext):
        ContextProfile(urgency=float("inf"))


def test_invalid_category_raises():
    with pytest.raises(InvalidContext):
        ContextProfile(device="spaceship")


def test_unknown_preset_raises():
    with pytest.raises(UnknownContextPreset):
        get_context_preset("nonexistent")


def test_all_builtin_presets_resolve():
    for name in list_context_presets():
        preset = get_context_preset(name)
        assert preset.name == name
        assert preset.version == 1


def test_resolve_none_is_default():
    ctx = resolve_context(None)
    assert ctx.name == "default"
    assert ctx == DEFAULT_CONTEXT


def test_versioned_preset_name():
    preset = get_context_preset("car@1")
    assert preset.versioned_name == "car@1"


def test_preset_version_mismatch():
    with pytest.raises(UnknownContextPreset):
        get_context_preset("car@99")


def test_custom_context_from_dict():
    ctx = resolve_context({"name": "custom", "device": "phone", "urgency": 0.5})
    assert ctx.device == "phone"
    assert ctx.urgency == 0.5
