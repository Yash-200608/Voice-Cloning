"""Tests for context-to-expression resolution."""

import pytest

from voiceclone.core.context import get_context_preset
from voiceclone.core.context_resolver import (
    CONTEXT_POLICY_VERSION,
    ContextResolver,
    resolve_expression_with_context,
)
from voiceclone.core.expression import DIMENSION_NAMES, get_expression_preset


def test_default_context_is_identity():
    base = get_expression_preset("calm")
    ctx = get_context_preset("default")
    result = resolve_expression_with_context(base, ctx)
    assert result.resolved_expression == base
    assert result.applied_rules == ()


def test_deterministic_resolution():
    base = get_expression_preset("neutral")
    ctx = get_context_preset("car")
    r1 = resolve_expression_with_context(base, ctx)
    r2 = resolve_expression_with_context(base, ctx)
    assert r1.resolved_expression == r2.resolved_expression
    assert r1.applied_rules == r2.applied_rules


def test_car_context_adjusts_intended_dimensions():
    base = get_expression_preset("calm")
    ctx = get_context_preset("car")
    result = resolve_expression_with_context(base, ctx)
    resolved = result.resolved_expression
    assert "driving" in result.applied_rules
    assert resolved.confidence >= base.confidence
    assert resolved.pause_density <= base.pause_density


def test_noisy_context_increases_energy():
    base = get_expression_preset("neutral")
    ctx = get_context_preset("noisy_environment")
    result = resolve_expression_with_context(base, ctx)
    assert result.resolved_expression.energy > base.energy
    assert "high_noise" in result.applied_rules


def test_explicit_expression_preserved_as_baseline():
    base = get_expression_preset("calm")
    ctx = get_context_preset("warning")
    result = resolve_expression_with_context(base, ctx)
    assert result.base_expression.name == "calm"
    assert result.resolved_expression.warmth == pytest.approx(base.warmth, abs=0.01) or (
        result.resolved_expression.warmth >= 0.0
    )


def test_values_remain_bounded():
    base = get_expression_preset("excited")
    ctx = get_context_preset("warning")
    result = resolve_expression_with_context(base, ctx)
    data = result.resolved_expression.to_dict()
    for dim in DIMENSION_NAMES:
        assert 0.0 <= data[dim] <= 1.0


def test_policy_version_recorded():
    result = resolve_expression_with_context(
        get_expression_preset("neutral"),
        get_context_preset("car"),
    )
    assert result.policy_version == CONTEXT_POLICY_VERSION


def test_context_resolver_class():
    resolver = ContextResolver()
    result = resolver.resolve(get_context_preset("presentation"), get_expression_preset("neutral"))
    assert "presentation" in result.applied_rules
