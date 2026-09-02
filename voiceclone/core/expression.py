"""Expressive voice models, presets, and validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from .exceptions import InvalidExpression, UnknownExpressionPreset

EXPRESSION_VERSION = 1

DIMENSION_NAMES = (
    "energy",
    "warmth",
    "arousal",
    "seriousness",
    "confidence",
    "urgency",
    "expressiveness",
    "speaking_rate",
    "pitch_shift",
    "pause_density",
)


def _clamp01(value: float) -> float:
    if not isinstance(value, (int, float)):
        raise InvalidExpression(f"Expected numeric value, got {type(value).__name__}")
    if math.isnan(value) or math.isinf(value):
        raise InvalidExpression(f"Invalid numeric value: {value}")
    if value < 0.0 or value > 1.0:
        raise InvalidExpression(f"Value {value} out of range [0.0, 1.0]")
    return float(value)


@dataclass(frozen=True)
class ExpressionProfile:
    """Structured expressive delivery settings for HOW an identity speaks."""

    name: str = "neutral"
    version: int = EXPRESSION_VERSION
    energy: float = 0.5
    warmth: float = 0.5
    arousal: float = 0.5
    seriousness: float = 0.5
    confidence: float = 0.5
    urgency: float = 0.0
    expressiveness: float = 0.5
    speaking_rate: float = 0.5
    pitch_shift: float = 0.5
    pause_density: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", _clamp01(self.energy))
        object.__setattr__(self, "warmth", _clamp01(self.warmth))
        object.__setattr__(self, "arousal", _clamp01(self.arousal))
        object.__setattr__(self, "seriousness", _clamp01(self.seriousness))
        object.__setattr__(self, "confidence", _clamp01(self.confidence))
        object.__setattr__(self, "urgency", _clamp01(self.urgency))
        object.__setattr__(self, "expressiveness", _clamp01(self.expressiveness))
        object.__setattr__(self, "speaking_rate", _clamp01(self.speaking_rate))
        object.__setattr__(self, "pitch_shift", _clamp01(self.pitch_shift))
        object.__setattr__(self, "pause_density", _clamp01(self.pause_density))
        if not self.name or not self.name.strip():
            raise InvalidExpression("Expression name cannot be empty")
        if self.version < 1:
            raise InvalidExpression(f"Invalid expression version: {self.version}")

    @property
    def versioned_name(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpressionProfile:
        if not isinstance(data, dict):
            raise InvalidExpression("Expression data must be a dictionary")
        kwargs = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        if not kwargs:
            raise InvalidExpression("Expression data is missing required fields")
        return cls(**kwargs)


def _preset(name: str, **dims: float) -> ExpressionProfile:
    return ExpressionProfile(name=name, version=EXPRESSION_VERSION, **dims)


BUILTIN_PRESETS: dict[str, ExpressionProfile] = {
    "neutral": _preset(
        "neutral",
        energy=0.5, warmth=0.5, arousal=0.5, seriousness=0.5, confidence=0.5,
        urgency=0.0, expressiveness=0.5, speaking_rate=0.5, pitch_shift=0.5, pause_density=0.5,
    ),
    "calm": _preset(
        "calm",
        energy=0.25, warmth=0.6, arousal=0.2, seriousness=0.4, confidence=0.6,
        urgency=0.0, expressiveness=0.35, speaking_rate=0.4, pitch_shift=0.45, pause_density=0.6,
    ),
    "warm": _preset(
        "warm",
        energy=0.4, warmth=0.85, arousal=0.35, seriousness=0.2, confidence=0.7,
        urgency=0.1, expressiveness=0.5, speaking_rate=0.5, pitch_shift=0.52, pause_density=0.45,
    ),
    "friendly": _preset(
        "friendly",
        energy=0.6, warmth=0.75, arousal=0.55, seriousness=0.15, confidence=0.75,
        urgency=0.1, expressiveness=0.65, speaking_rate=0.55, pitch_shift=0.55, pause_density=0.35,
    ),
    "professional": _preset(
        "professional",
        energy=0.5, warmth=0.4, arousal=0.4, seriousness=0.75, confidence=0.85,
        urgency=0.2, expressiveness=0.4, speaking_rate=0.5, pitch_shift=0.5, pause_density=0.4,
    ),
    "serious": _preset(
        "serious",
        energy=0.45, warmth=0.3, arousal=0.35, seriousness=0.9, confidence=0.8,
        urgency=0.3, expressiveness=0.35, speaking_rate=0.45, pitch_shift=0.48, pause_density=0.5,
    ),
    "excited": _preset(
        "excited",
        energy=0.85, warmth=0.65, arousal=0.9, seriousness=0.15, confidence=0.8,
        urgency=0.5, expressiveness=0.85, speaking_rate=0.7, pitch_shift=0.58, pause_density=0.2,
    ),
    "concerned": _preset(
        "concerned",
        energy=0.4, warmth=0.55, arousal=0.45, seriousness=0.7, confidence=0.5,
        urgency=0.4, expressiveness=0.55, speaking_rate=0.45, pitch_shift=0.47, pause_density=0.55,
    ),
    "urgent": _preset(
        "urgent",
        energy=0.8, warmth=0.3, arousal=0.75, seriousness=0.8, confidence=0.75,
        urgency=1.0, expressiveness=0.7, speaking_rate=0.75, pitch_shift=0.52, pause_density=0.25,
    ),
}


def list_expression_presets() -> list[str]:
    return sorted(BUILTIN_PRESETS.keys())


def get_expression_preset(name: str) -> ExpressionProfile:
    key = name.strip().lower()
    if "@" in key:
        base, _, ver = key.partition("@")
        try:
            version = int(ver)
        except ValueError as e:
            raise UnknownExpressionPreset(f"Invalid preset version: {name}") from e
        preset = BUILTIN_PRESETS.get(base)
        if preset is None:
            raise UnknownExpressionPreset(f"Unknown expression preset: {name}")
        if preset.version != version:
            raise UnknownExpressionPreset(f"Preset version mismatch: {name}")
        return preset
    preset = BUILTIN_PRESETS.get(key)
    if preset is None:
        raise UnknownExpressionPreset(
            f"Unknown expression preset: {name}",
            user_message=f"Unknown expression preset '{name}'.",
        )
    return preset


def resolve_expression(
    expression: str | ExpressionProfile | dict[str, Any] | None,
) -> ExpressionProfile:
    """Resolve preset name, profile, dict, or None (neutral) to ExpressionProfile."""
    if expression is None:
        return BUILTIN_PRESETS["neutral"]
    if isinstance(expression, ExpressionProfile):
        return expression
    if isinstance(expression, str):
        return get_expression_preset(expression)
    if isinstance(expression, dict):
        base_name = str(expression.get("name", "custom")).strip() or "custom"
        if base_name in BUILTIN_PRESETS and len(expression) == 1:
            return get_expression_preset(base_name)
        return ExpressionProfile.from_dict(expression)
    raise InvalidExpression(f"Unsupported expression type: {type(expression).__name__}")


def merge_expression(
    base: ExpressionProfile,
    overrides: dict[str, Any],
) -> ExpressionProfile:
    """Apply fine-tuning overrides on top of a base profile."""
    data = base.to_dict()
    for key, value in overrides.items():
        if key in DIMENSION_NAMES or key in ("name", "version"):
            data[key] = value
    return ExpressionProfile.from_dict(data)
