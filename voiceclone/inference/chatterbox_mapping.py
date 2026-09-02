"""Chatterbox-specific mapping from ExpressionProfile to render settings."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.expression import ExpressionProfile

MAPPING_VERSION = 1


@dataclass(frozen=True)
class ChatterboxRenderSettings:
    """Renderer-specific generation parameters for Chatterbox."""

    exaggeration: float
    cfg_weight: float
    speaking_rate_factor: float = 1.0
    pitch_semitones: float = 0.0
    mapping_version: int = MAPPING_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "exaggeration", _clamp(self.exaggeration, 0.25, 1.0))
        object.__setattr__(self, "cfg_weight", _clamp(self.cfg_weight, 0.2, 1.0))
        object.__setattr__(self, "speaking_rate_factor", _clamp(self.speaking_rate_factor, 0.75, 1.35))
        object.__setattr__(self, "pitch_semitones", _clamp(self.pitch_semitones, -4.0, 4.0))

    def to_dict(self) -> dict:
        return {
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "speaking_rate_factor": self.speaking_rate_factor,
            "pitch_semitones": self.pitch_semitones,
            "mapping_version": self.mapping_version,
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def map_expression_to_chatterbox(expression: ExpressionProfile) -> ChatterboxRenderSettings:
    """
    Deterministically map semantic expression dimensions to Chatterbox parameters.

    These mappings control expressive delivery characteristics and are versioned
    separately from preset definitions.
    """
    exaggeration = (
        0.25
        + 0.35 * expression.expressiveness
        + 0.20 * expression.energy
        + 0.15 * expression.arousal
        + 0.05 * expression.warmth
    )
    cfg_weight = (
        0.20
        + 0.30 * expression.seriousness
        + 0.25 * expression.confidence
        + 0.15 * (1.0 - expression.expressiveness)
        + 0.10 * expression.urgency
    )
    speaking_rate_factor = 0.75 + 0.60 * expression.speaking_rate
    pitch_semitones = (expression.pitch_shift - 0.5) * 8.0

    return ChatterboxRenderSettings(
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        speaking_rate_factor=speaking_rate_factor,
        pitch_semitones=pitch_semitones,
        mapping_version=MAPPING_VERSION,
    )
