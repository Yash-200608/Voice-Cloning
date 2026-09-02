"""Deterministic context-to-expression resolution policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import ContextProfile, DEFAULT_CONTEXT
from .expression import DIMENSION_NAMES, ExpressionProfile, merge_expression

CONTEXT_POLICY_VERSION = "1"

# Maximum bounded adjustment per dimension from context alone.
MAX_CONTEXT_DELTA = 0.25


@dataclass(frozen=True)
class ContextResolutionResult:
    """Outcome of applying context policy to a base expression."""

    base_expression: ExpressionProfile
    context: ContextProfile
    resolved_expression: ExpressionProfile
    applied_rules: tuple[str, ...]
    policy_version: str = CONTEXT_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_expression": self.base_expression.to_dict(),
            "base_expression_name": self.base_expression.versioned_name,
            "context": self.context.to_dict(),
            "context_name": self.context.versioned_name,
            "resolved_expression": self.resolved_expression.to_dict(),
            "resolved_expression_name": self.resolved_expression.versioned_name,
            "applied_rules": list(self.applied_rules),
            "context_policy_version": self.policy_version,
        }

    def summary(self) -> str:
        """Human-readable resolution summary for UI/debug."""
        base = self.base_expression.name
        resolved = self.resolved_expression.name
        if self.context.is_default and not self.applied_rules:
            return f"Base: {base}"
        rules = ", ".join(self.applied_rules) if self.applied_rules else "no adjustments"
        return f"Base: {base} | Context: {self.context.name} | Resolved: {resolved} ({rules})"


def _bounded_delta(value: float) -> float:
    if value > MAX_CONTEXT_DELTA:
        return MAX_CONTEXT_DELTA
    if value < -MAX_CONTEXT_DELTA:
        return -MAX_CONTEXT_DELTA
    return value


def _apply_delta(
    deltas: dict[str, float],
    dimension: str,
    amount: float,
) -> None:
    deltas[dimension] = deltas.get(dimension, 0.0) + _bounded_delta(amount)


def _effective_noise(context: ContextProfile) -> float:
    noise = context.noise_level
    if context.environment == "noisy":
        noise = max(noise, 0.65)
    elif context.environment == "very_noisy":
        noise = max(noise, 0.85)
    elif context.environment == "quiet":
        noise = min(noise, 0.15)
    return noise


class ContextResolver:
    """Applies deterministic, bounded context adjustments to a base expression."""

    def resolve(
        self,
        context: ContextProfile,
        base_expression: ExpressionProfile,
    ) -> ContextResolutionResult:
        if context.is_default:
            return ContextResolutionResult(
                base_expression=base_expression,
                context=context,
                resolved_expression=base_expression,
                applied_rules=(),
            )

        deltas: dict[str, float] = {}
        rules: list[str] = []
        noise = _effective_noise(context)

        if noise >= 0.6:
            _apply_delta(deltas, "energy", 0.08 + (noise - 0.6) * 0.3)
            _apply_delta(deltas, "expressiveness", 0.06 + (noise - 0.6) * 0.2)
            _apply_delta(deltas, "speaking_rate", 0.05 + (noise - 0.6) * 0.15)
            _apply_delta(deltas, "pause_density", -0.1 - (noise - 0.6) * 0.2)
            _apply_delta(deltas, "confidence", 0.05)
            rules.append("high_noise")

        if context.device == "car" or context.activity == "driving":
            _apply_delta(deltas, "confidence", 0.08)
            _apply_delta(deltas, "seriousness", 0.06)
            _apply_delta(deltas, "speaking_rate", 0.05)
            _apply_delta(deltas, "pause_density", -0.12)
            _apply_delta(deltas, "expressiveness", 0.05)
            rules.append("driving")

        if context.urgency >= 0.6:
            scale = (context.urgency - 0.6) / 0.4
            _apply_delta(deltas, "urgency", 0.1 + scale * 0.15)
            _apply_delta(deltas, "energy", 0.08 + scale * 0.12)
            _apply_delta(deltas, "seriousness", 0.05 + scale * 0.1)
            _apply_delta(deltas, "speaking_rate", 0.06 + scale * 0.1)
            rules.append("high_urgency")

        if context.time_of_day == "night":
            _apply_delta(deltas, "energy", -0.1)
            _apply_delta(deltas, "arousal", -0.08)
            _apply_delta(deltas, "expressiveness", -0.05)
            rules.append("night")

        if context.interaction_mode == "presentation" or context.activity == "presenting":
            _apply_delta(deltas, "confidence", 0.1)
            _apply_delta(deltas, "seriousness", 0.08)
            _apply_delta(deltas, "speaking_rate", -0.05)
            _apply_delta(deltas, "pause_density", 0.05)
            _apply_delta(deltas, "warmth", -0.05)
            rules.append("presentation")

        if context.interaction_mode == "notification":
            _apply_delta(deltas, "urgency", 0.08)
            _apply_delta(deltas, "energy", 0.06)
            _apply_delta(deltas, "pause_density", -0.08)
            rules.append("notification")

        if context.interaction_mode == "warning":
            _apply_delta(deltas, "urgency", 0.15)
            _apply_delta(deltas, "energy", 0.1)
            _apply_delta(deltas, "seriousness", 0.1)
            _apply_delta(deltas, "speaking_rate", 0.05)
            rules.append("warning")

        if context.environment == "quiet" and noise <= 0.15:
            _apply_delta(deltas, "energy", -0.05)
            _apply_delta(deltas, "pause_density", 0.05)
            rules.append("quiet_environment")

        if context.device == "phone":
            _apply_delta(deltas, "speaking_rate", 0.03)
            _apply_delta(deltas, "confidence", 0.03)
            rules.append("phone")

        if context.device == "desktop" and context.activity == "working":
            _apply_delta(deltas, "seriousness", 0.03)
            rules.append("desktop_working")

        if not deltas:
            resolved = base_expression
        else:
            overrides: dict[str, Any] = {
                "name": f"{base_expression.name}+{context.name}",
            }
            for dim in DIMENSION_NAMES:
                if dim in deltas:
                    base_val = base_expression.to_dict()[dim]
                    clamped = max(0.0, min(1.0, base_val + deltas[dim]))
                    overrides[dim] = clamped
            resolved = merge_expression(base_expression, overrides)

        return ContextResolutionResult(
            base_expression=base_expression,
            context=context,
            resolved_expression=resolved,
            applied_rules=tuple(rules),
        )


_DEFAULT_RESOLVER = ContextResolver()


def resolve_expression_with_context(
    expression: ExpressionProfile,
    context: ContextProfile,
    *,
    resolver: ContextResolver | None = None,
) -> ContextResolutionResult:
    """Resolve base expression under contextual policy."""
    ctx = context if context is not None else DEFAULT_CONTEXT
    engine = resolver or _DEFAULT_RESOLVER
    return engine.resolve(ctx, expression)
