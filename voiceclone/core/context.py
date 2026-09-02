"""Context-aware voice models, presets, and validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from .exceptions import InvalidContext, UnknownContextPreset

CONTEXT_VERSION = 1

DEVICE_VALUES = frozenset({"desktop", "laptop", "phone", "car", "speaker", "unknown"})
ENVIRONMENT_VALUES = frozenset({"quiet", "normal", "noisy", "very_noisy"})
ACTIVITY_VALUES = frozenset(
    {"idle", "reading", "working", "driving", "walking", "presenting", "unknown"}
)
TIME_OF_DAY_VALUES = frozenset({"morning", "afternoon", "evening", "night"})
INTERACTION_MODE_VALUES = frozenset(
    {"conversational", "notification", "instruction", "warning", "presentation"}
)
AUDIENCE_VALUES = frozenset({"private", "familiar", "public", "unknown"})

CATEGORICAL_FIELDS = {
    "device": DEVICE_VALUES,
    "environment": ENVIRONMENT_VALUES,
    "activity": ACTIVITY_VALUES,
    "time_of_day": TIME_OF_DAY_VALUES,
    "interaction_mode": INTERACTION_MODE_VALUES,
    "audience": AUDIENCE_VALUES,
}


def _clamp01(value: float) -> float:
    if not isinstance(value, (int, float)):
        raise InvalidContext(f"Expected numeric value, got {type(value).__name__}")
    if math.isnan(value) or math.isinf(value):
        raise InvalidContext(f"Invalid numeric value: {value}")
    if value < 0.0 or value > 1.0:
        raise InvalidContext(f"Value {value} out of range [0.0, 1.0]")
    return float(value)


def _validate_category(field: str, value: str) -> str:
    allowed = CATEGORICAL_FIELDS[field]
    key = str(value).strip().lower()
    if key not in allowed:
        raise InvalidContext(
            f"Invalid {field}: {value!r}. Allowed: {sorted(allowed)}",
            user_message=f"Invalid context value for {field.replace('_', ' ')}.",
        )
    return key


@dataclass(frozen=True)
class ContextProfile:
    """Structured situational context for WHY/HOW delivery should adapt."""

    name: str = "default"
    version: int = CONTEXT_VERSION
    device: str = "unknown"
    environment: str = "normal"
    activity: str = "unknown"
    noise_level: float = 0.0
    urgency: float = 0.0
    time_of_day: str = "afternoon"
    interaction_mode: str = "conversational"
    audience: str = "unknown"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise InvalidContext("Context name cannot be empty")
        if self.version < 1:
            raise InvalidContext(f"Invalid context version: {self.version}")
        object.__setattr__(self, "device", _validate_category("device", self.device))
        object.__setattr__(
            self, "environment", _validate_category("environment", self.environment)
        )
        object.__setattr__(self, "activity", _validate_category("activity", self.activity))
        object.__setattr__(
            self, "time_of_day", _validate_category("time_of_day", self.time_of_day)
        )
        object.__setattr__(
            self,
            "interaction_mode",
            _validate_category("interaction_mode", self.interaction_mode),
        )
        object.__setattr__(self, "audience", _validate_category("audience", self.audience))
        object.__setattr__(self, "noise_level", _clamp01(self.noise_level))
        object.__setattr__(self, "urgency", _clamp01(self.urgency))

    @property
    def versioned_name(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def is_default(self) -> bool:
        return self.name == "default" and self == DEFAULT_CONTEXT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextProfile:
        if not isinstance(data, dict):
            raise InvalidContext("Context data must be a dictionary")
        kwargs = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        if not kwargs:
            raise InvalidContext("Context data is missing required fields")
        return cls(**kwargs)


def _preset(name: str, **kwargs: Any) -> ContextProfile:
    return ContextProfile(name=name, version=CONTEXT_VERSION, **kwargs)


DEFAULT_CONTEXT = _preset("default")

BUILTIN_PRESETS: dict[str, ContextProfile] = {
    "default": DEFAULT_CONTEXT,
    "desktop": _preset(
        "desktop",
        device="desktop",
        environment="normal",
        activity="working",
        noise_level=0.1,
        urgency=0.1,
        time_of_day="afternoon",
        interaction_mode="conversational",
        audience="private",
    ),
    "phone": _preset(
        "phone",
        device="phone",
        environment="normal",
        activity="idle",
        noise_level=0.2,
        urgency=0.2,
        interaction_mode="conversational",
        audience="familiar",
    ),
    "car": _preset(
        "car",
        device="car",
        environment="normal",
        activity="driving",
        noise_level=0.45,
        urgency=0.4,
        interaction_mode="instruction",
        audience="familiar",
    ),
    "noisy_environment": _preset(
        "noisy_environment",
        device="unknown",
        environment="noisy",
        activity="unknown",
        noise_level=0.75,
        urgency=0.3,
        interaction_mode="instruction",
        audience="familiar",
    ),
    "quiet_environment": _preset(
        "quiet_environment",
        device="desktop",
        environment="quiet",
        activity="reading",
        noise_level=0.05,
        urgency=0.0,
        interaction_mode="conversational",
        audience="private",
    ),
    "presentation": _preset(
        "presentation",
        device="desktop",
        environment="normal",
        activity="presenting",
        noise_level=0.15,
        urgency=0.2,
        interaction_mode="presentation",
        audience="public",
    ),
    "notification": _preset(
        "notification",
        device="phone",
        environment="normal",
        activity="idle",
        noise_level=0.25,
        urgency=0.5,
        interaction_mode="notification",
        audience="familiar",
    ),
    "warning": _preset(
        "warning",
        device="unknown",
        environment="normal",
        activity="unknown",
        noise_level=0.3,
        urgency=0.85,
        interaction_mode="warning",
        audience="familiar",
    ),
}


def list_context_presets() -> list[str]:
    return sorted(BUILTIN_PRESETS.keys())


def get_context_preset(name: str) -> ContextProfile:
    key = name.strip().lower()
    if "@" in key:
        base, _, ver = key.partition("@")
        try:
            version = int(ver)
        except ValueError as e:
            raise UnknownContextPreset(f"Invalid preset version: {name}") from e
        preset = BUILTIN_PRESETS.get(base)
        if preset is None:
            raise UnknownContextPreset(f"Unknown context preset: {name}")
        if preset.version != version:
            raise UnknownContextPreset(f"Preset version mismatch: {name}")
        return preset
    preset = BUILTIN_PRESETS.get(key)
    if preset is None:
        raise UnknownContextPreset(
            f"Unknown context preset: {name}",
            user_message=f"Unknown context preset '{name}'.",
        )
    return preset


def resolve_context(
    context: str | ContextProfile | dict[str, Any] | None,
) -> ContextProfile:
    """Resolve preset name, profile, dict, or None (default) to ContextProfile."""
    if context is None:
        return DEFAULT_CONTEXT
    if isinstance(context, ContextProfile):
        return context
    if isinstance(context, str):
        return get_context_preset(context)
    if isinstance(context, dict):
        base_name = str(context.get("name", "custom")).strip() or "custom"
        if base_name in BUILTIN_PRESETS and len(context) == 1:
            return get_context_preset(base_name)
        return ContextProfile.from_dict(context)
    raise InvalidContext(f"Unsupported context type: {type(context).__name__}")
