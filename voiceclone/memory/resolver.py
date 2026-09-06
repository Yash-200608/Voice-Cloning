"""Deterministic voice-memory resolution for synthesis inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.context import ContextProfile
from ..core.context_resolver import (
    ContextResolutionResult,
    ContextResolver,
    resolve_expression_with_context,
)
from ..core.expression import ExpressionProfile, merge_expression
from ..text_utils import normalize_text
from .models import VoiceMemoryItem
from .pronunciation import apply_pronunciation_memories

MEMORY_RESOLUTION_VERSION = "1"


@dataclass(frozen=True)
class MemoryResolutionResult:
    """Outcome of applying voice memory to text and expression baseline."""

    original_text: str
    normalized_text: str
    render_text: str
    base_expression: ExpressionProfile
    memory_adjusted_expression: ExpressionProfile
    memories_consulted: tuple[str, ...] = ()
    memories_applied: tuple[str, ...] = ()
    memories_ignored: tuple[tuple[str, str], ...] = ()
    policy_version: str = MEMORY_RESOLUTION_VERSION
    resolution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "render_text": self.render_text,
            "base_expression_name": self.base_expression.versioned_name,
            "memory_adjusted_expression_name": self.memory_adjusted_expression.versioned_name,
            "memories_consulted": list(self.memories_consulted),
            "memories_applied": list(self.memories_applied),
            "memories_ignored": [
                {"memory_id": memory_id, "reason": reason}
                for memory_id, reason in self.memories_ignored
            ],
            "memory_resolution_version": self.policy_version,
            "memory_resolution_time_ms": self.resolution_time_ms,
        }


@dataclass(frozen=True)
class FullRenderPlan:
    """Combined memory + context rendering plan.

    Backward-compatible accessors mirror ContextResolutionResult fields used by
    Phase 3 callers (base_expression, context, resolved_expression, etc.).
    """

    memory: MemoryResolutionResult
    context_result: ContextResolutionResult

    @property
    def resolved_expression(self) -> ExpressionProfile:
        return self.context_result.resolved_expression

    @property
    def base_expression(self) -> ExpressionProfile:
        return self.memory.base_expression

    @property
    def context(self) -> ContextProfile:
        return self.context_result.context

    @property
    def applied_rules(self) -> tuple[str, ...]:
        return self.context_result.applied_rules

    @property
    def policy_version(self) -> str:
        return self.context_result.policy_version

    @property
    def render_text(self) -> str:
        return self.memory.render_text

    def summary(self) -> str:
        applied = len(self.memory.memories_applied)
        return f"{self.context_result.summary()} | memory applied: {applied}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "context": self.context_result.to_dict(),
            "render_text": self.render_text,
            "resolved_expression_name": self.resolved_expression.versioned_name,
        }


class VoiceMemoryResolver:
    """Applies identity-specific memory before context resolution.

    Precedence (deterministic):
      1. Explicit current expression request
      2. Context-derived adjustment
      3. Persistent voice preference memory (style defaults only when expression omitted)
      4. Built-in defaults

    Pronunciation/name/vocabulary memories always apply at render time when enabled
    and matched, because they transform text rather than override expression intent.

    Pipeline ordering:
      raw text → normalize_text → pronunciation memory → style memory defaults
      → context resolution → final expression → renderer
    """

    def __init__(self, context_resolver: ContextResolver | None = None):
        self._context_resolver = context_resolver or ContextResolver()

    def resolve_memory(
        self,
        text: str,
        base_expression: ExpressionProfile,
        memories: list[VoiceMemoryItem],
        *,
        use_memory: bool = True,
        expression_explicit: bool = False,
    ) -> MemoryResolutionResult:
        import time

        started = time.perf_counter()
        normalized = normalize_text(text)
        if not use_memory or not memories:
            return MemoryResolutionResult(
                original_text=text,
                normalized_text=normalized,
                render_text=normalized,
                base_expression=base_expression,
                memory_adjusted_expression=base_expression,
                memories_consulted=(),
                memories_applied=(),
                resolution_time_ms=round((time.perf_counter() - started) * 1000, 3),
            )

        enabled = [item for item in memories if item.enabled]
        render_text, pronunciation_ids = apply_pronunciation_memories(normalized, enabled)
        adjusted, style_ids, style_ignored = self._apply_style_preferences(
            base_expression,
            enabled,
            expression_explicit=expression_explicit,
        )

        applied = list(dict.fromkeys([*pronunciation_ids, *style_ids]))
        ignored: list[tuple[str, str]] = list(style_ignored)
        for item in enabled:
            if item.id in applied or any(item.id == mid for mid, _ in ignored):
                continue
            if item.category in {"pronunciation", "name", "vocabulary"}:
                ignored.append((item.id, "no_text_match"))
            elif item.category == "delivery_preference" and "preference" not in item.value:
                ignored.append((item.id, "descriptive_only"))
            else:
                ignored.append((item.id, "not_applicable"))

        return MemoryResolutionResult(
            original_text=text,
            normalized_text=normalized,
            render_text=render_text,
            base_expression=base_expression,
            memory_adjusted_expression=adjusted,
            memories_consulted=tuple(item.id for item in enabled),
            memories_applied=tuple(applied),
            memories_ignored=tuple(ignored),
            resolution_time_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def resolve_full(
        self,
        text: str,
        base_expression: ExpressionProfile,
        context: ContextProfile,
        memories: list[VoiceMemoryItem],
        *,
        use_memory: bool = True,
        expression_explicit: bool = False,
    ) -> FullRenderPlan:
        memory_result = self.resolve_memory(
            text,
            base_expression,
            memories,
            use_memory=use_memory,
            expression_explicit=expression_explicit,
        )
        context_result = resolve_expression_with_context(
            memory_result.memory_adjusted_expression,
            context,
            resolver=self._context_resolver,
        )
        # Preserve the caller's explicit base expression in context provenance.
        context_result = ContextResolutionResult(
            base_expression=base_expression,
            context=context_result.context,
            resolved_expression=context_result.resolved_expression,
            applied_rules=context_result.applied_rules,
            policy_version=context_result.policy_version,
        )
        return FullRenderPlan(memory=memory_result, context_result=context_result)

    def _apply_style_preferences(
        self,
        base: ExpressionProfile,
        memories: list[VoiceMemoryItem],
        *,
        expression_explicit: bool,
    ) -> tuple[ExpressionProfile, list[str], list[tuple[str, str]]]:
        """Apply style preferences as defaults when no explicit expression was requested.

        When the caller provided an expression, style memories are ignored with reason
        ``explicitly_overridden`` so memory never silently overrides current intent.
        """
        style_items: list[VoiceMemoryItem] = []
        ignored: list[tuple[str, str]] = []
        for item in memories:
            is_style = item.category == "style_preference"
            is_delivery_pref = (
                item.category == "delivery_preference" and "preference" in item.value
            )
            if not (is_style or is_delivery_pref):
                continue
            if expression_explicit:
                ignored.append((item.id, "explicitly_overridden"))
                continue
            if is_delivery_pref and not item.value.get("dimension"):
                ignored.append((item.id, "descriptive_only"))
                continue
            style_items.append(item)

        if not style_items:
            return base, [], ignored

        overrides: dict[str, Any] = {"name": f"{base.name}+memory"}
        applied_ids: list[str] = []
        for item in style_items:
            dim = str(item.value["dimension"])
            preference = float(item.value["preference"])
            current = base.to_dict()[dim]
            blended = current + (preference - current) * item.confidence
            overrides[dim] = max(0.0, min(1.0, blended))
            applied_ids.append(item.id)

        return merge_expression(base, overrides), applied_ids, ignored
