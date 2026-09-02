"""Voice rendering abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..core.expression import ExpressionProfile
    from .chatterbox_mapping import ChatterboxRenderSettings


@runtime_checkable
class VoiceRenderer(Protocol):
    """Protocol for converting a voice reference into speech."""

    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        output_path: str | Path | None = None,
        *,
        expression: ExpressionProfile | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        normalize: bool = True,
    ) -> str:
        ...


class BaseVoiceRenderer(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        output_path: str | Path | None = None,
        *,
        expression: ExpressionProfile | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        normalize: bool = True,
    ) -> str:
        raise NotImplementedError


class ChatterboxRenderer(BaseVoiceRenderer):
    """Chatterbox-backed voice renderer with expression mapping."""

    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        output_path: str | Path | None = None,
        *,
        expression: ExpressionProfile | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        normalize: bool = True,
    ) -> str:
        from ..core.expression import resolve_expression
        from .chatterbox_mapping import map_expression_to_chatterbox
        from .postprocess import apply_expression_postprocess
        from ..cloner import synthesize_with_settings
        from ..audio_utils import load_audio, save_audio

        profile = resolve_expression(expression)
        settings = map_expression_to_chatterbox(profile)

        if exaggeration is not None:
            settings = settings.__class__(
                exaggeration=exaggeration,
                cfg_weight=settings.cfg_weight if cfg_weight is None else cfg_weight,
                speaking_rate_factor=settings.speaking_rate_factor,
                pitch_semitones=settings.pitch_semitones,
                mapping_version=settings.mapping_version,
            )
        elif cfg_weight is not None:
            settings = settings.__class__(
                exaggeration=settings.exaggeration,
                cfg_weight=cfg_weight,
                speaking_rate_factor=settings.speaking_rate_factor,
                pitch_semitones=settings.pitch_semitones,
                mapping_version=settings.mapping_version,
            )

        out_path = synthesize_with_settings(
            text,
            str(reference_audio),
            output_path=str(output_path) if output_path else None,
            exaggeration=settings.exaggeration,
            cfg_weight=settings.cfg_weight,
            normalize=normalize,
        )

        audio, sr = load_audio(out_path)
        processed = apply_expression_postprocess(audio, sr, settings)
        if processed is not audio:
            save_audio(out_path, processed, sr)

        self._last_settings = settings
        self._last_expression = profile
        return out_path

    @property
    def last_render_settings(self) -> ChatterboxRenderSettings | None:
        return getattr(self, "_last_settings", None)

    @property
    def last_expression(self) -> ExpressionProfile | None:
        return getattr(self, "_last_expression", None)
