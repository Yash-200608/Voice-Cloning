"""Voice rendering abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class VoiceRenderer(Protocol):
    """Protocol for converting a voice reference into speech."""

    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        output_path: str | Path | None = None,
        *,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
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
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        normalize: bool = True,
    ) -> str:
        raise NotImplementedError


class ChatterboxRenderer(BaseVoiceRenderer):
    """Chatterbox-backed voice renderer."""

    def synthesize(
        self,
        text: str,
        reference_audio: str | Path,
        output_path: str | Path | None = None,
        *,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        normalize: bool = True,
    ) -> str:
        from ..cloner import clone

        return clone(
            text,
            str(reference_audio),
            output_path=str(output_path) if output_path else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            normalize=normalize,
        )
