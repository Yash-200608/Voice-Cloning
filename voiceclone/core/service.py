"""Application service for voice identity operations."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from ..Config import REFERENCE_SECONDS, SAMPLE_RATE
from ..audio_utils import audio_info, preprocess_reference
from ..compatibility.legacy import migrate_legacy_voices
from ..core.exceptions import (
    InvalidVoiceIdentity,
    MissingEmbedding,
    MissingReferenceAudio,
    VoiceIdentityNotFound,
    VoiceRenderError,
)
from ..core.expression import (
    ExpressionProfile,
    get_expression_preset,
    list_expression_presets,
    resolve_expression,
)
from ..core.models import VoiceIdentity, utc_now
from ..evaluation.render_metadata import save_render_metadata
from ..identity.embeddings import EmbeddingStore, get_embedding_version
from ..identity.repository import VoiceRepository, get_renderer_version
from ..inference.chatterbox_mapping import map_expression_to_chatterbox
from ..inference.renderer import ChatterboxRenderer, VoiceRenderer
from ..recorder import capture_to_path, import_to_path

logger = logging.getLogger(__name__)


class VoiceIdentityService:
    """Coordinates repository, audio, embeddings, expression, and rendering."""

    def __init__(
        self,
        repository: VoiceRepository | None = None,
        renderer: VoiceRenderer | None = None,
        embedding_store: EmbeddingStore | None = None,
        *,
        auto_migrate: bool = True,
    ):
        self.repository = repository or VoiceRepository()
        self.embedding_store = embedding_store or self.repository.embedding_store
        self.renderer = renderer or ChatterboxRenderer()
        self._auto_migrate = auto_migrate
        if auto_migrate:
            migrate_legacy_voices(self.repository)

    def list_identities(self) -> list[VoiceIdentity]:
        if self._auto_migrate:
            migrate_legacy_voices(self.repository)
        return self.repository.list_identities()

    def get_identity(self, identity_id: str) -> VoiceIdentity:
        return self.repository.get(identity_id)

    def get_identity_by_name(self, name: str) -> VoiceIdentity:
        return self.repository.get_by_name(name)

    def list_expression_presets(self) -> list[str]:
        return list_expression_presets()

    def get_expression_preset(self, name: str) -> ExpressionProfile:
        return get_expression_preset(name)

    def create_from_recording(
        self,
        name: str,
        duration: float = REFERENCE_SECONDS,
        sr: int = SAMPLE_RATE,
    ) -> VoiceIdentity:
        with self.repository.staged_creation(name) as staged:
            capture_to_path(staged.raw_path, duration=duration, sr=sr)
            preprocess_reference(staged.raw_path, out_path=staged.processed_path)
            return self._finalize_staged(staged)

    def create_from_file(self, name: str, source_path: str | Path) -> VoiceIdentity:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Reference file not found: {source}")

        with self.repository.staged_creation(name) as staged:
            import_to_path(source, staged.raw_path)
            preprocess_reference(
                staged.raw_path,
                out_path=staged.processed_path,
                max_seconds=REFERENCE_SECONDS,
            )
            return self._finalize_staged(staged)

    def rename_identity(self, identity_id: str, new_name: str) -> VoiceIdentity:
        return self.repository.rename(identity_id, new_name)

    def delete_identity(self, identity_id: str) -> bool:
        return self.repository.delete(identity_id)

    def rebuild_embedding(self, identity_id: str) -> VoiceIdentity:
        identity = self.repository.get(identity_id)
        processed = self.repository.resolve_path(identity, identity.processed_audio)
        embedding_path = self.repository.resolve_path(identity, identity.embedding_path)
        self.embedding_store.generate(processed, embedding_path)
        identity.embedding_version = get_embedding_version()
        identity.updated_at = utc_now()
        from ..identity.metadata import save_metadata
        save_metadata(self.repository.identity_dir(identity_id) / "metadata.json", identity)
        return identity

    def synthesize(
        self,
        identity_id: str,
        text: str,
        output_path: str | Path | None = None,
        *,
        expression: str | ExpressionProfile | dict[str, Any] | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        save_metadata: bool = True,
    ) -> str:
        identity = self.repository.get(identity_id)
        processed = self.repository.resolve_path(identity, identity.processed_audio)
        if not processed.exists():
            raise MissingReferenceAudio(
                f"Processed reference missing for {identity_id}",
                user_message="Reference audio is missing for this voice identity.",
            )

        profile = resolve_expression(expression)

        if output_path is None:
            out_dir = self.repository.output_dir(identity_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"out_{uuid.uuid4().hex[:8]}.wav"

        try:
            result = self.renderer.synthesize(
                text,
                processed,
                output_path,
                expression=profile,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        except Exception as e:
            raise VoiceRenderError(
                f"Synthesis failed: {e}",
                user_message="Speech generation failed. Please try again.",
            ) from e

        if save_metadata:
            self._write_render_metadata(identity, profile, result)
        return result

    def compare(self, identity_id: str, generated_audio: str | Path) -> float:
        from ..similarity import compare_with_embedding

        identity = self.repository.get(identity_id)
        embedding_path = self.repository.resolve_path(identity, identity.embedding_path)

        try:
            ref_embedding = self._load_cached_embedding(identity, embedding_path)
        except (MissingEmbedding, InvalidVoiceIdentity):
            logger.info("Rebuilding missing/invalid embedding for %s", identity_id)
            self.rebuild_embedding(identity_id)
            identity = self.repository.get(identity_id)
            embedding_path = self.repository.resolve_path(identity, identity.embedding_path)
            ref_embedding = self._load_cached_embedding(identity, embedding_path)

        return compare_with_embedding(ref_embedding, str(generated_audio))

    def synthesize_best_of(
        self,
        identity_id: str,
        text: str,
        n: int = 3,
        *,
        expression: str | ExpressionProfile | dict[str, Any] | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> tuple[str, float]:
        identity = self.repository.get(identity_id)
        embedding_path = self.repository.resolve_path(identity, identity.embedding_path)
        ref_embedding = self._load_cached_embedding(identity, embedding_path)
        profile = resolve_expression(expression)

        from ..similarity import compare_with_embedding

        candidates: list[tuple[float, str]] = []
        for _ in range(n):
            path = self.synthesize(
                identity_id,
                text,
                expression=profile,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                save_metadata=False,
            )
            score = compare_with_embedding(ref_embedding, path)
            candidates.append((score, path))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path = candidates[0]

        for _, path in candidates[1:]:
            try:
                Path(path).unlink()
                meta = Path(str(path) + ".meta.json")
                if meta.exists():
                    meta.unlink()
            except OSError:
                pass

        self._write_render_metadata(identity, profile, best_path, similarity=best_score)
        return best_path, best_score

    def benchmark(
        self,
        identity_id: str,
        sentences: list[str] | None = None,
        csv_path: str | Path | None = None,
        expression: str | ExpressionProfile | dict[str, Any] | None = None,
    ) -> dict:
        from ..benchmarking import run_benchmark

        identity = self.repository.get(identity_id)
        processed = self.repository.resolve_path(identity, identity.processed_audio)
        embedding_path = self.repository.resolve_path(identity, identity.embedding_path)
        ref_embedding = self._load_cached_embedding(identity, embedding_path)
        profile = resolve_expression(expression)
        settings = map_expression_to_chatterbox(profile)

        return run_benchmark(
            processed_audio=str(processed),
            reference_embedding=ref_embedding,
            sentences=sentences,
            csv_path=csv_path,
            output_dir=self.repository.output_dir(identity_id),
            render_kwargs={
                "exaggeration": settings.exaggeration,
                "cfg_weight": settings.cfg_weight,
            },
        )

    def _write_render_metadata(
        self,
        identity: VoiceIdentity,
        profile: ExpressionProfile,
        audio_path: str,
        similarity: float | None = None,
    ) -> None:
        settings = map_expression_to_chatterbox(profile)
        try:
            from ..audio_utils import audio_info
            duration = audio_info(audio_path)["duration_seconds"]
        except Exception:
            duration = None

        save_render_metadata(
            audio_path,
            identity_id=identity.id,
            identity_name=identity.name,
            expression=profile,
            renderer=identity.renderer_model,
            renderer_version=get_renderer_version(),
            generation_parameters=settings.to_dict(),
            similarity=similarity,
            duration_seconds=duration,
        )

    def _finalize_staged(self, staged) -> VoiceIdentity:
        info = audio_info(staged.processed_path)
        self.embedding_store.generate(staged.processed_path, staged.embedding_path)
        identity = staged.build_identity(
            sample_rate=info["sample_rate"],
            duration_seconds=info["duration_seconds"],
        )
        return self.repository.publish_staged(staged, identity)

    def _load_cached_embedding(self, identity: VoiceIdentity, embedding_path: Path):
        current_version = get_embedding_version()
        if identity.embedding_version != current_version:
            raise InvalidVoiceIdentity(
                f"Embedding version mismatch for {identity.id}: "
                f"stored={identity.embedding_version}, current={current_version}"
            )
        return self.embedding_store.validate(
            embedding_path,
            embedding_model=identity.embedding_model,
            embedding_version=identity.embedding_version,
        )
