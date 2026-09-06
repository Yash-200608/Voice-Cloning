"""Application service for voice identity operations."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from ..Config import REFERENCE_SECONDS, SAMPLE_RATE
from ..audio_utils import audio_info, preprocess_reference
from ..compatibility.legacy import migrate_legacy_voices
from ..core.context import (
    ContextProfile,
    get_context_preset,
    list_context_presets,
    resolve_context,
)
from ..core.context_resolver import (
    ContextResolutionResult,
    ContextResolver,
    resolve_expression_with_context,
)
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
from ..memory.resolver import FullRenderPlan, VoiceMemoryResolver
from ..memory.service import VoiceMemoryService
from ..recorder import capture_to_path, import_to_path

logger = logging.getLogger(__name__)


class VoiceIdentityService:
    """Coordinates repository, audio, embeddings, context, expression, and rendering."""

    def __init__(
        self,
        repository: VoiceRepository | None = None,
        renderer: VoiceRenderer | None = None,
        embedding_store: EmbeddingStore | None = None,
        memory_service: VoiceMemoryService | None = None,
        *,
        auto_migrate: bool = True,
        context_resolver: ContextResolver | None = None,
        memory_resolver: VoiceMemoryResolver | None = None,
        debug_context: bool = False,
        debug_memory: bool = False,
    ):
        self.repository = repository or VoiceRepository()
        self.embedding_store = embedding_store or self.repository.embedding_store
        self.renderer = renderer or ChatterboxRenderer()
        self.memory = memory_service or VoiceMemoryService(voice_repository=self.repository)
        self._auto_migrate = auto_migrate
        self._context_resolver = context_resolver or ContextResolver()
        self._memory_resolver = memory_resolver or VoiceMemoryResolver(self._context_resolver)
        self._debug_context = debug_context
        self._debug_memory = debug_memory
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

    def list_context_presets(self) -> list[str]:
        return list_context_presets()

    def get_context_preset(self, name: str) -> ContextProfile:
        return get_context_preset(name)

    def resolve_render_plan(
        self,
        expression: str | ExpressionProfile | dict[str, Any] | None = None,
        context: str | ContextProfile | dict[str, Any] | None = None,
        *,
        identity_id: str | None = None,
        text: str = "",
        use_memory: bool = True,
    ) -> FullRenderPlan:
        """Resolve memory, expression, and context into a full rendering plan.

        Phase 3 callers that omit identity_id/text continue to work: memory is
        skipped and pronunciation leaves normalized text unchanged.
        """
        expression_explicit = expression is not None
        base = resolve_expression(expression)
        ctx = resolve_context(context)
        memories = []
        if use_memory and identity_id:
            memories = self.memory.resolve_memories(identity_id)
        return self._memory_resolver.resolve_full(
            text,
            base,
            ctx,
            memories,
            use_memory=use_memory and bool(identity_id),
            expression_explicit=expression_explicit,
        )

    def add_memory(self, identity_id: str, category: str, key: str, value: dict[str, Any], **kwargs):
        return self.memory.add_memory(identity_id, category, key, value, **kwargs)

    def add_pronunciation_memory(self, identity_id: str, term: str, pronunciation_value: str, **kwargs):
        return self.memory.add_pronunciation_memory(identity_id, term, pronunciation_value, **kwargs)

    def add_style_preference(self, identity_id: str, dimension: str, preference: float, **kwargs):
        return self.memory.add_style_preference(identity_id, dimension, preference, **kwargs)

    def list_memories(self, identity_id: str, *, enabled_only: bool = False):
        return self.memory.list_memories(identity_id, enabled_only=enabled_only)

    def get_memory(self, identity_id: str, memory_id: str):
        return self.memory.get_memory(identity_id, memory_id)

    def update_memory(self, identity_id: str, memory_id: str, **kwargs):
        return self.memory.update_memory(identity_id, memory_id, **kwargs)

    def delete_memory(self, identity_id: str, memory_id: str) -> bool:
        return self.memory.delete_memory(identity_id, memory_id)

    def enable_memory(self, identity_id: str, memory_id: str):
        return self.memory.enable_memory(identity_id, memory_id)

    def disable_memory(self, identity_id: str, memory_id: str):
        return self.memory.disable_memory(identity_id, memory_id)

    def export_memories(self, identity_id: str) -> dict[str, Any]:
        return self.memory.export_memories(identity_id)

    def import_memories(self, identity_id: str, payload: dict[str, Any], *, replace: bool = False):
        return self.memory.import_memories(identity_id, payload, replace=replace)

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
        context: str | ContextProfile | dict[str, Any] | None = None,
        use_memory: bool = True,
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

        plan = self.resolve_render_plan(
            expression,
            context,
            identity_id=identity_id,
            text=text,
            use_memory=use_memory,
        )

        if output_path is None:
            out_dir = self.repository.output_dir(identity_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"out_{uuid.uuid4().hex[:8]}.wav"

        try:
            result = self.renderer.synthesize(
                plan.render_text,
                processed,
                output_path,
                expression=plan.resolved_expression,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        except Exception as e:
            raise VoiceRenderError(
                f"Synthesis failed: {e}",
                user_message="Speech generation failed. Please try again.",
            ) from e

        if save_metadata:
            self._write_render_metadata(identity, plan, result, use_memory=use_memory)
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
        context: str | ContextProfile | dict[str, Any] | None = None,
        use_memory: bool = True,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> tuple[str, float]:
        identity = self.repository.get(identity_id)
        embedding_path = self.repository.resolve_path(identity, identity.embedding_path)
        ref_embedding = self._load_cached_embedding(identity, embedding_path)
        plan = self.resolve_render_plan(
            expression,
            context,
            identity_id=identity_id,
            text=text,
            use_memory=use_memory,
        )

        from ..similarity import compare_with_embedding

        candidates: list[tuple[float, str]] = []
        for _ in range(n):
            path = self.synthesize(
                identity_id,
                text,
                expression=expression,
                context=context,
                use_memory=use_memory,
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

        self._write_render_metadata(
            identity, plan, best_path, similarity=best_score, use_memory=use_memory
        )
        return best_path, best_score

    def benchmark(
        self,
        identity_id: str,
        sentences: list[str] | None = None,
        csv_path: str | Path | None = None,
        expression: str | ExpressionProfile | dict[str, Any] | None = None,
        contexts: list[str | ContextProfile | dict[str, Any] | None] | None = None,
        *,
        use_memory: bool = True,
        compare_memory: bool = False,
    ) -> dict:
        from ..benchmarking import run_benchmark

        identity = self.repository.get(identity_id)
        processed = self.repository.resolve_path(identity, identity.processed_audio)
        embedding_path = self.repository.resolve_path(identity, identity.embedding_path)
        ref_embedding = self._load_cached_embedding(identity, embedding_path)
        sample = (sentences or ["Benchmark sentence."])[0]

        def _run(enabled: bool, context, out_csv, label: str) -> dict:
            plan = self.resolve_render_plan(
                expression,
                context,
                identity_id=identity_id,
                text=sample,
                use_memory=enabled,
            )
            settings = map_expression_to_chatterbox(plan.resolved_expression)

            def synthesize_fn(text: str) -> str:
                return self.synthesize(
                    identity_id,
                    text,
                    expression=expression,
                    context=context,
                    use_memory=enabled,
                    save_metadata=False,
                )

            return run_benchmark(
                processed_audio=str(processed),
                reference_embedding=ref_embedding,
                sentences=sentences,
                csv_path=out_csv,
                output_dir=self.repository.output_dir(identity_id),
                synthesize_fn=synthesize_fn,
                context_label=plan.context.versioned_name,
                base_expression_label=plan.base_expression.versioned_name,
                resolved_expression_label=plan.resolved_expression.versioned_name,
                memory_label=label,
                use_memory=enabled,
                memories_consulted=len(plan.memory.memories_consulted),
                memory_resolution_time_ms=plan.memory.resolution_time_ms,
            )

        if compare_memory:
            results = {}
            for label, enabled in (("memory_disabled", False), ("memory_enabled", True)):
                out_csv = None
                if csv_path is not None:
                    out_csv = Path(csv_path).with_name(
                        f"{Path(csv_path).stem}_{label}{Path(csv_path).suffix}"
                    )
                results[label] = _run(enabled, None, out_csv, label)
            return results

        context_list = contexts if contexts is not None else [None]
        if len(context_list) == 1:
            return _run(
                use_memory,
                context_list[0],
                csv_path,
                "enabled" if use_memory else "disabled",
            )

        summaries = []
        for ctx in context_list:
            plan = self.resolve_render_plan(
                expression,
                ctx,
                identity_id=identity_id,
                text=sample,
                use_memory=use_memory,
            )
            out_csv = None
            if csv_path is not None:
                stem = Path(csv_path).stem
                suffix = plan.context.name
                out_csv = Path(csv_path).with_name(f"{stem}_{suffix}{Path(csv_path).suffix}")
            result = _run(
                use_memory,
                ctx,
                out_csv,
                "enabled" if use_memory else "disabled",
            )
            summaries.append({"context": plan.context.versioned_name, **result})
        return {"contexts": summaries}

    def _write_render_metadata(
        self,
        identity: VoiceIdentity,
        plan: FullRenderPlan,
        audio_path: str,
        similarity: float | None = None,
        *,
        use_memory: bool = True,
    ) -> None:
        profile = plan.resolved_expression
        settings = map_expression_to_chatterbox(profile)
        try:
            from ..audio_utils import audio_info
            duration = audio_info(audio_path)["duration_seconds"]
        except Exception:
            duration = None

        extra: dict[str, Any] = {
            "base_expression_name": plan.base_expression.versioned_name,
            "base_expression_profile": plan.base_expression.to_dict(),
            "context_name": plan.context.versioned_name,
            "context_version": plan.context.version,
            "context_profile": plan.context.to_dict(),
            "resolved_expression_name": plan.resolved_expression.versioned_name,
            "context_policy_version": plan.policy_version,
            "applied_context_rules": list(plan.applied_rules),
            "memory_items_used": list(plan.memory.memories_applied),
            "memory_items_consulted": list(plan.memory.memories_consulted),
            "memory_resolution_version": plan.memory.policy_version,
            "memory_resolution_time_ms": plan.memory.resolution_time_ms,
            "use_memory": use_memory,
            "render_text": plan.render_text,
        }
        if self._debug_context:
            extra["context_resolution"] = plan.context_result.to_dict()
        if self._debug_memory:
            extra["memory_resolution"] = plan.memory.to_dict()

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
            extra=extra,
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
