import csv
import time
import statistics
import uuid
from pathlib import Path

from .cloner import clone
from .similarity import compare, compare_with_embedding
from .Config import OUTPUTS_DIR, ensure_directories

DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "She sells seashells by the seashore on a bright summer afternoon.",
    "Innovation requires both imagination and the discipline to execute carefully.",
    "Yesterday I walked to the market and bought apples, bread, and fresh milk.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
]


def _wer(reference, hypothesis):
    r = reference.lower().split()
    h = hypothesis.lower().split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    if not r:
        return 0.0
    return d[len(r)][len(h)] / len(r)


_whisper = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        try:
            from faster_whisper import WhisperModel
            print("Loading Whisper for WER (first run only)...")
            _whisper = WhisperModel("base", device="cpu", compute_type="int8")
        except ImportError:
            return None
    return _whisper


def _transcribe(path):
    model = _get_whisper()
    if model is None:
        return None
    segments, _ = model.transcribe(path, beam_size=1)
    return " ".join(s.text for s in segments).strip()


def run_benchmark(
    processed_audio: str,
    reference_embedding=None,
    sentences=None,
    csv_path=None,
    output_dir: Path | None = None,
    render_kwargs: dict | None = None,
    *,
    context_label: str | None = None,
    base_expression_label: str | None = None,
    resolved_expression_label: str | None = None,
) -> dict:
    """Run benchmark against a processed reference path."""
    ensure_directories()
    sentences = sentences or DEFAULT_SENTENCES
    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is None:
        csv_path = out_dir / f"benchmark_{int(time.time())}.csv"

    rows = []
    sims = []
    wers = []
    times = []

    render_kwargs = render_kwargs or {}

    for i, text in enumerate(sentences):
        t0 = time.time()
        out = clone(
            text,
            processed_audio,
            output_path=out_dir / f"bench_{uuid.uuid4().hex[:8]}.wav",
            **render_kwargs,
        )
        gen_time = round(time.time() - t0, 2)

        if reference_embedding is not None:
            sim = compare_with_embedding(reference_embedding, out)
        else:
            sim = compare(processed_audio, out)

        transcript = _transcribe(out)
        wer = _wer(text, transcript) if transcript else None

        sims.append(sim)
        times.append(gen_time)
        if wer is not None:
            wers.append(wer)

        row = {
            "idx": i,
            "text": text,
            "output": out,
            "similarity": round(sim, 4),
            "wer": round(wer, 4) if wer is not None else "",
            "transcript": transcript or "",
            "gen_time_s": gen_time,
        }
        if context_label is not None:
            row["context"] = context_label
        if base_expression_label is not None:
            row["base_expression"] = base_expression_label
        if resolved_expression_label is not None:
            row["resolved_expression"] = resolved_expression_label
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "n": len(sentences),
        "similarity_mean": round(statistics.mean(sims), 4),
        "similarity_std": round(statistics.stdev(sims), 4) if len(sims) > 1 else 0.0,
        "wer_mean": round(statistics.mean(wers), 4) if wers else None,
        "time_mean_s": round(statistics.mean(times), 2),
        "csv": str(csv_path),
    }


def benchmark(voice_file, sentences=None, csv_path=None):
    """Compatibility API: benchmark using a voice file path."""
    return run_benchmark(
        processed_audio=voice_file,
        reference_embedding=None,
        sentences=sentences,
        csv_path=csv_path,
    )
