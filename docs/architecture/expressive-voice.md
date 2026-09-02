# Expressive Voice Architecture

Phase 2 adds an **Expressive Voice Layer** between semantic intent and the voice renderer.

## Concepts

| Layer | Question | Phase |
|-------|----------|-------|
| Voice Identity | WHO is speaking? | Phase 1 |
| Expression | HOW is that identity speaking? | Phase 2 |

Expression does **not** modify the underlying `VoiceIdentity`. It is a temporary rendering instruction applied at synthesis time.

## Pipeline

```text
VoiceIdentity + Text + ExpressionProfile
        ↓
VoiceIdentityService
        ↓
Expression resolution (preset or custom)
        ↓
Renderer mapping (ChatterboxRenderSettings)
        ↓
ChatterboxRenderer → Chatterbox
        ↓
Optional post-processing (rate/pitch)
        ↓
Audio + render metadata
```

## ExpressionProfile

Structured delivery dimensions, normalized to `[0.0, 1.0]`:

- `energy`, `warmth`, `arousal`, `seriousness`, `confidence`, `urgency`
- `expressiveness`, `speaking_rate`, `pitch_shift`, `pause_density`

Custom profiles can override any dimension. Presets are convenience starting points.

## Built-in Presets (versioned)

| Preset | Purpose |
|--------|---------|
| `neutral` | Default balanced delivery |
| `calm` | Low energy, measured pace |
| `warm` | Friendly warmth, softer delivery |
| `friendly` | Upbeat, approachable |
| `professional` | Confident, restrained |
| `serious` | Formal, low expressiveness |
| `excited` | High energy and arousal |
| `concerned` | Measured, serious undertone |
| `urgent` | Fast, high urgency |

Presets are versioned (`calm@1`). Mapping improvements create new mapping versions without silently changing preset semantics.

## Renderer Mapping

Only `voiceclone/inference/chatterbox_mapping.py` translates `ExpressionProfile` into Chatterbox parameters:

```python
ChatterboxRenderSettings(
    exaggeration=...,
    cfg_weight=...,
    speaking_rate_factor=...,
    pitch_semitones=...,
)
```

The public API never exposes `exaggeration` or `cfg_weight` as the primary concept.

## Identity Preservation

- Speaker similarity still uses the **cached identity embedding** from Phase 1.
- Similarity measures **identity fidelity**, not expressive correctness.
- Best-of-N selection optimizes identity fidelity, not expression quality.

## Generation Metadata

Each output may have a sidecar file:

```text
out_abc123.wav
out_abc123.wav.meta.json
```

Contains: `identity_id`, `expression_name`, `expression_profile`, `renderer`, `generation_parameters`, `timestamp`, optional `speaker_similarity`.

This is a **rendering event**, not identity metadata.

## API Usage

```python
from voiceclone import VoiceIdentityService

service = VoiceIdentityService()

# Default (neutral)
service.synthesize(identity_id, "Hello.")

# Preset
service.synthesize(identity_id, "Hello.", expression="calm")

# Custom profile
from voiceclone import ExpressionProfile
profile = ExpressionProfile(name="custom", energy=0.8, warmth=0.7)
service.synthesize(identity_id, "Hello.", expression=profile)

# Best-of-3 with expression
service.synthesize_best_of(identity_id, "Hello.", n=3, expression="excited")
```

## Evaluation

Run the expressive evaluation script:

```bash
python scripts/evaluate_expressions.py --identity-id voice_<uuid>
```

Produces a CSV with generation time, speaker similarity, duration, and blank columns for human ratings:

- identity fidelity
- intelligibility
- expressive distinction
- naturalness
- artifact presence

**Do not** interpret similarity as an emotion quality score.

## Current Limitations

- Expressive control is implemented via Chatterbox parameter mapping and light post-processing.
- `pause_density` is stored in profiles but not yet applied to audio.
- Preset mappings are configurable heuristics, not scientifically verified emotions.
- Expressive distinction requires human listening evaluation.
- First Chatterbox run requires model download and suitable hardware.

## Future Compatibility (Phase 3+)

Phase 3 will insert context-aware expression selection **before** `ExpressionProfile`:

```text
Context → Expression Selection → ExpressionProfile → Renderer
```

Phase 2 exposes clean `ExpressionProfile` inputs without automatic selection.
