# Context-Aware Voice (Phase 3)

Phase 3 adds **situational context** to the voice rendering pipeline. Context answers *why* or *in what situation* an identity should speak a certain way — without changing the underlying voice identity.

## Semantic distinction

| Layer | Question | Example |
|-------|----------|---------|
| **Voice Identity** | WHO is speaking? | Yash |
| **Expression** | HOW is the identity speaking? | calm, excited, professional |
| **Context** | WHAT situation applies? | car, noisy environment, presentation |

Context influences expressive delivery. It does **not** create new identities, modify embeddings, or bypass the renderer abstraction.

## Pipeline

```text
VoiceIdentity
      +
Text
      +
Base Expression (preset or custom)
      +
Context (preset or structured)
      ↓
ContextResolver (policy v1)
      ↓
Resolved ExpressionProfile
      ↓
Chatterbox mapping adapter
      ↓
ChatterboxRenderer
      ↓
Audio + .meta.json
```

Context never maps directly to Chatterbox parameters (`exaggeration`, `cfg_weight`). That mapping remains expression-only (Phase 2).

## Context model

`ContextProfile` fields:

| Field | Type | Description |
|-------|------|-------------|
| `device` | enum | `desktop`, `laptop`, `phone`, `car`, `speaker`, `unknown` |
| `environment` | enum | `quiet`, `normal`, `noisy`, `very_noisy` |
| `activity` | enum | `idle`, `reading`, `working`, `driving`, `walking`, `presenting`, `unknown` |
| `noise_level` | 0.0–1.0 | Normalized ambient noise |
| `urgency` | 0.0–1.0 | Situational urgency |
| `time_of_day` | enum | `morning`, `afternoon`, `evening`, `night` |
| `interaction_mode` | enum | `conversational`, `notification`, `instruction`, `warning`, `presentation` |
| `audience` | enum | `private`, `familiar`, `public`, `unknown` |

All normalized numeric fields are validated and clamped. Invalid categories raise `InvalidContext`.

## Built-in context presets

| Preset | Typical use |
|--------|-------------|
| `default` | Neutral/default — no contextual adjustment |
| `desktop` | Desktop workstation |
| `phone` | Mobile handset |
| `car` | In-vehicle / driving |
| `noisy_environment` | High background noise |
| `quiet_environment` | Low-noise private space |
| `presentation` | Formal delivery to an audience |
| `notification` | Short alert-style delivery |
| `warning` | High-urgency alert |

Presets are versioned (`car@1`). They are rendering conditions, not identities.

## Resolution policy

`ContextResolver.resolve(context, base_expression)` applies **deterministic, bounded adjustments** to the base expression.

### Precedence

1. **Explicit expression** establishes the baseline (user preset or custom profile).
2. **Context** applies bounded deltas (max ±0.25 per dimension per rule).
3. All results are clamped to valid `[0.0, 1.0]` ranges.

Context does not silently replace user expression intent. A `calm` expression under `warning` context becomes a context-adjusted calm delivery, not an unrelated preset.

### Example rules (policy v1)

| Condition | Typical adjustments |
|-----------|---------------------|
| High noise | +energy, +expressiveness, +speaking_rate, −pause_density |
| Driving / car | +confidence, +seriousness, −pause_density |
| High urgency | +urgency, +energy, +seriousness, +speaking_rate |
| Night | −energy, −arousal |
| Presentation | +confidence, +seriousness, deliberate pacing |
| Warning | +urgency, +energy, +seriousness |

These are heuristics, not scientifically verified situational models.

## API usage

```python
from voiceclone import VoiceIdentityService, ContextProfile

service = VoiceIdentityService()

# No context (Phase 2 behavior preserved)
service.synthesize(identity_id, "Hello.")

# Context only (neutral base expression)
service.synthesize(identity_id, "Please be careful.", context="car")

# Expression + context
service.synthesize(
    identity_id,
    "Good morning.",
    expression="warm",
    context=ContextProfile(device="desktop", environment="quiet", activity="working"),
)

# Inspect resolution plan
plan = service.resolve_render_plan(expression="calm", context="car")
print(plan.summary())
```

## Generation metadata

Each output `.wav.meta.json` records:

- `base_expression_name` / `base_expression_profile`
- `context_name` / `context_profile`
- `resolved_expression_name` / `expression_profile` (resolved)
- `context_policy_version`
- `applied_context_rules`
- Standard Phase 1/2 fields (identity, renderer, generation parameters)

Optional `context_resolution` diagnostics appear when `VoiceIdentityService(debug_context=True)`.

## Best-of-N

Best-of-3 uses the **same resolved expression** for all candidates. Speaker similarity still measures identity fidelity only — not contextual appropriateness.

## Benchmarking

```python
service.benchmark(identity_id, contexts=["default", "car", "presentation"])
```

Records context, base expression, and resolved expression per row.

## Identity isolation

Context resolution never modifies:

- `VoiceIdentity` metadata
- Reference audio files
- Cached speaker embeddings
- Identity version

## GUI

The desktop app adds a **Context** preset selector and a resolution preview line showing base expression, context, and resolved delivery.

## Limitations (Phase 3)

- Context must be **explicitly supplied** — no automatic sensor/OS inference
- No persistent memory or learned preferences (Phase 4)
- Policy mappings are configurable heuristics, not validated situational models
- `pause_density` remains stored but not fully applied to audio
- Contextual appropriateness requires human evaluation

## Evaluation

```bash
python scripts/evaluate_context.py --identity-id voice_<uuid>
python scripts/smoke_context.py
```

## Future (Phase 4+)

Phase 4 will add **Voice Memory** — persistent adaptation layered on top of context, expression, and identity. Phase 3 intentionally keeps Context, Expression, and Identity as separate concepts.
