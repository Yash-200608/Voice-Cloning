# Voice Memory (Phase 4)

Phase 4 adds **voice memory**: persistent, identity-scoped knowledge about vocal interaction.

## Conceptual separation

| Layer | Question | Example |
|-------|----------|---------|
| **Voice Identity** | WHO is speaking? | Yash |
| **Expression** | HOW is the identity speaking right now? | calm |
| **Context** | WHAT situation exists now? | driving |
| **Memory** | WHAT has been learned/remembered about vocal interaction? | “Minitorch” → “mini-torch”; preferred speaking_rate=0.65 |

Memory is **not** general conversation memory, chatbot vector memory, or autonomous profiling.

## Categories

- `pronunciation` / `name` / `vocabulary` — preferred spoken form for a term/phrase
- `style_preference` — persistent preferred expression dimension (e.g. `speaking_rate`)
- `delivery_preference` — descriptive delivery preference, optionally with a dimension

## Persistence

```text
~/.voiceclone/voices/<voice_id>/memory/memories.json
```

Writes are atomic: temp file → validation → `os.replace`. Corrupt stores raise errors rather than being silently ignored.

## Explicit vs inferred

Sources:

- `explicit_user` — user-provided / confirmed (typical for Phase 4 persistence)
- `system_observation` — observation records (not auto-promoted to permanent preference)
- `imported` — validated import

Phase 4 does **not** implement autonomous learning.

## Resolution & precedence

Pipeline:

```text
raw text
  → text normalization
  → pronunciation memory (render-time text transform)
  → style memory defaults (only when expression omitted)
  → context resolution
  → final expression
  → renderer
```

Precedence:

1. Explicit current expression request
2. Context-derived adjustment
3. Persistent voice preference memory
4. Built-in defaults

Pronunciation memory always applies when enabled/matched because it transforms text, not expression intent.

## Identity isolation

Memories are per `VoiceIdentity`. Identity A’s pronunciation for a term never applies to Identity B.

## Enable / disable / delete

- Disabled memory exists but does not affect synthesis
- Deleted memory is removed and does not affect synthesis

## Privacy

Store only scoped vocal-interaction preferences. Do not auto-store full conversations or arbitrary personal data. Memory is inspectable and deletable.
