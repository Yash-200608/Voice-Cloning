# Real-Time Voice (Phase 5)

Phase 5 adds a **real-time runtime** around the existing Voice Identity stack so speech can begin playing before the full utterance finishes generating.

## Honest capability label

| Mode | Meaning |
|------|---------|
| **Native model streaming** | The TTS model emits tokens/waveform progressively |
| **Chunked generation** | Application splits text into speech units and synthesizes them incrementally (**this build**) |
| **Buffered simulation** | Full render completes, then playback is faked as progressive |

Chatterbox in this repository exposes `synthesize(text) -> WAV` only. Phase 5 therefore implements **chunked generation + buffered playback**, labeled `STREAMING_MODE = "chunked_generation"`.

## Conceptual separation

| Layer | Question |
|-------|----------|
| Voice Identity | WHO is speaking? |
| Voice Memory | WHAT has been remembered? |
| Context | WHAT situation is this? |
| Expression | HOW should it sound? |
| Render Plan | Final rendering intent (resolved once per session) |
| **Real-Time Runtime** | WHEN/HOW audio is delivered |
| Renderer | HOW the model produces audio |
| AudioSink | WHERE audio is played |

## Architecture

```text
VoiceIdentityService
        │
        ▼
RealTimeVoiceService
        │
        ├─ resolve_render_plan (once per session)
        ├─ split_speech_units (sentence/clause)
        ├─ InferenceScheduler (serialized model access)
        ├─ AudioChunkQueue (bounded, backpressure)
        └─ AudioSink (Null / Collecting / SoundDevice)
```

## Session lifecycle

```text
CREATED → PREPARING → GENERATING → PLAYING → COMPLETED
                         ↘ CANCELLED / FAILED
```

## API

```python
session = service.synthesize_realtime(
    identity.id,
    "Hello. This plays early.",
    expression="calm",
    context="desktop",
    use_memory=True,
    play_audio=True,
    save_final=True,
)
done = service.wait_realtime(session.session_id)
service.cancel_realtime(session.session_id)      # or interrupt_realtime
service.warm_up_realtime()                       # optional model load
```

Standard `synthesize(...)` is unchanged.

## Metrics

- `ttfa_ms` — time to first audio ready
- `ttfp_ms` — time to first playback start
- `total_generation_ms`, `rtf`, `chunk_count`, `underrun_count`, `cancel_latency_ms`

Configurable soft target: `target_first_audio_ms` (default 1500). Not a hard SLA.

## Cancellation

Cancel/interrupt stops further generation at the **next speech-unit boundary**, flushes the playback queue, stops the sink, and marks the session `CANCELLED`. Mid-unit Chatterbox calls cannot be preempted.

## Concurrency

Multiple session objects may exist, but model inference is serialized via `InferenceScheduler`. Do not assume Chatterbox is thread-safe.

## UI

Dashboard **Real-Time Mode** starts chunked playback and shows a status indicator. **Stop / Interrupt** cancels the active session. Best-of-3 is disabled in real-time mode.

## Limits (Phase 5)

- No native Chatterbox streaming
- No microphone barge-in detection (API is interrupt-ready only)
- No multi-device / network sinks
- No speculative / predictive generation
- No Jarvis wiring

## Evaluation

```bash
python scripts/evaluate_realtime.py --json-out /tmp/rt.json
python scripts/smoke_realtime.py
python scripts/smoke_realtime.py --real-models --play-audio   # when hardware allows
```
