# Voice Clone — Persistent Voice Identity Engine

Voice Clone AI is an offline Python desktop application for cloning voices from short audio samples and generating text-to-speech.

- **Phase 1 — Voice Identity:** persistent `VoiceIdentity` objects with stable IDs
- **Phase 2 — Expressive Voice:** control *how* an identity speaks via structured expression profiles
- **Phase 3 — Context-Aware Voice:** adapt delivery to *situation* via structured context profiles

## Voice Identity (Phase 1)

A voice is no longer just a WAV file. Each voice is a **VoiceIdentity** with:

- Stable ID (`voice_<uuid4>`) — survives renames
- Separate **raw** and **processed** reference audio
- Cached **Resemblyzer** speaker embedding
- Versioned **metadata** (`schema_version: 1`)
- Chatterbox renderer provenance

## Expressive Voice (Phase 2)

**Voice Identity = who is speaking.** **Expression = how the identity is speaking.**

The same identity can deliver the same text in different expressive styles (calm, warm, professional, excited, etc.) without creating new identities.

```
VoiceIdentity + text + ExpressionProfile → VoiceRenderer → Chatterbox → audio
```

Built-in presets: `neutral`, `calm`, `warm`, `friendly`, `professional`, `serious`, `excited`, `concerned`, `urgent`.

See [docs/architecture/expressive-voice.md](docs/architecture/expressive-voice.md) for full architecture details.

## Context-Aware Voice (Phase 3)

**Voice Identity = who is speaking.** **Expression = how the identity is speaking.** **Context = the situation in which the identity is speaking.**

The same identity can adapt delivery for desktop, car, noisy environments, presentations, and warnings — without creating new identities.

```
VoiceIdentity + text + Expression + Context → ContextResolver → VoiceRenderer → audio
```

Built-in context presets: `default`, `desktop`, `phone`, `car`, `noisy_environment`, `quiet_environment`, `presentation`, `notification`, `warning`.

See [docs/architecture/context-aware-voice.md](docs/architecture/context-aware-voice.md) for resolution policy and metadata details.

## Storage Layout

```text
~/.voiceclone/                    (or $VOICECLONE_HOME)
├── voices/
│   ├── voice_<uuid>/
│   │   ├── metadata.json
│   │   ├── raw/reference.wav
│   │   ├── processed/reference.wav
│   │   └── embeddings/speaker.npy
│   └── LegacyName.wav            (legacy flat files, preserved after migration)
├── outputs/
│   └── voice_<uuid>/             (generated audio per identity)
└── cache/
```

## Installation

```bash
git clone https://github.com/Yash-200608/Voice-Cloning.git
cd Voice-Cloning
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install -e ".[dev]"
```

**Requirements:** Python 3.10–3.12, ffmpeg (for video import), microphone/PortAudio (for recording). First Chatterbox run downloads model weights from Hugging Face.

## Usage

### Desktop GUI

```bash
python main.py
```

- **Record** or **Import** to create a voice identity
- Select a voice, choose an **Expression** preset (or custom sliders)
- Choose a **Context** preset (default, desktop, car, noisy, etc.)
- Enter text, optionally enable **Best-of-3**
- View identity info (name, ID, duration, embedding status, renderer)
- **Delete** removes the identity directory safely

### Python API

```python
from voiceclone import VoiceIdentityService

service = VoiceIdentityService()

# Create from file
identity = service.create_from_file("Yash", "/path/to/reference.wav")

# List / get
for i in service.list_identities():
    print(i.id, i.name)

# Synthesize (neutral default)
output = service.synthesize(identity.id, "Hello, this is my voice.")

# Synthesize with expression preset
output = service.synthesize(identity.id, "Hello.", expression="calm")

# Custom expression profile
from voiceclone import ExpressionProfile
profile = ExpressionProfile(name="custom", energy=0.8, warmth=0.7)
output = service.synthesize(identity.id, "Hello.", expression=profile)

# Best-of-3 with expression
best_path, best_score = service.synthesize_best_of(
    identity.id, "Higher quality.", n=3, expression="excited",
)

# Context-aware synthesis
output = service.synthesize(identity.id, "Please be careful.", context="car")

# Expression + context
output = service.synthesize(
    identity.id, "Good morning.", expression="warm", context="desktop",
)

# Inspect resolution
plan = service.resolve_render_plan(expression="calm", context="car")
print(plan.summary())

# Rename (ID unchanged)
service.rename_identity(identity.id, "My Voice")

# Delete
service.delete_identity(identity.id)
```

### Legacy Compatibility

Flat `.wav` files in `voices/` are **lazily migrated** on first `list_identities()` call. The original `.wav` is never deleted or modified during migration.

Path-based APIs still work:

```python
from voiceclone import list_voices, voice_path, clone, compare, benchmark

names = list_voices()
ref = voice_path(names[0])
out = clone("Hello.", ref)
score = compare(ref, out)
```

## Architecture

```text
apps/dashboard.py          → VoiceIdentityService (GUI)
voiceclone/core/
  models.py                → VoiceIdentity
  expression.py            → ExpressionProfile, presets
  context.py               → ContextProfile, presets
  context_resolver.py      → Context → Expression policy
  service.py               → VoiceIdentityService
  exceptions.py            → Domain errors
voiceclone/identity/
  repository.py            → Persistence
  metadata.py              → JSON schema
  embeddings.py            → Cached Resemblyzer embeddings
voiceclone/inference/
  renderer.py              → ChatterboxRenderer
  chatterbox_mapping.py    → Expression → Chatterbox settings
voiceclone/evaluation/
  render_metadata.py       → Generation event metadata
voiceclone/compatibility/
  legacy.py                → Flat WAV migration
voiceclone/benchmarking.py → Batch eval (similarity + optional WER)
voiceclone/cloner.py       → Chatterbox TTS (unchanged core)
```

## Tests

```bash
pytest -v
```

Run the Phase 1 completion gate:

```bash
python scripts/smoke_identity.py
```

Run the Phase 2 expressive voice gate:

```bash
python scripts/smoke_expressive.py
```

Run expressive evaluation (requires identity ID):

```bash
python scripts/evaluate_expressions.py --identity-id voice_<uuid>
```

Run the Phase 3 context-aware voice gate:

```bash
python scripts/smoke_context.py
```

Run context evaluation (requires identity ID):

```bash
python scripts/evaluate_context.py --identity-id voice_<uuid>
```

## Migration Behavior

| Legacy | Phase 1 |
|--------|---------|
| `voices/Yash.wav` | `voices/voice_<uuid>/` with metadata |
| Name = filename | Name in metadata; ID stable |
| In-place preprocessing | Raw + processed separate |
| Re-embed every compare | Cached `speaker.npy` |

## License

See repository for license details.
