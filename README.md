# Voice Clone — Persistent Voice Identity Engine

Voice Clone AI is an offline Python desktop application for cloning voices from short audio samples and generating text-to-speech. **Phase 1** introduces **persistent Voice Identities** — stable, ID-based voice profiles that survive application restarts.

## Voice Identity Concept

A voice is no longer just a WAV file. Each voice is a **VoiceIdentity** with:

- Stable ID (`voice_<uuid4>`) — survives renames
- Separate **raw** and **processed** reference audio
- Cached **Resemblyzer** speaker embedding
- Versioned **metadata** (`schema_version: 1`)
- Chatterbox renderer provenance

```
VoiceIdentity + text → VoiceRenderer → Chatterbox → audio
```

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
- Select a voice, enter text, optionally enable **Best-of-3**
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

# Synthesize
output = service.synthesize(identity.id, "Hello, this is my voice.")
score = service.compare(identity.id, output)

# Best-of-3
best_path, best_score = service.synthesize_best_of(identity.id, "Higher quality.", n=3)

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
  service.py               → VoiceIdentityService
  exceptions.py            → Domain errors
voiceclone/identity/
  repository.py            → Persistence
  metadata.py              → JSON schema
  embeddings.py            → Cached Resemblyzer embeddings
voiceclone/inference/
  renderer.py              → ChatterboxRenderer
voiceclone/compatibility/
  legacy.py                → Flat WAV migration
voiceclone/benchmarking.py → Batch eval (similarity + optional WER)
voiceclone/cloner.py       → Chatterbox TTS (unchanged core)
```

## Tests

```bash
pytest -v
```

Run the Phase 1 completion gate smoke test:

```bash
python scripts/smoke_identity.py
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
