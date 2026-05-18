# Voice Clone

Voice Clone AI is a powerful offline desktop application built in Python that allows you to clone voices from short audio samples and generate realistic text-to-speech. Whether you want to record your own voice directly through the app or import audio/video files to use as references, Voice Clone AI handles it seamlessly.

## 🚀 Features

- **Direct Voice Recording**: Quickly record a 12-second voice sample straight from your microphone to use as a cloning reference.
- **Media Import**: Import reference voices from various audio and video formats (WAV, MP3, M4A, MP4, MKV, etc.). The app automatically processes and extracts the audio.
- **Text-to-Speech Generation**: Type any text and generate speech using your selected cloned voice.
- **"Best-of-3" Quality Option**: A toggle that generates three versions of the audio and automatically selects the highest quality one using similarity scoring.
- **Performance Metrics**: View the generation time, CPU usage, and voice similarity score after each generation.
- **Simple Desktop UI**: An intuitive graphical interface built with Tkinter, making it easy to manage your voice profiles.

## 📁 Project Structure

```
VoiceCloning/
├── main.py                 # The main entry point for the application.
├── apps/
│   └── dashboard.py        # The Tkinter GUI dashboard and UI logic.
├── voiceclone/             # The core logic package
│   ├── __init__.py
│   ├── audio_utils.py      # Utilities for preprocessing and normalizing audio.
│   ├── benchmark.py        # Performance benchmarking.
│   ├── cloner.py           # Core text-to-speech cloning logic.
│   ├── recorder.py         # Handles microphone recording and media imports.
│   ├── similarity.py       # Compares cloned audio to the original reference.
│   ├── text_utils.py       # Text normalization utilities.
│   └── voices.py           # Manages saved voice profiles.
└── .gitignore              # Configured to ignore Python caches, envs, and itself.
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yash-200608/Voice-Cloning.git
   cd Voice-Cloning
   ```

2. **Set up a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   Make sure you have all required dependencies installed for audio processing and TTS generation. 
   *(Note: Since no `requirements.txt` is currently provided, ensure you have the necessary backend TTS libraries and PyAudio/SoundFile depending on the `voiceclone` package requirements).*

## 💻 Usage

To launch the Voice Clone AI dashboard, run the main script from the root directory:

```bash
python main.py
```

### Navigating the App:
1. **Add a Voice**: Click **Record New Voice** to use your microphone, or **Import Reference File** to load an existing recording.
2. **Select a Voice**: Choose your newly added voice from the dropdown menu.
3. **Generate Speech**: Enter the text you want the cloned voice to say.
4. **Best-of-3**: Check the "Best-of-3" box if you want the highest possible quality (this takes slightly longer).
5. **Click "Generate Speech"**: The result will be saved, and performance metrics will be displayed at the bottom.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
