import tkinter as tk
from tkinter import ttk, simpledialog, filedialog
import threading
from pathlib import Path

from voiceclone import (
    record,
    import_reference,
    clone,
    clone_best_of,
    compare,
    list_voices,
    voice_path,
)
from voiceclone.performance import start_timer, stop_timer, cpu_usage


def _ui(fn, *args, **kwargs):
    app.after(0, lambda: fn(*args, **kwargs))


def refresh_voices():
    voices = list_voices()
    voice_picker["values"] = voices
    if voices and not voice_picker.get():
        voice_picker.set(voices[0])


def on_record():
    name = simpledialog.askstring("New voice", "Voice name:", parent=app)
    if not name:
        return

    record_btn.config(state=tk.DISABLED)
    _ui(result_label.config, text=f"Recording 12s for '{name}'...")

    def _run():
        try:
            record(name)
            _ui(result_label.config, text=f"Saved voice '{name}'.")
            _ui(refresh_voices)
            _ui(voice_picker.set, name)
        except Exception as e:
            _ui(result_label.config, text=f"Recording error: {e}")
        finally:
            _ui(record_btn.config, state=tk.NORMAL)

    threading.Thread(target=_run, daemon=True).start()


def on_generate():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        _ui(result_label.config, text="Please enter text.")
        return

    name = voice_picker.get()
    if not name:
        _ui(result_label.config, text="No voice selected. Record one first.")
        return

    generate_btn.config(state=tk.DISABLED)
    _ui(result_label.config, text="Generating speech...")

    def _run():
        try:
            ref = voice_path(name)
            start = start_timer()
            if best_of_var.get():
                output, score = clone_best_of(text, ref, n=3)
            else:
                output = clone(text, ref)
                score = compare(ref, output)
            elapsed = stop_timer(start)
            _ui(
                result_label.config,
                text=f"Done — {output}\nTime: {elapsed}s | Similarity: {score:.3f} | CPU: {cpu_usage()}%",
            )
        except Exception as e:
            _ui(result_label.config, text=f"Error: {e}")
        finally:
            _ui(generate_btn.config, state=tk.NORMAL)

    threading.Thread(target=_run, daemon=True).start()


def on_import_file():
    source = filedialog.askopenfilename(
        title="Choose reference audio/video",
        filetypes=[
            ("Media files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma *.mp4 *.mkv *.mov *.webm *.avi"),
            ("All files", "*.*"),
        ],
    )
    if not source:
        return

    default_name = Path(source).stem
    name = simpledialog.askstring("Import voice", "Voice name:", initialvalue=default_name, parent=app)
    if not name:
        return

    import_btn.config(state=tk.DISABLED)
    _ui(result_label.config, text=f"Importing reference for '{name}'...")

    def _run():
        try:
            out = import_reference(name, source, preprocess=True)
            _ui(result_label.config, text=f"Imported voice '{name}': {out}")
            _ui(refresh_voices)
            _ui(voice_picker.set, name)
        except Exception as e:
            _ui(result_label.config, text=f"Import error: {e}")
        finally:
            _ui(import_btn.config, state=tk.NORMAL)

    threading.Thread(target=_run, daemon=True).start()


app = tk.Tk()
app.title("Voice Clone AI")
app.geometry("520x380")

frame = tk.Frame(app, padx=12, pady=12)
frame.pack(fill=tk.BOTH, expand=True)

tk.Label(frame, text="Voice:").pack(anchor="w")
voice_picker = ttk.Combobox(frame, state="readonly", width=40)
voice_picker.pack(fill=tk.X, pady=4)

record_btn = tk.Button(frame, text="Record New Voice (12s)", command=on_record)
record_btn.pack(pady=6)

import_btn = tk.Button(frame, text="Import Reference File", command=on_import_file)
import_btn.pack(pady=2)

tk.Label(frame, text="Text to speak:").pack(anchor="w", pady=(8, 0))
text_input = tk.Text(frame, height=4, width=50, wrap=tk.WORD)
text_input.pack(fill=tk.X, pady=4)

best_of_var = tk.BooleanVar(value=False)
tk.Checkbutton(frame, text="Best-of-3 (slower, higher quality)", variable=best_of_var).pack(anchor="w")

generate_btn = tk.Button(frame, text="Generate Speech", command=on_generate)
generate_btn.pack(pady=8)

result_label = tk.Label(frame, text="", wraplength=480, justify="left")
result_label.pack(pady=8, fill=tk.X)

refresh_voices()


if __name__ == "__main__":
    app.mainloop()