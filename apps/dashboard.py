import logging
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog, messagebox
import threading
from pathlib import Path

from voiceclone.core.service import VoiceIdentityService
from voiceclone.core.expression import ExpressionProfile, merge_expression
from voiceclone.core.exceptions import VoiceCloneError, VoiceIdentityNotFound
from voiceclone.evaluation.render_metadata import metadata_path_for_audio
from voiceclone.performance import start_timer, stop_timer, cpu_usage

logger = logging.getLogger(__name__)

service = VoiceIdentityService()
_voice_map: dict[str, str] = {}
_selected_id: str | None = None

PRESET_OPTIONS = ["neutral"] + [p for p in service.list_expression_presets() if p != "neutral"]
PRESET_OPTIONS.append("custom")


def _ui(fn, *args, **kwargs):
    app.after(0, lambda: fn(*args, **kwargs))


def _format_identity_info(identity) -> str:
    embedding_status = "cached" if identity.embedding_path else "missing"
    return (
        f"Name: {identity.name}\n"
        f"ID: {identity.id}\n"
        f"Created: {identity.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Duration: {identity.duration_seconds:.1f}s\n"
        f"Embedding: {embedding_status}\n"
        f"Renderer: {identity.renderer_model}"
    )


def _update_identity_panel(identity_id: str | None):
    global _selected_id
    _selected_id = identity_id
    if not identity_id:
        identity_info.config(text="No voice selected.")
        return
    try:
        identity = service.get_identity(identity_id)
        identity_info.config(text=_format_identity_info(identity))
    except VoiceIdentityNotFound:
        identity_info.config(text="Voice not found. Refresh the list.")


def _toggle_custom_controls(*_args):
    state = tk.NORMAL if expression_picker.get() == "custom" else tk.DISABLED
    for scale in custom_scales.values():
        scale.config(state=state)


def _build_expression():
    preset = expression_picker.get()
    if preset != "custom":
        return preset
    base = service.get_expression_preset("neutral")
    overrides = {name: var.get() for name, var in custom_vars.items()}
    return merge_expression(base, {**overrides, "name": "custom"})


def refresh_voices(select_name: str | None = None):
    global _voice_map
    identities = service.list_identities()
    _voice_map = {i.name: i.id for i in identities}
    names = sorted(_voice_map.keys(), key=str.lower)
    voice_picker["values"] = names

    if select_name and select_name in _voice_map:
        voice_picker.set(select_name)
        _update_identity_panel(_voice_map[select_name])
    elif names:
        current = voice_picker.get()
        if current in _voice_map:
            _update_identity_panel(_voice_map[current])
        else:
            voice_picker.set(names[0])
            _update_identity_panel(_voice_map[names[0]])
    else:
        voice_picker.set("")
        _update_identity_panel(None)


def on_voice_selected(_event=None):
    name = voice_picker.get()
    if name in _voice_map:
        _update_identity_panel(_voice_map[name])


def on_record():
    name = simpledialog.askstring("New voice", "Voice name:", parent=app)
    if not name:
        return

    record_btn.config(state=tk.DISABLED)
    _ui(result_label.config, text=f"Recording 12s for '{name}'...")

    def _run():
        try:
            service.create_from_recording(name)
            _ui(result_label.config, text=f"Saved voice identity '{name}'.")
            _ui(refresh_voices, name)
        except VoiceCloneError as e:
            _ui(result_label.config, text=e.user_message)
        except Exception as e:
            logger.exception("Recording failed")
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
    if not name or name not in _voice_map:
        _ui(result_label.config, text="No voice selected. Record or import one first.")
        return

    identity_id = _voice_map[name]
    expression = _build_expression()
    generate_btn.config(state=tk.DISABLED)
    _ui(result_label.config, text="Generating speech...")

    def _run():
        try:
            start = start_timer()
            if best_of_var.get():
                output, score = service.synthesize_best_of(
                    identity_id, text, n=3, expression=expression,
                )
            else:
                output = service.synthesize(identity_id, text, expression=expression)
                score = service.compare(identity_id, output)
            elapsed = stop_timer(start)
            expr_label = expression if isinstance(expression, str) else expression.versioned_name
            meta = metadata_path_for_audio(output)
            meta_note = "metadata saved" if meta.exists() else "no metadata"
            _ui(
                result_label.config,
                text=(
                    f"Done — {output}\n"
                    f"Expression: {expr_label} | {meta_note}\n"
                    f"Time: {elapsed}s | Similarity: {score:.3f} | CPU: {cpu_usage()}%"
                ),
            )
        except VoiceCloneError as e:
            _ui(result_label.config, text=e.user_message)
        except Exception as e:
            logger.exception("Generation failed")
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
            identity = service.create_from_file(name, source)
            _ui(result_label.config, text=f"Imported voice identity '{identity.name}'.")
            _ui(refresh_voices, name)
        except VoiceCloneError as e:
            _ui(result_label.config, text=e.user_message)
        except Exception as e:
            logger.exception("Import failed")
            _ui(result_label.config, text=f"Import error: {e}")
        finally:
            _ui(import_btn.config, state=tk.NORMAL)

    threading.Thread(target=_run, daemon=True).start()


def on_delete():
    name = voice_picker.get()
    if not name or name not in _voice_map:
        _ui(result_label.config, text="No voice selected to delete.")
        return

    if not messagebox.askyesno("Delete voice", f"Delete voice identity '{name}'?", parent=app):
        return

    identity_id = _voice_map[name]
    delete_btn.config(state=tk.DISABLED)

    def _run():
        try:
            service.delete_identity(identity_id)
            _ui(result_label.config, text=f"Deleted voice '{name}'.")
            _ui(refresh_voices)
        except VoiceCloneError as e:
            _ui(result_label.config, text=e.user_message)
        except Exception as e:
            logger.exception("Delete failed")
            _ui(result_label.config, text=f"Delete error: {e}")
        finally:
            _ui(delete_btn.config, state=tk.NORMAL)

    threading.Thread(target=_run, daemon=True).start()


app = tk.Tk()
app.title("Voice Clone AI")
app.geometry("520x680")

frame = tk.Frame(app, padx=12, pady=12)
frame.pack(fill=tk.BOTH, expand=True)

tk.Label(frame, text="Voice:").pack(anchor="w")
voice_picker = ttk.Combobox(frame, state="readonly", width=40)
voice_picker.pack(fill=tk.X, pady=4)
voice_picker.bind("<<ComboboxSelected>>", on_voice_selected)

btn_row = tk.Frame(frame)
btn_row.pack(fill=tk.X, pady=4)
record_btn = tk.Button(btn_row, text="Record New Voice (12s)", command=on_record)
record_btn.pack(side=tk.LEFT, padx=(0, 4))
import_btn = tk.Button(btn_row, text="Import Reference File", command=on_import_file)
import_btn.pack(side=tk.LEFT, padx=4)
delete_btn = tk.Button(btn_row, text="Delete Voice", command=on_delete)
delete_btn.pack(side=tk.LEFT, padx=4)

tk.Label(frame, text="Identity Info:").pack(anchor="w", pady=(8, 0))
identity_info = tk.Label(frame, text="", justify=tk.LEFT, anchor="w", wraplength=480)
identity_info.pack(fill=tk.X, pady=2)

tk.Label(frame, text="Expression:").pack(anchor="w", pady=(8, 0))
expression_picker = ttk.Combobox(frame, state="readonly", width=40, values=PRESET_OPTIONS)
expression_picker.pack(fill=tk.X, pady=4)
expression_picker.set("neutral")
expression_picker.bind("<<ComboboxSelected>>", _toggle_custom_controls)

custom_frame = tk.LabelFrame(frame, text="Custom Expression", padx=8, pady=6)
custom_frame.pack(fill=tk.X, pady=4)
custom_vars: dict[str, tk.DoubleVar] = {}
custom_scales: dict[str, tk.Scale] = {}
for dim in ("energy", "warmth", "seriousness", "confidence", "urgency", "speaking_rate"):
    row = tk.Frame(custom_frame)
    row.pack(fill=tk.X, pady=2)
    tk.Label(row, text=dim.replace("_", " ").title(), width=14, anchor="w").pack(side=tk.LEFT)
    var = tk.DoubleVar(value=0.5)
    custom_vars[dim] = var
    scale = tk.Scale(row, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=var, length=280)
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    scale.config(state=tk.DISABLED)
    custom_scales[dim] = scale

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
