"""Textual TUI (Phase 4, S13): one turn engine, screen frontend.

Public names: run_tui (entry from `ana tui` / `ana --tui`), AnaApp.
The app drives the chat through ana.cli._do_turn — provider, tools,
image queue, vision_model pick and voice-free streaming are THE SAME
behavior as the CLI (no forked engine). Textual is an optional
dependency: importing without the `tui` extra degrades to a plain
object backend with an actionable run_tui error.
"""

from __future__ import annotations

import os
import sys
import threading
import time

ANA_TUI_WELCOME = (
    "¡Hola! Soy Ana. Escribime, o con /voice on presioná Enter con el "
    "campo vacío para hablar. Comandos: /help · /model · /provider · "
    "/img · /voice · /tools · /new. Herramientas disponibles incluyen "
    "current_time (hora y fecha), system_info y web_search.")

ANA_TUI_HELP = (
    "Comandos:\n"
    "/model [name]      model override (auto-picked for images)\n"
    "/provider [name]   switch provider from the config\n"
    "/tools [on|off|list]   tool calling\n"
    "/voice [on|off|status]   voice: empty-Enter = push-to-talk,\n"
    "                   replies spoken (voxtype STT, Kokoro TTS)\n"
    "/img <ruta>        attach an image to the next message\n"
    "/new               clear this conversation\n"
    "/exit (Ctrl+Q)     leave\n"
    "Tools of note: current_time (hora/fecha), system_info, "
    "web_search, hypr_*.")

ANA_TUI_MISSING_MSG = (
    "textual not installed — run: .venv/bin/pip install -e '.[tui]'")

ANA_TUI_DEBUG = os.environ.get("ANA_TUI_DEBUG", "").strip()


def _dbg(fmt: str, *args) -> None:
    """Opt-in trace to $ANA_TUI_DEBUG (a file path) for TUI debugging."""
    if not ANA_TUI_DEBUG:
        return
    try:
        with open(ANA_TUI_DEBUG, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} "
                    f"t{threading.get_ident():<10} "
                    + (fmt % args if args else fmt) + "\n")
    except OSError:
        pass

try:  # the `tui` extra; AnaApp stays importable without it
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll
    from textual.widgets import Input, Static

    _TEXTUAL_INSTALLED = True
except ImportError:  # tui extra optional — plain chat still works
    App = object  # type: ignore[assignment,misc]
    ComposeResult = None
    _TEXTUAL_INSTALLED = False


class _PressGate:
    """Push-to-talk click seam: callable, driven by Ui input events.

    record_to_wav asks press() (no timeout, start) and then
    press(timeout=limit) (stop): each blocks until the UI side puts a
    token, or returns False at the timeout (auto-stop). A raw Queue is
    not callable and record_to_wav calls press() directly.
    """

    def __init__(self) -> None:
        import queue as _queue

        self._q: "_queue.Queue[str]" = _queue.Queue()

    def put(self, kind: str) -> None:
        self._q.put(kind)

    def __call__(self, timeout=None) -> bool:
        try:
            self._q.get(timeout=timeout)
            return True
        except Exception:  # queue.Empty (or closed app): auto-stop
            return False


def import_textual():
    """Fail closed when the tui extra is not installed."""
    if not _TEXTUAL_INSTALLED:
        print("error: " + ANA_TUI_MISSING_MSG, file=sys.stderr)
        raise SystemExit(1)


class ChatInput(Input if _TEXTUAL_INSTALLED else object):

    _ana_hist: list = []     # shared by the app (same list object)
    _hist_pos: int | None = None

    def __init__(self, **kwargs):
        history = kwargs.pop("history", None)  # test convenience only
        super().__init__(**kwargs)
        self._ana_hist = history or []
        self._hist_pos = len(self._ana_hist)
    """Input with an entry history: ArrowUp recalls, ArrowDown returns.

    Textual's Input does not keep a history; commands and chat turns land
    in the same list so recall repeats both (/model, requests...).
    """

    if _TEXTUAL_INSTALLED:
        BINDINGS = [
            ("up", "history_prev", "prev"),
            ("down", "history_next", "next"),
        ]

        def action_history_prev(self) -> None:
            self._hist_step(-1)

        def action_history_next(self) -> None:
            self._hist_step(+1)

        def _hist_step(self, step: int) -> None:
            history = self._ana_hist or []
            if not history:
                return
            pos = (self._hist_pos if self._hist_pos is not None
                   else len(history)) + step
            pos = max(0, min(pos, len(history)))
            self._hist_pos = pos
            self.value = "" if pos == len(history) else history[pos]


def run_tui(cfg_path, model_override=None, provider_override=None,
            tools=True, images=None, voice=True,
            stt: str | None = None, tts: str | None = None):
    """Entry point for `ana tui` / `ana --tui`."""
    import_textual()
    AnaApp(cfg_path, model_override=model_override,
           provider_override=provider_override, tools=tools,
           voice=voice, images=images, stt=stt, tts=tts).run()
    # Option B (user choice): closing the chat closes the overlay;
    # ana-quick reopens a fresh one on the next session.
    from .waifu import waifu_send
    waifu_send({"type": "quit"})
    from . import memory
    memory.shutdown()  # best-effort: end Ana's Engram session at exit


class AnaApp(App if _TEXTUAL_INSTALLED else object):
    """Chat screen: status line, scrolling transcript, one input."""

    if _TEXTUAL_INSTALLED:
        CSS = """
        #status { height: 1; background: $boost; color: $text; padding: 0 1; }
        #chat { border: none; }
        #chat > Static { padding: 0 1; }
        #chat > .you { color: $accent; }
        #chat > .reply { color: $text; }
        #chat > .note { color: $text-muted; }
        #chat > .error { color: $error; }
        Input { border: none; background: $surface; }
        """

    def __init__(self, cfg_path, model_override=None,
                 provider_override=None, tools=True, registry=None,
                 client_fx=None, stream=None, images=None,
                 max_rounds: int = 4, voice: bool = False,
                 stt: str | None = None, tts: str | None = None):
        super().__init__()
        self.cfg_path = cfg_path
        self.model_override = model_override
        self.provider_override = provider_override
        self.tools_on = tools
        self.registry = registry
        self.client_fx = client_fx
        self.stream = stream
        self.max_rounds = max_rounds
        self.history: list[dict] = []
        self.pending_images: list[str] = list(images or [])
        self.transcript = ""

        self.voice_fx = None   # injected seam (tests) or the live default
        self.voice_on = voice
        self.stt = stt
        self.tts = tts
        self._press_gate = None      # push-to-talk gate (None = not recording)
        self._entry_history: list[str] = []
        self.busy = True  # until the first mount finishes loading

    # ---- widgets ------------------------------------------------------

    def compose(self):
        yield Static(id="status")
        yield VerticalScroll(id="chat")
        yield ChatInput(id="entry")

    def on_mount(self):
        try:
            from .config import load_config

            self.cfg = load_config(self.cfg_path)
            from .tools import build_default_tools

            self.registry = self.registry or build_default_tools()
        except Exception as exc:  # ConfigError or anything environmental
            self.cfg = None
            self._note(f"error: {exc}")
        self.busy = False
        self._refresh_status()
        entry = self.query_one("#entry")
        entry._ana_hist = self._entry_history  # live: the same list object
        entry._hist_pos = len(self._entry_history)
        if not self.history and getattr(self.cfg, "providers", None):
            self._note(ANA_TUI_WELCOME)
        if entry:
            entry.focus()

    def _set_status(self, text: str) -> None:
        if self.is_running and _TEXTUAL_INSTALLED:
            self.query_one("#status").update(text)

    def _refresh_status(self, working: str = "") -> None:
        if not hasattr(self, "cfg") or self.cfg is None:
            self._set_status("ana: config not loaded")
            return
        from .cli import _default_model

        provider = self.provider_override or self.cfg.default_provider
        model = self.model_override or _default_model(self.cfg, provider,
                                                      None)
        tail = f" · {working}" if working else ""
        _dbg("refresh_status working=%r", working)

        self._set_status(
            f"model: {model} · provider: {provider} · "
            f"tools: {'on' if self.tools_on else 'off'} · "
            f"imgs: {len(self.pending_images)} · "
            f"voice: {'on' if self.voice_on else 'off'}" + tail)

    # ---- transcript ---------------------------------------------------

    def _note(self, text: str) -> None:
        self.transcript += ("" if not self.transcript else "\n") + text
        self._mount_item(text)

    def _append(self, chunk: str) -> None:
        """Streaming chunk: append to the CURRENT reply item.

        One Static per turn updates in place — RichLog.write() rendered
        each chunk as its own line in Textual 8, splitting the reply.
        """
        self.transcript += chunk
        self._stream_buf = getattr(self, "_stream_buf", "") + chunk
        self._render_reply()

    def _render_reply(self) -> None:
        if not (self.is_running and _TEXTUAL_INSTALLED):
            return
        chat = self.query_one("#chat")
        widget = getattr(self, "_stream_widget", None)
        buf = getattr(self, "_stream_buf", "")
        if widget is None:
            self._stream_widget = Static(buf, classes="reply")
            chat.mount(self._stream_widget)
            widget = self._stream_widget
        else:
            widget.update(buf)
        chat.scroll_end(animate=False)

    def _mount_item(self, text: str, kind: str = "note"):
        """A non-streaming line (user turn, command echo, error)."""
        if not (self.is_running and _TEXTUAL_INSTALLED):
            return None
        chat = self.query_one("#chat")
        widget = Static(text, classes=kind)
        chat.mount(widget)
        self._stream_widget = None  # replies restart after a plain line
        self._stream_buf = ""
        chat.scroll_end(animate=False)
        return widget

    # ---- input --------------------------------------------------------

    def on_input_submitted(self, event) -> None:
        _dbg("submitted: %r", event.value)
        self.submit(event.value)

    # Textual 8 headless routing: the naming-convention handler needs a
    # message namespace that is empty here, so register explicitly via
    # the @on decorator targeting the Input widgets.
    if _TEXTUAL_INSTALLED:
        from textual import on  # noqa: PLC0415 (import inside class scope)


    def submit(self, value: str) -> None:
        text = (value or "").strip()
        self.query_one("#entry").value = ""
        if not text:
            # S9 gesture parity: with voice on, an empty submit toggles the
            # push-to-talk (first press starts, second stops).
            if self.voice_on:
                self._voice_gate_press()
            return
        self._entry_history.append(text)
        entry = self.query_one("#entry")
        entry._hist_pos = len(self._entry_history)  # stay at the free end
        self._mount_item(f"you> {text}", "you")
        self.transcript += ("" if not self.transcript else "\n")             + f"you> {text}"
        if text.startswith("/"):
            self._command(text)
        else:
            self._run_turn(text)

    # ---- voice (S9): same voice seam as the CLI REPL -------------------

    def _voice_gate_press(self) -> None:
        if self._press_gate is not None:
            self._press_gate.put("stop")   # second press: stop recording
            return
        self._press_gate = _PressGate()
        self._press_gate.put("start")      # first press: start instantly
        self.busy = True
        self.run_worker(self._voice_record_worker, exclusive=True,
                        thread=True, name="ana-voice")

    def _voice_record_worker(self) -> None:
        """Thread worker: push-to-talk round, then the shared turn engine."""
        from .voice.errors import VoiceError

        from .cli import _DefaultVoice
        from .voice.stt import default_max_seconds

        gate = self._press_gate
        vfx = (self.voice_fx if self.voice_fx is not None
               else _DefaultVoice(self.stt, self.tts))
        try:
            wav = vfx.record(default_max_seconds(), press=gate)
            heard = (vfx.transcribe(wav) or "").strip()
        except VoiceError as exc:
            self.emit_threaded(f"(voice unavailable: {exc})")
        except Exception as exc:  # transcription/storage hazards
            self.emit_threaded(f"(voice unavailable: {exc})")
        else:
            if not heard:
                self.emit_threaded("(no escuché nada)")
            else:
                self.emit_threaded(f"you> {heard} (voz)")
                self._turn_worker(heard)
        finally:
            self._press_gate = None
            self.call_from_thread(self._finish_turn, "")

    def voice_fx_speak(self, reply: str) -> None:
        """Speak a reply; degrade to a note, never crash the turn."""
        try:
            from .cli import _DefaultVoice

            vfx = (self.voice_fx if self.voice_fx is not None
                   else _DefaultVoice(self.stt, self.tts))
            vfx.speak(reply)
        except Exception as exc:
            self.emit_threaded(f"(voice unavailable: {exc})")

    def _command(self, line: str) -> None:
        from .cli import parse_command

        cmd, arg = parse_command(line)  # ValueError: unknown shape
        if cmd in ("exit", "quit"):
            self.exit()
            return
        if cmd in ("help", "ayuda", "?"):
            self._note(ANA_TUI_HELP)
            return
        if cmd == "new":
            self.history.clear()
            self.pending_images.clear()
            self._note("historial limpiado")
            self._refresh_status()
        elif cmd == "model":
            if arg:
                self.model_override = arg
                self._refresh_status()
            else:
                from .cli import _default_model

                provider = self.provider_override or self.cfg.default_provider
                self._note("model: " + (self.model_override or _default_model(
                    self.cfg, provider, None)))
        elif cmd == "provider":
            if arg:
                self.cfg.provider(arg)  # validates
                self.provider_override = arg
                self.model_override = None
                self._refresh_status()
            else:
                self._note(f"provider: {self.provider_override or self.cfg.default_provider}")
        elif cmd == "voice":
            if arg == "on":
                self.voice_on = True
                self._note("voice on")
            elif arg == "off":
                self.voice_on = False
                self._note("voice off")
            elif arg == "status":
                self._note(f"voice: {'on' if self.voice_on else 'off'}")
            else:
                self._note(f"voice: {'on' if self.voice_on else 'off'} "
                           "(usage: /voice [on|off|status])")
            self._refresh_status()
        elif cmd in ("img", "image", "imagen"):
            if not arg:
                if self.pending_images:
                    self._note("imgs pendientes: " +
                               " ".join(self.pending_images))
                else:
                    self._note("no hay imágenes pendientes (usage: /img ruta)")
            else:
                from .cli import _encode_image

                try:
                    _encode_image(arg)
                except (ValueError, OSError) as exc:
                    self._note(f"error: {exc}")
                else:
                    self.pending_images.append(arg)
                    self._note(f"(img añadida: {arg})")
                    self._refresh_status()
        else:
            self._note(f"unknown command: /{cmd}")

    # ---- the one turn engine ------------------------------------------

    def _run_turn(self, line: str) -> None:
        self.busy = True
        self._refresh_status(working="⏳ respondiendo")
        self._turn_worker(line)

    def _turn_worker(self, line: str) -> None:
        """Real streaming runs in a Textual thread worker; tests inject
        `stream` and it costs the same path as the CLI.
        """
        from .cli import _do_turn, _switchable
        from .waifu import waifu_send

        def work() -> None:
            _dbg("turn starts: %r", line)
            images = list(self.pending_images)
            self.pending_images.clear()
            current_stream = None
            if self.stream is not None:
                # the plain stream closure mirror: current_stream() -> fn
                current_stream = lambda: self.stream  # noqa: E731
            waifu_send({"type": "state", "state": "talk"})
            try:
                state = _do_turn(
                    self.cfg, self.provider_override, None,
                    self.model_override, self.history, line,
                    self.emit_threaded, current_stream, self.tools_on,
                    self.registry, self.client_fx, self.max_rounds,
                    images=images)
                reply = (self.history[-1]["content"]
                         if (self.history
                             and self.history[-1]["role"] == "assistant")
                         else "")
                if reply and state in ("ok", "tools-unavailable"):
                    waifu_send({"type": "say", "text": reply})
                if (self.voice_on and state in ("ok", "tools-unavailable")
                        and reply):
                    self.voice_fx_speak(reply)
            except Exception as exc:  # _do_turn notes expected errors
                _dbg("turn raised: %s: %s", type(exc).__name__, exc)
                self.emit_threaded(f"error: {exc}")
            finally:
                waifu_send({"type": "state", "state": "idle"})
                _dbg("turn done")
                try:
                    self.call_from_thread(self._finish_turn, line)
                except Exception:  # app is closing; nothing to update
                    pass
                _dbg("finish scheduled")

        self.run_worker(work, thread=True, exclusive=True, name="ana-turn")

    def emit_threaded(self, text: str) -> None:
        """Called from the worker thread; safe to schedule on the event loop."""
        _dbg("emit chunk: %r", text)
        try:
            self.call_from_thread(self._append, text)
        except Exception as exc:  # app closed mid-stream or a widget bug
            # diagnostics only: silent swallowing HID real bugs (the
            # RichLog.write regression) — never stay silent without a log.
            _dbg("emit failed: %s: %s", type(exc).__name__, exc)

    def _finish_turn(self, line: str) -> None:
        _dbg("finish runs on loop")
        self.busy = False
        self._stream_widget = None
        self._stream_buf = ""
        self._refresh_status()
