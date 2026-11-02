"""Textual TUI (Phase 4, S13): one turn engine, screen frontend.

Public names: run_tui (entry from `ana tui` / `ana --tui`), AnaApp.
The app drives the chat through ana.cli._do_turn — provider, tools,
image queue, vision_model pick and voice-free streaming are THE SAME
behavior as the CLI (no forked engine). Textual is an optional
dependency: importing without the `tui` extra degrades to a plain
object backend with an actionable run_tui error.
"""

from __future__ import annotations

import sys

ANA_TUI_MISSING_MSG = (
    "textual not installed — run: .venv/bin/pip install -e '.[tui]'")

try:  # the `tui` extra; AnaApp stays importable without it
    from textual.app import App, ComposeResult
    from textual.containers import Vertical
    from textual.widgets import Input, RichLog, Static

    _TEXTUAL_INSTALLED = True
except ImportError:  # tui extra optional — plain chat still works
    App = object  # type: ignore[assignment,misc]
    ComposeResult = None
    _TEXTUAL_INSTALLED = False


def import_textual():
    """Fail closed when the tui extra is not installed."""
    if not _TEXTUAL_INSTALLED:
        print("error: " + ANA_TUI_MISSING_MSG, file=sys.stderr)
        raise SystemExit(1)


def run_tui(cfg_path, model_override=None, provider_override=None,
            tools=True, images=None):
    """Entry point for `ana tui` / `ana --tui`."""
    import_textual()
    AnaApp(cfg_path, model_override=model_override,
           provider_override=provider_override, tools=tools,
           images=images).run()


class AnaApp(App if _TEXTUAL_INSTALLED else object):
    """Chat screen: status line, scrolling transcript, one input."""

    if _TEXTUAL_INSTALLED:
        CSS = """
        #status { height: 1; background: $boost; color: $text; padding: 0 1; }
        #chat { border: none; }
        Input { border: none; background: $surface; }
        """

    def __init__(self, cfg_path, model_override=None,
                 provider_override=None, tools=True, registry=None,
                 client_fx=None, stream=None, images=None,
                 max_rounds: int = 4):
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
        self.busy = True  # until the first mount finishes loading

    # ---- widgets ------------------------------------------------------

    def compose(self):
        yield Static(id="status")
        yield RichLog(id="chat", markup=False, wrap=True)
        yield Input(id="entry")

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
        if self.query_one("#entry"):
            self.query_one("#entry").focus()

    def _set_status(self, text: str) -> None:
        if self.is_running and _TEXTUAL_INSTALLED:
            self.query_one("#status").update(text)

    def _refresh_status(self):
        if not hasattr(self, "cfg") or self.cfg is None:
            self._set_status("ana: config not loaded")
            return
        from .cli import _default_model

        provider = self.provider_override or self.cfg.default_provider
        model = self.model_override or _default_model(self.cfg, provider,
                                                      None)
        self._set_status(
            f"model: {model} · provider: {provider} · "
            f"tools: {'on' if self.tools_on else 'off'} · "
            f"imgs: {len(self.pending_images)}")

    # ---- transcript ---------------------------------------------------

    def _note(self, text: str) -> None:
        self.transcript += ("" if not self.transcript else "\n") + text
        if hasattr(self, "is_running") and self.is_running and _TEXTUAL_INSTALLED:
            chat = self.query_one("#chat")
            chat.write(text)
            chat.scroll_end(animate=False)

    def _append(self, chunk: str) -> None:
        self.transcript += chunk
        if self.is_running and _TEXTUAL_INSTALLED:
            chat = self.query_one("#chat")
            chat.write(chunk, no_wrap=False)
            chat.scroll_end(animate=False)

    # ---- input --------------------------------------------------------

    def on_input_submitted(self, event) -> None:
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
            return
        self._note(f"you> {text}")
        if text.startswith("/"):
            self._command(text)
        else:
            self._run_turn(text)

    def _command(self, line: str) -> None:
        from .cli import parse_command

        cmd, arg = parse_command(line)  # ValueError: unknown shape
        if cmd in ("exit", "quit"):
            self.exit()
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
        self._turn_worker(line)

    def _turn_worker(self, line: str) -> None:
        """Real streaming runs in a Textual thread worker; tests inject
        `stream` and it costs the same path as the CLI.
        """
        from .cli import _do_turn, _switchable

        def work() -> None:
            images = list(self.pending_images)
            self.pending_images.clear()
            current_stream = None
            if self.stream is not None:
                # the plain stream closure mirror: current_stream() -> fn
                current_stream = lambda: self.stream  # noqa: E731
            try:
                _do_turn(
                    self.cfg, self.provider_override, None,
                    self.model_override, self.history, line,
                    self.emit_threaded, current_stream, self.tools_on,
                    self.registry, self.client_fx, self.max_rounds,
                    images=images)
            except Exception as exc:  # _do_turn notes expected errors
                self.emit_threaded(f"error: {exc}")
            finally:
                try:
                    self.call_from_thread(self._finish_turn, line)
                except Exception:  # app is closing; nothing to update
                    pass

        self.run_worker(work, thread=True, exclusive=True, name="ana-turn")

    def emit_threaded(self, text: str) -> None:
        """Called from the worker thread; safe to schedule on the event loop."""
        try:
            self.call_from_thread(self._append, text)
        except Exception:  # app closed mid-stream; nothing to update
            pass

    def _finish_turn(self, line: str) -> None:
        self.busy = False
        self._refresh_status()
