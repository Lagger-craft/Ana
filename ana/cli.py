"""ana CLI — interactive chat REPL and one-shot mode.

Commands inside the REPL:

- ``/exit`` (or ``/quit``): leave the chat
- ``/new``: start a new conversation (clears history)
- ``/model [name]``: show or switch the model
- ``/provider [name]``: show or switch the provider
- ``/voice [on|off|status]``: toggle the opt-in voice loop (Phase 2)
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from . import __version__
from .agent import AgentError, ToolRegistry, run_agent
from .client import ProviderClient, ProviderError
from .config import Config, ConfigError, Provider, load_config, write_template, default_config_path
from .tools import build_default_tools
from .voice import VoiceError
from .voice.stt import (default_max_seconds, mic_label, min_peak_db,
                        peak_db, stdin_press)

StreamFn = Callable[..., Iterator[str]]

COMMAND_HELP = (
    "commands: /exit, /new, /model [name], /provider [name], "
    "/tools [on|off|list], /voice [on|off|status]")


def build_stream(cfg: Config, provider: Provider) -> StreamFn:
    """Return a stream function bound to the given provider."""
    client = ProviderClient.from_provider(cfg, provider)

    def stream(messages, model, system=None):
        return client.stream_chat(messages, model, system=system)

    return stream


def parse_command(line: str) -> tuple[str, str]:
    """Split '/cmd arg' into ('cmd', 'arg'). Non-commands raise ValueError."""
    if not line.startswith("/"):
        raise ValueError(f"not a command: {line!r}")
    parts = line[1:].split(None, 1)
    if not parts:
        raise ValueError("empty command")
    return parts[0], (parts[1] if len(parts) > 1 else "")


def run_repl(
    cfg_path,
    lines=None,
    stream: StreamFn | None = None,
    emit: Callable[[str], None] | None = None,
    models: dict[str, str] | None = None,
    model_override: str | None = None,
    provider_override: str | None = None,
    tools: bool = True,
    registry: ToolRegistry | None = None,
    client_fx: object | None = None,
    max_rounds: int = 4,
    voice: bool = False,
    voice_secs: float | None = None,
    voice_fx: object | None = None,
    stt: str | None = None,
    images: list[str] | None = None,
) -> list[dict[str, str]]:
    """Run the interactive chat loop.

    ``lines`` lets tests script input; ``stream`` lets tests supply a
    canned backend. Returns the conversation history (user + assistant
    messages) accumulated during the session.

    ``voice`` starts the session in voice mode (S9): push-to-talk — the
    round prints ``(🎙 Enter para empezar, Enter para terminar (max Ns))``,
    the first Enter starts recording and the second Enter stops it (or
    ``voice_secs`` / ``ANA_VOICE_MAX_SECONDS`` / 12 s, whichever comes
    first); the transcript is transcribed locally and runs through the
    normal turn, and the final reply is spoken. ``voice_fx`` is the
    injectable seam with ``record`` / ``transcribe`` / ``speak`` methods
    (tests use fakes; the default is the live :class:`_DefaultVoice`
    backend). Voice errors degrade to ``(voice unavailable: ...)``
    without leaving the REPL.
    """
    cfg = load_config(cfg_path)
    emit = emit or (lambda text: print(text, flush=True))
    prompt = "you> " if lines is None else ""

    scripted = iter(lines) if lines is not None else None
    provider = provider_override or cfg.default_provider
    model: str | None = model_override
    history: list[dict[str, str]] = []
    injected = stream is not None
    stream_fn = stream
    tools_on = tools
    active_registry: ToolRegistry | None = registry
    voice_on = voice
    vfx = voice_fx if voice_fx is not None else _DefaultVoice(stt)
    pending_images: list[str] = list(images or [])

    def press(timeout: float | None = None) -> bool:
        """Push-to-talk click seam: block until Enter (or up to timeout).

        Scripted ``lines`` presses arrive instantly; exhausting them
        counts as "no press" when a timeout applies (auto-stop).
        """
        if scripted is not None:
            try:
                next(scripted)
                return True
            except StopIteration:
                if timeout is not None:
                    return False
                raise VoiceError(
                    "input exhausted before recording started") from None
        return stdin_press(timeout)

    def current_stream() -> StreamFn:
        nonlocal stream_fn
        if stream_fn is None:
            stream_fn = _switchable(cfg, provider)
        return stream_fn

    while True:
        try:
            if scripted is None:
                line = input(prompt).strip()
            else:
                line = next(scripted).strip()
        except StopIteration:
            break
        except EOFError:
            break
        except KeyboardInterrupt:
            emit("\n(bye)")
            break

        if not line:
            if not voice_on:
                continue
            # push-to-talk voice round (S9): Enter starts, Enter stops
            try:
                max_secs = (voice_secs if voice_secs is not None
                            else default_max_seconds())
                emit(f"(🎙 mic: {mic_label()} · Enter para empezar, "
                     f"Enter para terminar (max {max_secs:g}s))")
                wav = vfx.record(max_secs, press)
                try:
                    level = peak_db(wav)
                except VoiceError:
                    level = None  # unreadable wav: transcribe will report it
                if level is not None and level < min_peak_db():
                    emit(f"(🎙 grabación casi silenciosa ({level:.1f} dB) "
                         "— hablá cerca o revisá el mute del mic)")
                heard = (vfx.transcribe(wav) or "").strip()
            except VoiceError as exc:
                emit(f"(voice unavailable: {exc})")
                continue
            except KeyboardInterrupt:
                emit("\n(record stopped)")
                continue
            if not heard:
                emit("(no escuché nada)")
                continue
            emit(f"you> {heard}")
            line = heard

        if line.startswith("/"):
            try:
                cmd, arg = parse_command(line)
            except ValueError as exc:
                emit(f"error: {exc}")
                continue
            if cmd in ("exit", "quit"):
                break
            elif cmd == "new":
                history.clear()
                continue
            elif cmd in ("img", "image", "imagen"):
                if not arg:
                    if pending_images:
                        emit("imgs pendientes: " + " ".join(pending_images))
                    else:
                        emit("no hay imágenes pendientes "
                             "(usage: /img <ruta>)")
                else:
                    try:
                        _encode_image(arg)  # validate before queueing
                    except (ValueError, OSError) as exc:
                        emit(f"error: {exc}")
                    else:
                        pending_images.append(arg)
                        emit(f"(img añadida: {arg})")
                continue
            elif cmd == "model":
                if arg:
                    model = arg
                else:
                    active = model or _default_model(cfg, provider, models)
                    emit(f"model: {active}")
                continue
            elif cmd == "provider":
                if arg:
                    try:
                        cfg.provider(arg)
                        provider = arg
                        model = None  # adopt the provider's default_model
                        if not injected:
                            stream_fn = None  # rebuild the client next turn
                    except ConfigError as exc:
                        emit(f"error: {exc}")
                else:
                    emit(f"provider: {provider} "
                         f"(model: {model or _default_model(cfg, provider, models)})")
                continue
            elif cmd == "tools":
                if arg == "off":
                    tools_on = False
                    active_registry = None
                    emit("tools off")
                elif arg == "on":
                    tools_on = True
                    active_registry = registry or _default_registry()
                    emit(f"tools on ({', '.join(active_registry.names())})")
                elif arg == "list":
                    names = (active_registry or _default_registry()).names()
                    emit("tools: " + (", ".join(names) or "(none)"))
                else:
                    current = "on" if tools_on else "off"
                    emit(f"tools: {current} (usage: /tools [on|off|list])")
                continue
            elif cmd == "voice":
                if arg == "on":
                    voice_on = True
                    emit("voice on")
                elif arg == "off":
                    voice_on = False
                    emit("voice off")
                elif arg == "status":
                    emit(f"voice: {'on' if voice_on else 'off'}")
                else:
                    current = "on" if voice_on else "off"
                    emit(f"voice: {current} "
                         f"(usage: /voice [on|off|status])")
                continue
            else:
                emit(f"unknown command: /{cmd} ({COMMAND_HELP})")
                continue

        # regular chat turn: pending images ride this message once
        attachments = list(pending_images)
        pending_images.clear()
        turn_state = _do_turn(cfg, provider, models, model, history, line,
                              emit, current_stream, tools_on,
                              active_registry, client_fx, max_rounds,
                              images=attachments)
        if turn_state == "tools-unavailable":
            tools_on = False  # stay off for the rest of the session
        # S15b: the REPL drives the overlay like the TUI does — the
        # subtitle gets every NEW reply (never a rolled-back stale one),
        # and voice wraps the speak with talk→idle. waifu_send is a
        # silent no-op when no overlay listens.
        from .waifu import waifu_send

        if (turn_state in ("ok", "tools-unavailable")
                and history and history[-1]["role"] == "assistant"):
            reply = history[-1]["content"]
            waifu_send({"type": "say", "text": reply})
            if voice_on:
                waifu_send({"type": "state", "state": "talk"})
                try:
                    vfx.speak(reply)
                except VoiceError as exc:
                    emit(f"(voice unavailable: {exc})")
                finally:
                    waifu_send({"type": "state", "state": "idle"})
        emit("")

    return history


class _DefaultVoice:
    """Live voice backend: push-to-talk arecord → STT → Piper.

    STT backend resolution follows S11: $ANA_STT or the --stt flag;
    default voxtype (falls back details in ana.voice.stt.build_stt).
    Imports the voice modules lazily so plain text sessions never touch
    them. The recorded WAV is deleted as soon as transcription finishes
    (or if recording fails), so mic audio does not linger on disk.
    """

    def __init__(self, stt_backend: str | None = None) -> None:
        self._stt_backend = stt_backend

    def record(self, max_seconds: float, press):
        import os
        import tempfile
        from pathlib import Path

        from .voice.media import pause_media, resume_media
        from .voice.stt import record_to_wav

        fd, name = tempfile.mkstemp(prefix="ana-rec-", suffix=".wav")
        os.close(fd)
        os.unlink(name)  # arecord creates the file; avoid a stale empty one
        path = Path(name)
        # voxtype-parity: playing media (browser videos, games) dominates
        # the mic feed; pause MPRIS players for the capture and resume
        # them after, even on error. ANA_VOICE_PAUSE_MEDIA=0 opts out.
        paused = (os.environ.get("ANA_VOICE_PAUSE_MEDIA", "1") != "0"
                  and pause_media())
        try:
            return record_to_wav(path, max_seconds=max_seconds, press=press)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            if paused:
                resume_media(paused)

    def transcribe(self, wav):
        from pathlib import Path

        from .voice.stt import build_stt

        try:
            return build_stt(self._stt_backend).transcribe(wav)
        finally:
            Path(wav).unlink(missing_ok=True)

    def speak(self, text: str) -> None:
        from .voice.tts import PiperTTS

        PiperTTS().speak(text)


def _do_turn(cfg, provider, models, model, history, line, emit,
             current_stream, tools_on, active_registry, client_fx, max_rounds,
             images=None):
    """Execute one chat turn with tools (or a plain stream fallback).

    ``client_fx`` (test/adapter object) is the turn-level client for both
    paths when given. On a provider rejection of the tools payload
    (HTTP 4xx), the turn retries once without tools; the caller disables
    tools for the rest of the session. Any failure rolls the whole
    user turn out of history. ``images`` (pending attachments) ride on
    this user message once and then leave history untouched.
    """
    has_images = bool(images)
    base_len = len(history)
    history.append(_user_message(line, images))
    active_model = _model_for_turn(
        cfg.provider(provider), model, has_images=has_images,
        fallback=(None if model is not None
                  else _default_model(cfg, provider, models)) or "")
    system = cfg.system_prompt or None
    used_tools = tools_on and active_registry is not None and len(active_registry)
    try:
        if used_tools:
            client = client_fx if client_fx is not None else _turn_client(cfg, provider)
            run_agent(
                client, history, active_model, active_registry,
                emit=emit, system=system, max_rounds=max_rounds,
                on_tool=lambda name, args: emit(f"⚙ {name}({args})"),
            )
        else:
            _plain_stream(history, active_model, system, emit,
                          current_stream, client_fx)
    except (ProviderError, ConfigError, AgentError) as exc:
        msg = str(exc)
        if used_tools and "HTTP 4" in msg:
            # the rejection applies to the tools payload; go plain once
            del history[base_len:]
            history.append({"role": "user", "content": line})
            try:
                _plain_stream(history, active_model, system, emit,
                              current_stream, client_fx)
                emit("(tools disabled for this model, continuing plain)")
                return "tools-unavailable"
            except (ProviderError, ConfigError, AgentError) as retry_exc:
                emit(f"error: {retry_exc}")
                del history[base_len:]
                emit("")
                return "error"
        emit(f"error: {msg}")
        del history[base_len:]  # roll back this whole turn
        emit("")  # keep typing UX sane after a streamed failure
        return "error"
    return "ok"


def _plain_stream(history, active_model, system, emit, current_stream,
                  client_fx):
    """Stream a reply without tools; append assistant to history."""
    reply = []
    if client_fx is not None:
        events = client_fx.stream_turn(history, active_model, system=system,
                                       tools=None)
        for ev in events:
            if ev["type"] == "text":
                emit(ev["content"])
                reply.append(ev["content"])
            elif ev["type"] == "finish":
                break
    else:
        for text in current_stream()(history, active_model, system=system):
            emit(text)
            reply.append(text)
    history.append({"role": "assistant", "content": "".join(reply)})


def _turn_client(cfg: Config, provider_name: str) -> ProviderClient:
    p = cfg.provider(provider_name)
    return ProviderClient.from_provider(cfg, p)


def _default_registry() -> ToolRegistry:
    return build_default_tools()


def _probe_image_size(path: Path):
    """(width, height) from ffprobe; None when ffprobe can't tell."""
    try:
        proc = _probe_run(["ffprobe", "-v", "error", "-show_streams",
                           "-of", "json", str(path)])
    except OSError:
        return None
    try:
        stream = json.loads(proc.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def _shrink_image(path: Path, max_px: int) -> bytes | None:
    """Resized bytes (same format) when ffmpeg can build them."""
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=path.suffix or ".png",
                                         delete=False) as out:
            out_path = out.name
        proc = _probe_run([
            "ffmpeg", "-y", "-v", "error", "-i", str(path),
            "-vf", (f"scale='min({max_px},iw)':'min({max_px},ih)':"
                    "force_original_aspect_ratio=decrease"),
            out_path,
        ])
        if Path(out_path).exists():
            data = Path(out_path).read_bytes()
            Path(out_path).unlink(missing_ok=True)
            if proc.returncode == 0 and data:
                return data
        Path(out_path).unlink(missing_ok=True)
    except OSError:
        pass
    return None


def _probe_run(cmd, **kw):
    """Subprocess seam for ffprobe/ffmpeg (tests monkeypatch this)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return proc


def _encode_image(file_path: str | Path) -> dict:
    """File path -> {media_type, data(base64)} for a message image.

    Images larger than $ANA_IMAGE_MAX_PX (default 1024) on either side
    are rescheduled with ffmpeg first: the vision encoder's grid scales
    with pixels, so a 4K screenshot burns quadruple the prompt tokens
    of the same shot at 1024. Lower res passes through untouched.
    Raises ValueError with an actionable message when the path is not
    an image or does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"la imagen no existe: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        suffix = path.suffix or "(sin extensión)"
        raise ValueError(
            f"{path.name} ({suffix}) no es una imagen soportada "
            "(png, jpg, webp, gif...)")
    data = path.read_bytes()
    limit = _image_max_px()
    if limit > 0:
        size = _probe_image_size(path)
        if size and (size[0] > limit or size[1] > limit):
            data = _shrink_image(path, limit) or data
    return {"media_type": mime, "data": base64.b64encode(data).decode(
        "ascii")}


def _image_max_px() -> int:
    """Resize threshold: $ANA_IMAGE_MAX_PX or 1024; 0 disables."""
    raw = os.environ.get("ANA_IMAGE_MAX_PX", "").strip()
    try:
        return int(raw) if raw else 1024
    except ValueError:
        return 1024


def _user_message(text: str, image_paths) -> dict:
    """The user turn dict; images attach only when paths are pending."""
    if image_paths:
        return {"role": "user", "content": text,
                "images": [_encode_image(p) for p in image_paths]}
    return {"role": "user", "content": text}


def _model_for_turn(provider: Provider, explicit: str | None,
                    has_images: bool, fallback: str = "") -> str:
    """Model id for one turn: explicit > vision_model (images) > fallback.

    ``fallback`` preserves the /model and config-level override order the
    plain-text path already had (it defaults to provider.default_model).
    """
    if explicit:
        return explicit
    if has_images and provider.vision_model:
        return provider.vision_model
    return fallback or provider.default_model


def _default_model(cfg: Config, provider_name: str, models: dict[str, str] | None) -> str:
    if models and provider_name in models:
        return models[provider_name]
    return cfg.provider(provider_name).default_model


def _switchable(cfg: Config, provider_name: str) -> StreamFn:
    """Stream fn that resolves the provider at call time, so /provider-now
    switches take effect without rebuilding the caller's closure."""

    def switchable(messages, model, system=None):
        fn = build_stream(cfg, cfg.provider(provider_name))
        return fn(messages, model, system=system)

    return switchable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ana",
        description="Ana - waifu virtual assistant (Phase 0: chat CLI)",
    )
    parser.add_argument("--version", action="version",
                        version=f"ana {__version__}")
    parser.add_argument("--tui", action="store_true",
                        help="full-screen Textual interface (same as "
                             "`ana tui`)")
    parser.add_argument("command", nargs="?",
                        choices=("init", "chat", "tui", "waifu", "talk"),
                        help="init: write starter config; chat: REPL "
                             "(default); tui: full-screen interface; "
                             "waifu: PNG avatar overlay window; talk: "
                             "headless voice hotkey — record → turn → "
                             "spoken reply, no window (global "
                             "push-to-talk for Hyprland)")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("--provider", help="provider name from the config")
    parser.add_argument("--model", help="model id override")
    parser.add_argument("--message", "-m",
                        help="one-shot prompt; print and exit")
    parser.add_argument("--no-tools", action="store_true",
                        help="disable tool calling for this session")
    parser.add_argument("--voice", action="store_true",
                        help="voice mode: push-to-talk (Enter starts/"
                             "stops recording; local STT, replies are "
                             "spoken via Piper)")
    parser.add_argument("--stt", choices=("voxtype", "whisper"),
                        help="STT backend for --voice; default voxtype "
                             "(same engine as local dictation), "
                             "ANA_STT also works")
    parser.add_argument("--image", "-i", action="append", metavar="RUTA",
                        help="image to attach to -m (png/jpg/webp/gif); "
                             "the provider's vision_model is used for it")

    args = parser.parse_args(argv)

    if args.command == "init":
        path = default_config_path()
        if path.exists():
            print(f"config already exists: {path}")
            return 0
        write_template(path)
        print(f"wrote starter config: {path}")
        return 0

    cfg_path = args.config
    if cfg_path is None:
        cfg_path = default_config_path()

    stream_fn_provider = args.provider
    try:
        if args.message:
            cfg = load_config(cfg_path)
            p = cfg.provider(args.provider)
            attached = args.image or []
            user_msg = _user_message(args.message, attached)
            model = args.model or (
                p.vision_model if attached and p.vision_model
                else p.default_model)
            try:
                if args.no_tools:
                    stream = build_stream(cfg, p)
                    for text in stream([user_msg],
                                       model, system=cfg.system_prompt or None):
                        print(text, end="", flush=True)
                    print()
                else:
                    run_agent(
                        _turn_client(cfg, args.provider),
                        [user_msg],
                        model, _default_registry(),
                        emit=lambda t: print(t, end="", flush=True),
                        system=cfg.system_prompt or None,
                    )
                    print()
            except (ProviderError, ConfigError, AgentError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return 0

        if args.command == "waifu":
            from .overlay import main as overlay_main, run_overlay

            return run_overlay()

        if args.command == "talk":
            # headless voice hotkey (S15c): no TTY, no window — state
            # under $XDG_RUNTIME_DIR/ana/, toggle per key press
            from .talk import run_talk

            return run_talk(cfg_path, stt=args.stt)

        if args.tui or args.command == "tui":
            from .tui import run_tui

            run_tui(
                cfg_path,
                model_override=args.model,
                provider_override=args.provider,
                tools=not args.no_tools,
                images=args.image or [],
            )
            return 0

        history = run_repl(
            cfg_path,
            model_override=args.model,
            provider_override=args.provider,
            tools=not args.no_tools,
            voice=args.voice,
            stt=args.stt,
            images=args.image or [],
        )
        return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
