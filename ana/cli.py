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
import sys
from collections.abc import Callable, Iterator

from . import __version__
from .agent import AgentError, ToolRegistry, run_agent
from .client import ProviderClient, ProviderError
from .config import Config, ConfigError, Provider, load_config, write_template, default_config_path
from .tools import build_default_tools
from .voice import VoiceError
from .voice.stt import default_max_seconds, stdin_press

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
    vfx = voice_fx if voice_fx is not None else _DefaultVoice()

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
                emit("(🎙 Enter para empezar, Enter para terminar "
                     f"(max {max_secs:g}s))")
                wav = vfx.record(max_secs, press)
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

        # regular chat turn
        turn_state = _do_turn(cfg, provider, models, model, history, line,
                              emit, current_stream, tools_on,
                              active_registry, client_fx, max_rounds)
        if turn_state == "tools-unavailable":
            tools_on = False  # stay off for the rest of the session
        if (voice_on and turn_state in ("ok", "tools-unavailable")
                and history and history[-1]["role"] == "assistant"):
            try:
                vfx.speak(history[-1]["content"])
            except VoiceError as exc:
                emit(f"(voice unavailable: {exc})")
        emit("")

    return history


class _DefaultVoice:
    """Live voice backend: push-to-talk arecord → faster-whisper → Piper.

    Imports the voice modules lazily so plain text sessions never touch
    them. The recorded WAV is deleted as soon as transcription finishes
    (or if recording fails), so mic audio does not linger on disk.
    """

    def record(self, max_seconds: float, press):
        import os
        import tempfile
        from pathlib import Path

        from .voice.stt import record_to_wav

        fd, name = tempfile.mkstemp(prefix="ana-rec-", suffix=".wav")
        os.close(fd)
        os.unlink(name)  # arecord creates the file; avoid a stale empty one
        path = Path(name)
        try:
            return record_to_wav(path, max_seconds=max_seconds, press=press)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def transcribe(self, wav):
        from pathlib import Path

        from .voice.stt import WhisperSTT

        try:
            return WhisperSTT().transcribe(wav)
        finally:
            Path(wav).unlink(missing_ok=True)

    def speak(self, text: str) -> None:
        from .voice.tts import PiperTTS

        PiperTTS().speak(text)


def _do_turn(cfg, provider, models, model, history, line, emit,
             current_stream, tools_on, active_registry, client_fx, max_rounds):
    """Execute one chat turn with tools (or a plain stream fallback).

    ``client_fx`` (test/adapter object) is the turn-level client for both
    paths when given. On a provider rejection of the tools payload
    (HTTP 4xx), the turn retries once without tools; the caller disables
    tools for the rest of the session. Any failure rolls the whole
    user turn out of history.
    """
    base_len = len(history)
    history.append({"role": "user", "content": line})
    active_model = model or _default_model(cfg, provider, models)
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
    parser.add_argument("command", nargs="?", choices=("init", "chat"),
                        help="init: write starter config; chat: REPL (default)")
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
            model = args.model or p.default_model
            try:
                if args.no_tools:
                    stream = build_stream(cfg, p)
                    for text in stream([{ "role": "user",
                                          "content": args.message}],
                                       model, system=cfg.system_prompt or None):
                        print(text, end="", flush=True)
                    print()
                else:
                    run_agent(
                        _turn_client(cfg, args.provider),
                        [{"role": "user", "content": args.message}],
                        model, _default_registry(),
                        emit=lambda t: print(t, end="", flush=True),
                        system=cfg.system_prompt or None,
                    )
                    print()
            except (ProviderError, ConfigError, AgentError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return 0

        history = run_repl(
            cfg_path,
            model_override=args.model,
            provider_override=args.provider,
            tools=not args.no_tools,
            voice=args.voice,
        )
        return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
