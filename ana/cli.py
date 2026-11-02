"""ana CLI — interactive chat REPL and one-shot mode.

Commands inside the REPL:

- ``/exit`` (or ``/quit``): leave the chat
- ``/new``: start a new conversation (clears history)
- ``/model [name]``: show or switch the model
- ``/provider [name]``: show or switch the provider
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator

from . import __version__
from .client import ProviderClient, ProviderError
from .config import Config, ConfigError, Provider, load_config, write_template, default_config_path

StreamFn = Callable[..., Iterator[str]]

COMMAND_HELP = (
    "commands: /exit, /new, /model [name], /provider [name]")


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
) -> list[dict[str, str]]:
    """Run the interactive chat loop.

    ``lines`` lets tests script input; ``stream`` lets tests supply a
    canned backend. Returns the conversation history (user + assistant
    messages) accumulated during the session.
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
            continue

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
            else:
                emit(f"unknown command: /{cmd} ({COMMAND_HELP})")
                continue

        # regular chat turn
        history.append({"role": "user", "content": line})
        active_model = model or _default_model(cfg, provider, models)
        chunks: Iterator[str] = iter(())
        try:
            chunks = current_stream()(history, active_model,
                                     system=cfg.system_prompt or None)
            reply = []
            for text in chunks:
                emit(text)
                reply.append(text)
        except (ProviderError, ConfigError) as exc:
            emit(f"error: {exc}")
            history.pop()  # roll back the failed turn
            emit("")  # keep typing UX sane after a streamed failure
            continue
        history.append({"role": "assistant", "content": "".join(reply)})
        emit("")

    return history


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
            stream = build_stream(cfg, p)
            model = args.model or p.default_model
            try:
                for text in stream([{"role": "user", "content": args.message}],
                                   model, system=cfg.system_prompt or None):
                    print(text, end="", flush=True)
                print()
            except (ProviderError, ConfigError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return 0

        history = run_repl(
            cfg_path,
            model_override=args.model,
            provider_override=args.provider,
        )
        return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
