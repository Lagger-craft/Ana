"""Waifu overlay logic (S14): state machine and asset resolution, pure.

The GTK shell (ana/overlay.py) consumes these names; everything testable
lives here so the window stays thin. Inputs are JSON lines over stdin or
a FIFO:

    {"type": "state", "state": "talk"}   -> avatar mouth states
    {"type": "say",   "text": "..."}     -> subtitle line

Assets: ~/.local/share/ana/waifu/{idle,talk,blink}.png (1k px, with
transparency); $ANA_WAIFU_DIR overrides. Sources for packs are in the
README (itch.io CC0 pngtuber packs, Inochi2D sample Aka, waifu2x).
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path

__all__ = ["WaifuState", "parse_event", "resolve_asset", "waifu_dir"]


class WaifuState(Enum):
    IDLE = "idle"
    TALK = "talk"
    BLINK = "blink"

    @classmethod
    def coerce(cls, name: str) -> "WaifuState":
        """Known name → that state; anything else falls back to IDLE."""
        try:
            return cls(name)
        except ValueError:
            return cls.IDLE


def waifu_dir() -> Path:
    env = os.environ.get("ANA_WAIFU_DIR", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".local" / "share" / "ana" / "waifu"


_WAIFU_DIR = None  # test seam: a Path override wins over env/home


def waifu_dir() -> Path:
    if _WAIFU_DIR is not None:   # internal/test seam takes precedence
        return _WAIFU_DIR
    env = os.environ.get("ANA_WAIFU_DIR", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".local" / "share" / "ana" / "waifu"


def resolve_asset(state: str) -> Path | None:
    """The PNG for a WaifuState when the file exists; None otherwise."""
    try:
        return waifu_file(state)
    except FileNotFoundError:
        return None


def waifu_file(state: str) -> Path:
    root = waifu_dir()
    for suffix in (".png", ".webp", ".gif"):
        candidate = root / f"{state}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no {state}.* under {root}")


def parse_event(line: str) -> "tuple[str | None, str | None]":
    """(state, subtitle) from one JSON line; None where unchanged."""
    try:
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        return None, None
    if event.get("type") == "state":
        state = str(event.get("state", ""))
        known = {s.value for s in WaifuState}
        return (state if state in known else None), None
    if event.get("type") == "say":
        text = str(event.get("text", ""))
        return None, (text if text.strip() else None)
    return None, None


def fifo_path() -> Path:
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(xdg_runtime) / "ana" / "waifu.fifo"
