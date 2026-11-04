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
import re
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
    if event.get("type") == "quit":
        return "quit", None
    return None, None


def fifo_path() -> Path:
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(xdg_runtime) / "ana" / "waifu.fifo"


DEFAULT_AVOID_MONITOR = "DP-1"   # the user's gaming/video output


def choose_monitor(names, preferred: str | None, avoid: str) -> str:
    """Monitor connector for the waifu, pure so the desktop smoke stays out.

    An explicitly named connector wins (even if it would 'break' the
    avoid rule — the user knows their own screens). Default behavior:
    the first connector that is NOT the avoid one (the user plays on
    DP-1 and wants the waifu OUT of there).
    """
    if preferred and preferred in names:
        return preferred
    others = [n for n in names if n != avoid]
    return others[0] if others else (names[0] if names else avoid)


def blink_params() -> "tuple[int, int]":
    """Blink cadence in MILLISECONDS: (interval_ms, duration_ms).

    $ANA_WAIFU_BLINK accepts "interval_seconds:duration_ms" (e.g. "2:80").
    Defaults: (4000, 150) with random jitter added by the runtime.
    """
    raw = os.environ.get("ANA_WAIFU_BLINK", "").strip()
    try:
        parts = raw.split(":")
        interval_ms = int(float(parts[0]) * 1000)
        duration_ms = int(parts[1]) if len(parts) > 1 else 150
        if 0 < interval_ms <= 30000 and 0 < duration_ms <= 2000:
            return interval_ms, duration_ms
    except (ValueError, IndexError):
        pass
    return 4000, 150


def avatar_size() -> int:
    """Avatar edge in px: $ANA_WAIFU_SIZE or 220."""
    try:
        v = int(os.environ.get("ANA_WAIFU_SIZE", "").strip())
        if 40 <= v <= 2000:
            return v
    except ValueError:
        pass
    return 220


def fade_steps(steps: int = 10) -> list[float]:
    """Monotonic opacity ramp 0→1 for the fade-in."""
    if steps <= 1:
        return [1.0]
    return [round(i / (steps - 1), 4) for i in range(steps)]


_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"(?<![\w*])([*_~]{1,2})([^*_~]+)\1")
# emoji/symbol blocks: piper/espeak would read them literally or choke
_SYMBOL_RANGES = (
    (0x2190, 0x23FF),   # arrows + misc technical (⏱ ⌨)
    (0x25A0, 0x27BF),   # geometric + misc symbols + dingbats (❤ ✨)
    (0x2B00, 0x2BFF),   # misc symbols and arrows
    (0x1F000, 0x1FAFF), # emoji planes
)
_SINGLE_SYMBOLS = {0xFE0F, 0x200D, 0x2049, 0x203C, 0x2122, 0x00A9,
                   0x00AE, 0x20E3, 0x2022}


def _is_symbol(ch: str) -> bool:
    o = ord(ch)
    return o in _SINGLE_SYMBOLS or any(lo <= o <= hi
                                       for lo, hi in _SYMBOL_RANGES)


def sanitize_for_speech(text: str) -> str:
    """Plain prose for the TTS voice and the overlay subtitle.

    Replies arrive as markdown with emoji; Piper/espeak would read the
    markers literally ("asterisco asterisco") and the label would show
    raw ``**``. Spanish punctuation and accented letters (¡ ¿ á é ü…)
    are kept untouched.
    """
    t = _MD_LINK.sub(r"\1", text)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = _EMPHASIS.sub(r"\2", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)
    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", t, flags=re.M)
    t = "".join(ch for ch in t if not _is_symbol(ch))
    return re.sub(r"\s+", " ", t).strip()


def waifu_send(event: dict) -> None:
    """Send one JSON event to the running overlay (if any); never raises.

    The FIFO is created by `ana waifu`; the producers open it
    non-blocking so a missing overlay is a silent nicety.
    """
    import errno

    path = fifo_path()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return  # nobody listening (overlay not running)
    try:
        payload = event
        if event.get("type") == "say" and isinstance(event.get("text"),
                                                     str):
            # the subtitle is speech: never show raw markdown/emojis
            payload = {**event,
                       "text": sanitize_for_speech(event["text"])}
        os.write(fd, (json.dumps(payload) + "\n").encode())
    except OSError as exc:
        if exc.errno == errno.ENXIO:
            pass  # reader not ready: the nicety is a nicety
    finally:
        os.close(fd)


def _waifu_fifo_exists() -> bool:
    return fifo_path().exists()
