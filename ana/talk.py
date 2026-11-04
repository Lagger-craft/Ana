"""Headless voice hotkey round (S15c / T35): the `ana talk` subcommand.

One keybinding press = one full voice round with NO window and NO
terminal focus, so a global binding can trigger it — on KDE Plasma
(System Settings → Shortcuts → command "Ana — Hablar"):

    Exec=/path/to/Ana/scripts/ana-talk-hotkey

or on Hyprland:

    bind = SUPER ALT, A, exec, /path/to/.venv/bin/ana talk

Global bindings fire on key press only (no release), so this is a
two-press toggle, not hold-to-talk. Session state lives beside the
waifu FIFO under $XDG_RUNTIME_DIR/ana/:

- press 1 (no state): write ``talk.json`` (pid + phase "recording"),
  notify, then record the mic — stopping at ``default_max_seconds()``
  (ANA_VOICE_MAX_SECONDS, default 12) or when ``talk.stop`` appears,
  whichever comes first;
- press 2 (phase "recording"): touch ``talk.stop`` only and exit 0
  immediately — never blocks;
- press 3 (phase "answering", worker past the mic): no-op — creating a
  stop-file now would corrupt the next round.

A dead pid in talk.json means the worker crashed mid-round: the state
is treated as absent and a fresh round starts. The worker clears both
files in a ``finally`` on every exit path.

The transcript runs through the EXISTING turn engine — ``run_repl``
with ``lines=[transcript, "/exit"]``, ``voice=True`` and
``close_overlay=False`` (the overlay stays up across hotkey rounds —
Option B's closing ``quit`` only applies to the chat frontends) — so
tools, provider config, error rollback, TTS (Kokoro by default) and
the waifu say/talk/idle events (T34) all come for free; this module is
orchestration only.

Headless feedback: a tiny ``notify()`` via ``notify-send`` (all
failures swallowed silently) and one ``waifu_send`` say event at
recording start. Streamed turn output goes nowhere by default (no TTY
to write to); set ``ANA_TALK_DEBUG=/path/to/log`` to append it to a
file instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from . import waifu
from .waifu import waifu_send
from .cli import _DefaultVoice, run_repl
from .voice import VoiceError
from .voice.stt import default_max_seconds

__all__ = ["make_press", "notify", "read_state", "run_talk",
           "talk_state_path", "talk_stop_path", "write_state"]

_POLL_SECONDS = 0.1          # stop-file poll interval while recording
_NOTIFY_TIMEOUT = 5.0        # notify-send subprocess budget
_PHASES = ("recording", "answering")


# ------------------------------------------------------------- state


def _runtime_dir() -> Path:
    """$XDG_RUNTIME_DIR/ana/ — the same dir as the waifu FIFO.

    Reached through the waifu module (not a bound ``fifo_path`` name)
    so the test suite's global fifo re-pointing redirects talk state
    into the test sandbox too.
    """
    return waifu.fifo_path().parent


def talk_state_path() -> Path:
    return _runtime_dir() / "talk.json"


def talk_stop_path() -> Path:
    return _runtime_dir() / "talk.stop"


def write_state(phase: str) -> None:
    """Record this process as the live worker in ``phase``."""
    path = talk_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "phase": phase}),
                    encoding="utf-8")


def read_state() -> dict | None:
    """Live session state, or None when absent/stale/corrupt.

    A recorded pid that is dead (crashed worker) means stale state:
    the next press must start a fresh round instead of signaling it.
    """
    try:
        data = json.loads(talk_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    pid, phase = data.get("pid"), data.get("phase")
    if not isinstance(pid, int) or phase not in _PHASES:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None  # stale: the worker died mid-round
    except PermissionError:
        pass  # alive, owned by another user
    except (OverflowError, OSError):
        return None  # unusable pid
    return data


def clear_state() -> None:
    """Best-effort removal of both session files (``finally`` path)."""
    for path in (talk_state_path(), talk_stop_path()):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


# -------------------------------------------------------- press seam


def make_press(stop_path: Path, poll: float = _POLL_SECONDS):
    """The ``record_to_wav`` press seam for a headless round.

    Contract (ana.voice.stt.record_to_wav): ``press()`` → start now;
    ``press(timeout=N)`` → True = stop now, False = timeout elapsed.
    The first call returns immediately (nothing blocks on stdin); the
    timed call polls ``stop_path`` — the second hotkey press creates
    it — and reports timeout when no stop-file arrives in time
    (auto-stop at ``default_max_seconds()``).
    """
    def press(timeout: float | None = None) -> bool:
        if timeout is None:
            return True  # start recording right away (headless)
        deadline = time.monotonic() + timeout
        while True:
            if stop_path.exists():
                return True  # second hotkey press cut the recording
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False  # max seconds reached: auto-stop
            time.sleep(min(poll, remaining))

    return press


# ----------------------------------------------------------- notify


def notify(title: str, body: str) -> None:
    """Desktop notification via ``notify-send``; never raises.

    Headless runs have no dbus/binary in every environment (tests
    included): any failure — missing binary, nonzero exit, timeout —
    is swallowed silently. The waifu overlay subtitle is a bonus, not
    the channel.
    """
    try:
        subprocess.run(["notify-send", str(title), str(body)],
                       timeout=_NOTIFY_TIMEOUT,
                       capture_output=True, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _make_emit():
    """The run_repl ``emit`` for a terminal-less run.

    Default: discard (no TTY — nothing to paint). With
    ``ANA_TALK_DEBUG=/path/to/log`` the streamed turn output is
    appended to that file for post-mortems.
    """
    log_path = os.environ.get("ANA_TALK_DEBUG", "").strip()
    if not log_path:
        return lambda text: None

    def emit(text: str) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"{text}\n")
        except OSError:
            pass

    return emit


# ----------------------------------------------------- orchestration


def run_talk(cfg_path, stt: str | None = None, tts: str | None = None,
             voice_fx=None, notify_fn=None, emit=None) -> int:
    """One ``ana talk`` invocation: toggle, record, turn, notify.

    Returns 0 on every normal path (fresh round, stop request, answering
    no-op, empty transcript) and 1 on a voice failure. ``voice_fx`` and
    ``notify_fn`` are the test seams; the defaults are the live
    ``_DefaultVoice`` (T18 media pause, voxtype STT, Kokoro TTS via
    ``build_tts``) and ``notify``.
    """
    notify_fn = notify_fn or notify
    state = read_state()
    if state is not None:
        phase = state["phase"]
        if phase == "recording":
            # second press: signal the live worker, never block
            talk_stop_path().touch()
            notify_fn("Ana", "cortando…")
            return 0
        # phase "answering": the worker is past the mic (transcribing or
        # running the turn) — a stop-file here would corrupt its round
        # AND the next one. No-op instead.
        notify_fn("Ana", "ya estoy respondiendo")
        return 0

    # fresh session (no state, or a stale dead-pid one)
    runtime = _runtime_dir()
    runtime.mkdir(parents=True, exist_ok=True)
    talk_stop_path().unlink(missing_ok=True)  # stale stop from a crash
    vfx = voice_fx if voice_fx is not None else _DefaultVoice(stt, tts)
    emit_fn = emit or _make_emit()

    try:
        write_state("recording")
        notify_fn("Ana", "🎙 grabando… (otra vez el atajo para cortar)")
        waifu_send({"type": "say", "text": "🎙 te escucho…"})
        try:
            wav = vfx.record(default_max_seconds(),
                             make_press(talk_stop_path()))
        except VoiceError as exc:
            notify_fn("Ana", f"(voice unavailable: {exc})")
            return 1
        # past the mic: further hotkey presses must no-op from here on
        write_state("answering")
        try:
            heard = (vfx.transcribe(wav) or "").strip()
        except VoiceError as exc:
            notify_fn("Ana", f"(voice unavailable: {exc})")
            return 1
        if not heard:
            notify_fn("Ana", "(no escuché nada)")
            return 0
        # ONE turn engine: tools, provider config, error rollback,
        # the spoken reply and the waifu say/talk/idle events all ride
        # run_repl (T34) — talk.py never forks its own turn path.
        history = run_repl(cfg_path, lines=[heard, "/exit"], voice=True,
                           voice_fx=vfx, emit=emit_fn,
                           close_overlay=False)
        if history and history[-1].get("role") == "assistant":
            notify_fn("Ana", str(history[-1].get("content", ""))[:120])
        return 0
    finally:
        clear_state()
