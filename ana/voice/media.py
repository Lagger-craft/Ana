"""MPRIS media pause/resume around push-to-talk recording.

While a video (browser, game launcher) plays, the microphone feed is
dominated by the content: Ana recorded its dialogue instead of the
user's voice. voxtype — which the user dictates with — solves this with
``pause_media = true``: it pauses MPRIS players during capture and
resumes them afterwards. We do the same over the session D-Bus using
``dbus-send`` (present on any desktop with D-Bus; no playerctl
dependency).

Public names:
- pause_media(players=None): pause the currently "Playing" MPRIS
  players and return the exact list that was paused, so it can be
  handed back to resume_media.
- resume_media(players): resume those players (org.mpris.MediaPlayer2
  Play), ignoring any that disappeared in between.
- list_mpris_players: MPRIS bus names, empty list on any D-Bus failure.
"""

from __future__ import annotations

import re

from .errors import VoiceError

__all__ = ["list_mpris_players", "pause_media", "resume_media"]

_PLAYER_RE = re.compile(r'string "(:?org\.mpris\.MediaPlayer2\.[^"]*)"')

_STATUS_PATH = "/org/mpris/MediaPlayer2"
_STATUS_CMD = (
    "org.freedesktop.DBus.Properties.Get",
    "string:org.mpris.MediaPlayer2.Player",
    "string:PlaybackStatus",
)


def _run(cmd, **kw):
    """Run a subprocess. Module-level seam so tests can monkeypatch it."""
    return __import__("subprocess").run(cmd, **kw)


def _dbus_send(*args):
    return _run(
        ["dbus-send", "--session", "--print-reply", "--dest=" + args[0],
         *args[1:]], capture_output=True, text=True)


def list_mpris_players() -> list[str]:
    """Return the MPRIS bus names; [] when D-Bus is unavailable."""
    try:
        # dbus-send wants dotted interface.member: a bare member comes
        # back as "Data item ... is badly formed".
        proc = _dbus_send("org.freedesktop.DBus", "/org/freedesktop/DBus",
                          "org.freedesktop.DBus.ListNames")
        return _PLAYER_RE.findall(proc.stdout or "")
    except (OSError, AttributeError):
        # [] with no error: media pause is a nicety, never the feature
        return []


def _playback_status(name: str) -> str:
    proc = _dbus_send(name, _STATUS_CMD[0], *_STATUS_CMD[1:])
    match = re.search(r'string "(\w+)"', proc.stdout or "")
    return match.group(1) if match else "Stopped"


def pause_media() -> list[str]:
    """Pause MPRIS players that are Playing now; return their names.

    The returned list is the exact pause set for resume_media. Failures
    degrade silently: no player means nothing to mute.
    """
    paused = []
    for name in list_mpris_players():
        if _playback_status(name) == "Playing":
            _dbus_send(name, "/org/mpris/MediaPlayer2",
                       "org.mpris.MediaPlayer2.Player.Pause")
            paused.append(name)
    return paused


def resume_media(players: list[str]) -> None:
    """Resume exactly the players pause_media returned, ignoring exits."""
    for name in players:
        _dbus_send(name, "/org/mpris/MediaPlayer2",
                   "org.mpris.MediaPlayer2.Player.Play")
