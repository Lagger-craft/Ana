"""Tests for the MPRIS media pause/resume helper (voxtype-parity).

While Ana records push-to-talk audio, playing media (browser videos,
games) dominates the microphone feed: whisper transcribes the media
instead of the user. voxtype solves it with pause_media=true — it pauses
MPRIS players for the recording and resumes them after. MediaPlayer
pause/resume goes over the session bus with dbus-send (no playerctl
dep).
"""

import subprocess
from types import SimpleNamespace

from ana.voice.media import list_mpris_players, pause_media, resume_media

import ana.voice.media as media


class _DBus:
    """Scriptable dbus-send stand-in for the _run seam."""

    def __init__(self):
        self.names = ["org.mpris.MediaPlayer2.firefox.instance_1_246",
                      "org.freedesktop.DBus"]
        self.status = {self.names[0]: "Playing"}
        self.calls = []  # (dest, method)
        self.fail = False

    def __call__(self, cmd, **kw):
        if self.fail:
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        dest = next(c[len("--dest="):] for c in cmd
                    if c.startswith("--dest="))
        method = cmd[-1].split(".")[-1].removeprefix("string:")
        self.calls.append((dest, method))
        if method == "ListNames":
            out = "".join(f'      string "{n}"\n' for n in self.names)
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        if method == "PlaybackStatus":
            status = self.status.get(dest, "Stopped")
            return SimpleNamespace(returncode=0,
                                   stdout=f'   string "{status}"\n',
                                   stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _patch(monkeypatch, fake):
    monkeypatch.setattr(media, "_run", fake)


class TestListMprisPlayers:
    def test_only_mpris_names(
            self, monkeypatch):
        fake = _DBus()
        _patch(monkeypatch, fake)
        names = list_mpris_players()
        assert names == ["org.mpris.MediaPlayer2.firefox.instance_1_246"]

    def test_dbus_failure_yields_empty_no_error(self, monkeypatch):
        fake = _DBus()
        fake.fail = True
        _patch(monkeypatch, fake)
        assert list_mpris_players() == []


class TestPauseMedia:
    def test_pauses_only_currently_playing_players(self, monkeypatch):
        fake = _DBus()
        fake.status[fake.names[0]] = "Playing"
        _patch(monkeypatch, fake)

        paused = pause_media()

        assert paused == [fake.names[0]]
        assert fake.calls.count((fake.names[0], "Pause")) == 1
        for dest, method in fake.calls:
            assert method in ("ListNames", "PlaybackStatus", "Pause")

    def test_nothing_playing_pauses_nothing(self, monkeypatch):
        fake = _DBus()
        fake.status[fake.names[0]] = "Stopped"
        _patch(monkeypatch, fake)
        assert pause_media() == []
        # Pause is never called (Status is Stopped for everyone).
        assert not any(m == "Pause" for _, m in fake.calls)


class TestResumeMedia:
    def test_resumes_exactly_the_players_that_were_paused(self, monkeypatch):
        fake = _DBus()
        _patch(monkeypatch, fake)
        resume_media(["org.mpris.MediaPlayer2.firefox.instance_1_246"])
        assert fake.calls.count(
            ("org.mpris.MediaPlayer2.firefox.instance_1_246", "Play")) == 1

    def test_empty_list_is_a_noop(self, monkeypatch):
        fake = _DBus()
        _patch(monkeypatch, fake)
        resume_media([])
        assert fake.calls == []
