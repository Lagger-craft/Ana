"""T35 / S15(c): headless voice hotkey `ana talk`.

Hyprland keybindings fire on key press only, so the round is a
two-press toggle built on session state under $XDG_RUNTIME_DIR/ana/
(same dir as the waifu FIFO — tests/conftest.py re-points fifo_path
globally, which redirects the talk state beside it too):

- press 1: no state → create talk.json (pid + phase "recording") and
  record until the stop-file appears or max seconds elapse;
- press 2: state phase "recording" → touch talk.stop only, never block;
- press 3 (worker past the mic): phase "answering" → no-op, and never
  create a stop-file that would corrupt the next round.

A dead pid in talk.json means the worker crashed: state is treated as
absent and a fresh round starts.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ana.cli import main
from ana.config import write_template
from ana.voice import VoiceError
from ana.talk import (make_press, notify, read_state, run_talk,
                      talk_state_path, talk_stop_path, write_state)

import ana.talk as talk_mod


@pytest.fixture
def cfg(tmp_path) -> object:
    path = tmp_path / "config.toml"
    write_template(path)
    return path


@pytest.fixture
def notifies():
    """(title, body) tuples captured by the injected notify seam."""
    return []


def _run(cfg, voice_fx, notifies, **kw):
    """run_talk with notify captured; returns the exit code."""
    return run_talk(
        cfg, voice_fx=voice_fx,
        notify_fn=lambda title, body: notifies.append((title, body)), **kw)


class FakeTalkVoice:
    """Orchestration seam with _DefaultVoice's record/transcribe/speak.

    ``record`` captures the session state as it was WHILE recording —
    the toggle tests assert the worker wrote phase "recording" before
    touching the mic.
    """

    def __init__(self, transcript="hola", record_error=None,
                 transcribe_error=None):
        self.transcript = transcript
        self.record_error = record_error
        self.transcribe_error = transcribe_error
        self.recorded = []
        self.state_during_record = None
        self.spoken = []

    def record(self, max_seconds, press):
        if self.record_error is not None:
            raise self.record_error
        self.recorded.append(max_seconds)
        self.state_during_record = read_state()
        return Path("/fake/mic.wav")

    def transcribe(self, wav):
        if self.transcribe_error is not None:
            raise self.transcribe_error
        return self.transcript

    def speak(self, text):
        self.spoken.append(text)


def _fake_repl(seen, reply="ok!"):
    def fake(cfg_path, **kw):
        seen.update(kw)
        seen["cfg"] = cfg_path
        return [{"role": "assistant", "content": reply}]
    return fake


# ------------------------------------------------------- press seam


class TestPressSeam:
    def test_first_press_starts_unbounded(self, tmp_path):
        press = make_press(tmp_path / "talk.stop")
        assert press() is True  # start: no timeout asked for

    def test_stop_file_present_stops_before_timeout(self, tmp_path):
        stop = tmp_path / "talk.stop"
        stop.touch()
        press = make_press(stop)
        assert press() is True
        assert press(timeout=0.3) is True  # stop-file beats the timeout

    def test_stop_file_appearing_mid_wait_stops(self, tmp_path):
        stop = tmp_path / "talk.stop"
        press = make_press(stop, poll=0.05)
        threading.Timer(0.05, stop.touch).start()
        press()
        started = time.monotonic()
        assert press(timeout=0.3) is True
        assert time.monotonic() - started < 0.3  # polled, not slept out

    def test_timeout_without_stop_file_returns_false(self, tmp_path):
        press = make_press(tmp_path / "talk.stop")
        press()
        started = time.monotonic()
        assert press(timeout=0.15) is False  # auto-stop at max seconds
        assert 0.1 <= time.monotonic() - started < 0.5


# ------------------------------------------------------------ state


class TestState:
    def test_roundtrip_pid_and_phase(self):
        write_state("recording")
        try:
            state = read_state()
            assert state["pid"] == os.getpid()
            assert state["phase"] == "recording"
        finally:
            talk_state_path().unlink(missing_ok=True)

    def test_missing_state_is_absent(self):
        assert read_state() is None

    def test_corrupt_json_is_absent(self):
        talk_state_path().write_text("{not json", encoding="utf-8")
        assert read_state() is None

    def test_unknown_phase_is_absent(self):
        talk_state_path().write_text(
            json.dumps({"pid": os.getpid(), "phase": "sleeping"}),
            encoding="utf-8")
        assert read_state() is None

    def test_dead_pid_is_stale(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        talk_state_path().write_text(
            json.dumps({"pid": proc.pid, "phase": "recording"}),
            encoding="utf-8")
        assert read_state() is None  # crashed worker: start fresh


# ----------------------------------------------------------- toggle


class TestToggle:
    def test_first_invocation_records_with_state(self, cfg, monkeypatch,
                                                 notifies):
        monkeypatch.delenv("ANA_VOICE_MAX_SECONDS", raising=False)
        voice = FakeTalkVoice(transcript="hola")
        seen = {}
        sent = []
        monkeypatch.setattr(talk_mod, "run_repl", _fake_repl(seen))
        monkeypatch.setattr(talk_mod, "waifu_send", sent.append)

        assert _run(cfg, voice, notifies) == 0

        # state existed (and said recording) while the mic was open
        assert voice.state_during_record is not None
        assert voice.state_during_record["phase"] == "recording"
        assert voice.recorded == [12.0]  # default_max_seconds()
        assert any("grabando" in body for _, body in notifies)
        assert {"type": "say", "text": "🎙 te escucho…"} in sent
        # finally cleaned every session file
        assert not talk_state_path().exists()
        assert not talk_stop_path().exists()

    def test_second_invocation_touches_stop_file_and_returns(
            self, cfg, monkeypatch, notifies):
        write_state("recording")  # a live worker (this test process)
        voice = FakeTalkVoice()

        def boom(*a, **kw):
            raise AssertionError("second press must never record")

        monkeypatch.setattr(talk_mod, "_DefaultVoice", boom)
        assert _run(cfg, voice, notifies) == 0

        assert voice.recorded == []
        assert talk_stop_path().exists()          # the stop signal
        assert talk_state_path().exists()         # the worker's state stays
        assert any("cortando" in body for _, body in notifies)

    def test_third_invocation_while_answering_no_ops(self, cfg, monkeypatch,
                                                     notifies):
        write_state("answering")  # worker is past the mic
        voice = FakeTalkVoice()
        monkeypatch.setattr(
            talk_mod, "run_repl",
            lambda *a, **kw: pytest.fail("answering worker owns the turn"))

        assert _run(cfg, voice, notifies) == 0

        assert voice.recorded == []
        assert not talk_stop_path().exists()  # would corrupt the next round
        assert talk_state_path().exists()     # worker's state untouched
        assert any("ya estoy respondiendo" in body for _, body in notifies)

    def test_dead_pid_state_starts_a_fresh_round(self, cfg, monkeypatch,
                                                 notifies):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        talk_state_path().write_text(
            json.dumps({"pid": proc.pid, "phase": "recording"}),
            encoding="utf-8")
        talk_stop_path().touch()  # stale stop would kill a fresh round
        monkeypatch.delenv("ANA_VOICE_MAX_SECONDS", raising=False)
        voice = FakeTalkVoice(transcript="hola")
        seen = {}
        monkeypatch.setattr(talk_mod, "run_repl", _fake_repl(seen))

        assert _run(cfg, voice, notifies) == 0

        assert voice.recorded == [12.0]  # a fresh recording actually ran
        assert voice.state_during_record["pid"] == os.getpid()
        assert not talk_state_path().exists()
        assert not talk_stop_path().exists()


# ----------------------------------------------------- orchestration


class TestOrchestration:
    def test_empty_transcript_skips_turn_and_notifies(self, cfg, monkeypatch,
                                                     notifies):
        voice = FakeTalkVoice(transcript="")
        monkeypatch.setattr(
            talk_mod, "run_repl",
            lambda *a, **kw: pytest.fail("no turn without a transcript"))

        assert _run(cfg, voice, notifies) == 0

        assert any("no escuché" in body for _, body in notifies)
        assert not talk_state_path().exists()
        assert not talk_stop_path().exists()

    def test_transcript_feeds_run_repl_and_notifies_reply(self, cfg,
                                                          monkeypatch,
                                                          notifies):
        reply = "respuesta larga " + "x" * 200
        voice = FakeTalkVoice(transcript="hola")
        seen = {}
        monkeypatch.setattr(talk_mod, "run_repl", _fake_repl(seen, reply))

        assert _run(cfg, voice, notifies) == 0

        assert seen["cfg"] == cfg
        assert seen["lines"] == ["hola", "/exit"]
        assert seen["voice"] is True
        assert seen["voice_fx"] is voice
        assert callable(seen["emit"])
        # reply notification caps at ~120 chars
        assert any(body == reply[:120] for _, body in notifies)
        assert not talk_state_path().exists()
        assert not talk_stop_path().exists()

    def test_round_keeps_the_overlay_open(self, cfg, monkeypatch, notifies):
        # `ana talk` has no chat window to close (Option B does not
        # apply): the hotkey wrapper keeps the overlay up across rounds,
        # so the round must not end by sending the overlay quit event.
        voice = FakeTalkVoice(transcript="hola")
        seen = {}
        monkeypatch.setattr(talk_mod, "run_repl", _fake_repl(seen))

        assert _run(cfg, voice, notifies) == 0

        assert seen["close_overlay"] is False

    def test_record_voice_error_notifies_and_cleans_up(self, cfg, monkeypatch,
                                                       notifies):
        voice = FakeTalkVoice(record_error=VoiceError("arecord missing"))
        monkeypatch.setattr(
            talk_mod, "run_repl",
            lambda *a, **kw: pytest.fail("no turn after a record error"))

        assert _run(cfg, voice, notifies) == 1

        assert any("(voice unavailable: arecord missing)" in body
                   for _, body in notifies)
        assert not talk_state_path().exists()
        assert not talk_stop_path().exists()

    def test_transcribe_voice_error_notifies_and_cleans_up(self, cfg,
                                                           monkeypatch,
                                                           notifies):
        voice = FakeTalkVoice(transcribe_error=VoiceError("voxtype gone"))
        monkeypatch.setattr(
            talk_mod, "run_repl",
            lambda *a, **kw: pytest.fail("no turn after a transcribe error"))

        assert _run(cfg, voice, notifies) == 1

        assert any("(voice unavailable: voxtype gone)" in body
                   for _, body in notifies)
        assert not talk_state_path().exists()
        assert not talk_stop_path().exists()

    def test_stop_file_during_round_is_cleaned_in_finally(self, cfg,
                                                          monkeypatch,
                                                          notifies):
        class StopDuringRecord(FakeTalkVoice):
            def record(self, max_seconds, press):
                talk_stop_path().touch()  # hotkey pressed mid-recording
                return super().record(max_seconds, press)

        monkeypatch.setattr(talk_mod, "run_repl", _fake_repl({}))

        assert _run(cfg, StopDuringRecord(transcript="hola"), notifies) == 0

        assert not talk_stop_path().exists()
        assert not talk_state_path().exists()

    def test_default_emit_discards_without_a_tty(self, cfg, monkeypatch,
                                                 notifies):
        voice = FakeTalkVoice(transcript="hola")
        seen = {}
        monkeypatch.setattr(talk_mod, "run_repl", _fake_repl(seen))

        assert _run(cfg, voice, notifies) == 0  # no emit kwarg given

        seen["emit"]("streamed chunk")  # must not print, must not raise
        monkeypatch.delenv("ANA_TALK_DEBUG", raising=False)

    def test_ana_talk_debug_appends_stream_to_file(self, cfg, monkeypatch,
                                                   notifies, tmp_path):
        log = tmp_path / "talk.log"
        monkeypatch.setenv("ANA_TALK_DEBUG", str(log))
        voice = FakeTalkVoice(transcript="hola")
        seen = {}
        monkeypatch.setattr(talk_mod, "run_repl", _fake_repl(seen))

        assert _run(cfg, voice, notifies) == 0

        seen["emit"]("streamed chunk")
        assert log.read_text(encoding="utf-8") == "streamed chunk\n"


# ------------------------------------------------------------ notify


class TestNotify:
    @staticmethod
    def _patch_run(monkeypatch, side_effect=None, result=None):
        def fake_run(*a, **kw):
            if side_effect is not None:
                raise side_effect
            return result
        monkeypatch.setattr(talk_mod.subprocess, "run", fake_run)

    def test_missing_notify_send_never_raises(self, monkeypatch):
        self._patch_run(monkeypatch, side_effect=FileNotFoundError(
            "notify-send"))
        notify("Ana", "hola")  # no dbus / no binary in tests: silent

    def test_nonzero_exit_never_raises(self, monkeypatch):
        self._patch_run(monkeypatch,
                        result=SimpleNamespace(returncode=1, stdout="",
                                               stderr=""))
        notify("Ana", "hola")

    def test_timeout_never_raises(self, monkeypatch):
        self._patch_run(monkeypatch, side_effect=subprocess.TimeoutExpired(
            cmd="notify-send", timeout=5))
        notify("Ana", "hola")

    def test_sends_title_and_body(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kw):
            calls.append((cmd, kw))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        self._patch_run(monkeypatch, result=None)
        monkeypatch.setattr(talk_mod.subprocess, "run", fake_run)
        notify("Ana", "grabando…")
        cmd, kw = calls[0]
        assert cmd == ["notify-send", "Ana", "grabando…"]
        assert kw["timeout"] == 5


# -------------------------------------------------------- cli wiring


class TestTalkCommand:
    def test_dispatch_passes_config_and_stt(self, cfg, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            talk_mod, "run_talk",
            lambda path, stt=None: seen.update(cfg=path, stt=stt) or 0)
        assert main(["--config", str(cfg), "--stt", "whisper", "talk"]) == 0
        assert seen == {"cfg": str(cfg), "stt": "whisper"}

    def test_dispatch_defaults_stt_to_none(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            talk_mod, "run_talk",
            lambda path, stt=None: seen.update(cfg=path, stt=stt) or 0)
        assert main(["talk"]) == 0
        assert seen["stt"] is None

    def test_help_documents_the_hotkey(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        out = capsys.readouterr().out
        assert excinfo.value.code == 0
        assert "talk" in out
        assert "hotkey" in out.lower()
