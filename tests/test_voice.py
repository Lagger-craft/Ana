"""Tests for Phase 2 voice (S9/S10): STT, TTS, and the CLI voice loop.

Covers ana.voice.stt (WhisperSTT, record_to_wav), ana.voice.tts
(PiperTTS, resolve_voice) and the run_repl voice seam (/voice, --voice).
"""

import signal
import struct
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from ana.cli import main, run_repl
from ana.config import write_template
from ana.voice import PiperTTS, VoiceError, WhisperSTT, record_to_wav
from ana.voice.stt import default_max_seconds, stdin_press


@pytest.fixture
def cfg(tmp_path) -> object:
    path = tmp_path / "config.toml"
    write_template(path)
    return path


# ---------------------------------------------------------------- stt


def _write_wav(path: Path, nframes: int = 8, sample: int = 1000) -> Path:
    """Write a real 16 kHz mono S16_LE WAV — the record_to_wav format."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{nframes}h", *([sample] * nframes)))
    return path


class TestWhisperSTT:
    def test_transcribe_hands_float32_ndarray_to_faster_whisper(
            self, monkeypatch, tmp_path):
        seen = {}
        fake = types.ModuleType("faster_whisper")

        class FakeWhisperModel:
            def __init__(self, size, device=None, compute_type=None):
                seen["init"] = (size, device, compute_type)

            def transcribe(self, audio, language=None):
                seen["audio"] = audio
                seen["language"] = language
                return iter([("  hola  ", None)]), None

        fake.WhisperModel = FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)

        wav = _write_wav(tmp_path / "in.wav", nframes=3, sample=16384)
        stt = WhisperSTT()  # defaults: small / cpu / int8 / es
        assert stt.transcribe(wav) == "hola"
        assert seen["init"] == ("small", "cpu", "int8")
        assert seen["language"] == "es"
        audio = seen["audio"]
        assert isinstance(audio, np.ndarray)  # ndarray, never a file path
        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio, [0.5, 0.5, 0.5], atol=1e-6)

    def test_empty_wav_returns_empty_without_loading_model(
            self, monkeypatch, tmp_path):
        # no frames → "" and faster_whisper is never even imported
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        wav = _write_wav(tmp_path / "empty.wav", nframes=0)
        assert WhisperSTT().transcribe(wav) == ""

    def test_missing_numpy_raises_actionable_voice_error(
            self, monkeypatch, tmp_path):
        # None in sys.modules makes `import numpy` raise ImportError
        monkeypatch.setitem(sys.modules, "numpy", None)
        wav = _write_wav(tmp_path / "in.wav", nframes=1)
        stt = WhisperSTT()
        with pytest.raises(VoiceError) as excinfo:
            stt.transcribe(wav)
        msg = str(excinfo.value)
        assert "numpy required for audio decode" in msg
        assert "pip install numpy" in msg

    def test_missing_faster_whisper_raises_actionable_voice_error(
            self, monkeypatch, tmp_path):
        # None in sys.modules makes `import faster_whisper` raise ImportError
        # even if the package were installed
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        wav = _write_wav(tmp_path / "in.wav", nframes=1)
        stt = WhisperSTT()
        with pytest.raises(VoiceError) as excinfo:
            stt.transcribe(wav)
        msg = str(excinfo.value)
        assert "faster-whisper not installed" in msg
        assert ".venv/bin/pip install -e '.[voice]'" in msg

    def test_transcribe_path_never_touches_av(self, monkeypatch, tmp_path):
        # PyAV 19 removed av.open(metadata_errors=...); the transcribe path
        # must not import/call av at all (stdlib wave + numpy decode)
        monkeypatch.setitem(sys.modules, "av", None)  # any `import av` boom
        fake = types.ModuleType("faster_whisper")

        class FakeWhisperModel:
            def __init__(self, size, device=None, compute_type=None):
                pass

            def transcribe(self, audio, language=None):
                return iter([]), None

        fake.WhisperModel = FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)
        wav = _write_wav(tmp_path / "in.wav", nframes=1)
        assert WhisperSTT().transcribe(wav) == ""

    def test_module_imports_cleanly_without_faster_whisper(self, monkeypatch):
        # a fresh import of the module must succeed with the package absent
        monkeypatch.delitem(sys.modules, "ana.voice.stt", raising=False)
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        import importlib

        mod = importlib.import_module("ana.voice.stt")
        assert mod.WhisperSTT().model_size == "small"  # fresh module, no fw


class FakePress:
    """Injected click seam: scripted results, records each timeout."""

    def __init__(self, results=(True, True)):
        self._results = iter(results)
        self.calls = []

    def __call__(self, timeout=None):
        self.calls.append(timeout)
        return next(self._results)


class FakeProc:
    """Stand-in for the arecord Popen process (creates the WAV like it)."""

    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self.kw = kw
        self.signals = []
        self.wait_calls = []
        self.returncode = None
        Path(cmd[-1]).write_bytes(b"RIFF....WAVE")  # arecord creates it

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.returncode = -2  # stopped by SIGINT
        return self.returncode


class TestRecordToWav:
    @staticmethod
    def _patch_popen(monkeypatch):
        procs = []

        def fake_popen(cmd, **kw):
            proc = FakeProc(cmd, **kw)
            procs.append(proc)
            return proc

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        return procs

    def test_push_to_talk_starts_arecord_on_first_press(
            self, monkeypatch, tmp_path):
        procs = self._patch_popen(monkeypatch)
        press = FakePress(results=(True, True))
        out = tmp_path / "in.wav"

        result = record_to_wav(out, press=press)

        assert result == out
        cmd = procs[0].cmd
        assert cmd == ["arecord", "-f", "S16_LE", "-r", "16000",
                       "-q", str(out)]
        assert "-d" not in cmd  # duration comes from the presses now
        assert press.calls == [None, 12.0]  # 1st press unbounded, 2nd timed
        assert procs[0].signals == [signal.SIGINT]
        assert procs[0].wait_calls == [10]
        assert out.exists()

    def test_second_press_or_max_seconds_stops_arecord(
            self, monkeypatch, tmp_path):
        # both ways to end the recording: 2nd press (True) or timeout (False)
        for stopped in (True, False):
            procs = self._patch_popen(monkeypatch)
            press = FakePress(results=(True, stopped))
            out = tmp_path / "x.wav"

            record_to_wav(out, max_seconds=0.5, press=press)

            assert press.calls == [None, 0.5]
            assert procs[0].signals == [signal.SIGINT]
            assert procs[0].wait_calls == [10]  # arecord reaped after SIGINT

    def test_max_seconds_defaults_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANA_VOICE_MAX_SECONDS", "2.5")
        self._patch_popen(monkeypatch)
        press = FakePress(results=(True, False))

        record_to_wav(tmp_path / "in.wav", press=press)

        assert press.calls == [None, 2.5]

    def test_max_seconds_defaults_to_12(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANA_VOICE_MAX_SECONDS", raising=False)
        self._patch_popen(monkeypatch)
        press = FakePress(results=(True, False))

        record_to_wav(tmp_path / "in.wav", press=press)

        assert press.calls == [None, 12.0]

    def test_press_failure_before_start_leaves_no_arecord(
            self, monkeypatch, tmp_path):
        procs = self._patch_popen(monkeypatch)

        def press(timeout=None):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            record_to_wav(tmp_path / "in.wav", press=press)
        assert procs == []  # no recording started without the first press

    def test_arecord_failure_raises_voice_error(self, monkeypatch, tmp_path):
        # arecord exits without producing a WAV (e.g. no capture device)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

        class DeadProc(FakeProc):
            def __init__(self, cmd, **kw):
                super().__init__(cmd, **kw)
                Path(cmd[-1]).unlink(missing_ok=True)  # no WAV produced

            def wait(self, timeout=None):
                self.wait_calls.append(timeout)
                self.returncode = 1
                return 1

        monkeypatch.setattr(
            "subprocess.Popen", lambda cmd, **kw: DeadProc(cmd, **kw))
        press = FakePress(results=(True, True))

        with pytest.raises(VoiceError) as excinfo:
            record_to_wav(tmp_path / "in.wav", press=press)
        msg = str(excinfo.value)
        assert "recording failed" in msg
        assert "arecord exit 1" in msg

    def test_missing_arecord_raises_voice_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        press = FakePress(results=())
        with pytest.raises(VoiceError) as excinfo:
            record_to_wav("/tmp/x.wav", press=press)
        msg = str(excinfo.value)
        assert "alsa-utils" in msg
        assert "pw-record" in msg
        assert press.calls == []  # fails before waiting for a press

    def test_invalid_env_max_seconds_raises_voice_error(self, monkeypatch):
        monkeypatch.setenv("ANA_VOICE_MAX_SECONDS", "soon")
        with pytest.raises(VoiceError) as excinfo:
            default_max_seconds()
        assert "ANA_VOICE_MAX_SECONDS" in str(excinfo.value)


class TestStdinPress:
    def test_blocking_press_reads_enter(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda: "")
        assert stdin_press() is True

    def test_blocking_press_eof_becomes_voice_error(self, monkeypatch):
        def eof(_prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", eof)
        with pytest.raises(VoiceError) as excinfo:
            stdin_press()
        assert "input exhausted" in str(excinfo.value)

    def test_timed_press_returns_false_when_nothing_comes(self, monkeypatch):
        import ana.voice.stt as stt

        monkeypatch.setattr(stt.select, "select", lambda *a: ([], [], []))
        assert stdin_press(timeout=1.0) is False

    def test_timed_press_reads_enter_when_ready(self, monkeypatch):
        import ana.voice.stt as stt

        monkeypatch.setattr(stt.select, "select", lambda *a: (["x"], [], []))
        monkeypatch.setattr("builtins.input", lambda: "")
        assert stdin_press(timeout=1.0) is True


# ---------------------------------------------------------------- tts


class TestPiperTTS:
    def test_speak_synth_then_play_and_remove_tmp(self, monkeypatch):
        import ana.voice.tts as tts

        calls = []
        monkeypatch.setattr(
            tts, "_run", lambda cmd, **kw: calls.append((cmd, kw)) or None)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

        PiperTTS(voice="/models/es_ES-carmen.onnx").speak("hola ana")

        assert len(calls) == 2
        piper_cmd = calls[0][0]
        assert piper_cmd[0] == "/usr/bin/piper"
        assert piper_cmd[1:3] == ["--model", "/models/es_ES-carmen.onnx"]
        assert piper_cmd[3] == "--output_file"
        wav = piper_cmd[4]
        assert not Path(wav).exists()  # tmp wav gone after playback
        assert calls[1][0] == ["/usr/bin/paplay", wav]

    def test_missing_piper_raises_actionable_voice_error(self, monkeypatch,
                                                         tmp_path):
        monkeypatch.setattr("shutil.which", lambda name: None)
        # also hide the venv bin fallback: pretend we run from nowhere
        monkeypatch.setattr("ana.voice.tts.sys", _no_exec_sys(tmp_path))
        with pytest.raises(VoiceError) as excinfo:
            PiperTTS(voice="/models/x.onnx").speak("hola")
        msg = str(excinfo.value)
        assert "piper not installed" in msg
        assert "es_MX-danepar" in msg and "es_ES-carmen" in msg
        assert "pip install piper-tts" in msg

    def test_ana_piper_model_env_override_respected(self, monkeypatch):
        import ana.voice.tts as tts

        calls = []
        monkeypatch.setattr(tts, "_run", lambda cmd, **kw: calls.append(cmd))
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setenv("ANA_PIPER_MODEL", "/env/spanish-voice.onnx")

        PiperTTS().speak("hola")  # no explicit voice -> env wins

        assert calls[0][1:3] == ["--model", "/env/spanish-voice.onnx"]

    def test_default_voice_globs_es_onnx_under_local_share(
            self, monkeypatch, tmp_path):
        import ana.voice.tts as tts

        monkeypatch.delenv("ANA_PIPER_MODEL", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))  # Path.home() reads HOME
        voice_dir = tmp_path / ".local" / "share" / "piper" / "es_ES-carmen"
        voice_dir.mkdir(parents=True)
        model = voice_dir / "es_ES-carmen.onnx"
        model.write_bytes(b"")
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        calls = []
        monkeypatch.setattr(tts, "_run", lambda cmd, **kw: calls.append(cmd))

        PiperTTS().speak("hola")

        assert calls[0][1:3] == ["--model", str(model)]

    def test_no_voice_found_raises_with_location_and_download_hint(
            self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANA_PIPER_MODEL", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))  # empty fake home
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        with pytest.raises(VoiceError) as excinfo:
            PiperTTS().speak("hola")
        msg = str(excinfo.value)
        assert ".local/share/piper" in msg
        assert "ANA_PIPER_MODEL" in msg


# ------------------------------------------------------------ cli loop


class FakeVoice:
    """Injectable voice seam for run_repl (record/transcribe/speak).

    ``record`` plays the push-to-talk round: it drives the injected
    press seam exactly like record_to_wav does (first press starts,
    second press or timeout stops), so scripted input lines line up.
    """

    def __init__(self, transcript="hola ana", record_error=None,
                 speak_error=None):
        self.transcript = transcript
        self.record_error = record_error
        self.speak_error = speak_error
        self.recorded = []
        self.press_results = []
        self.spoken = []

    def record(self, max_seconds, press):
        if self.record_error is not None:
            raise self.record_error
        started = press()  # first press: start recording
        stopped = press(timeout=max_seconds)  # 2nd press or auto-stop
        self.recorded.append(max_seconds)
        self.press_results.append((started, stopped))
        return Path("/fake/mic.wav")

    def transcribe(self, wav):
        return self.transcript

    def speak(self, text):
        if self.speak_error is not None:
            raise self.speak_error
        self.spoken.append(text)


def stream_ok(reply="hola!"):
    return lambda msgs, model, system=None: iter([reply])


class TestVoiceLoop:
    def test_voice_round_records_transcribes_and_speaks_reply(
            self, cfg, capsys):
        fx = FakeVoice(transcript="di hola")
        history = run_repl(
            cfg_path=cfg, lines=["", "", "", "/exit"],
            stream=stream_ok("ok!"), voice=True, voice_fx=fx)
        out = capsys.readouterr().out
        assert ("(🎙 Enter para empezar, Enter para terminar (max 12s))"
                in out)
        assert fx.recorded == [12.0]  # default max seconds
        assert fx.press_results == [(True, True)]  # two-press round
        assert history[0] == {"role": "user", "content": "di hola"}
        assert history[-1] == {"role": "assistant", "content": "ok!"}
        assert fx.spoken == ["ok!"]
        assert "di hola" in out  # transcript echoed

    def test_voice_round_auto_stops_without_second_press(self, cfg, capsys):
        # scripted input runs out after the first press → auto-stop at max
        fx = FakeVoice(transcript="hola")
        run_repl(cfg_path=cfg, lines=["", ""],
                 stream=stream_ok(), voice=True, voice_fx=fx)
        out = capsys.readouterr().out
        assert fx.press_results == [(True, False)]  # 2nd press never came
        assert fx.recorded == [12.0]
        assert fx.spoken == ["hola!"]  # round still completed

    def test_voice_round_max_seconds_from_env(self, cfg, monkeypatch,
                                              capsys):
        monkeypatch.setenv("ANA_VOICE_MAX_SECONDS", "2.5")
        fx = FakeVoice()
        run_repl(cfg_path=cfg, lines=["", "", "", "/exit"],
                 stream=stream_ok(), voice=True, voice_fx=fx)
        out = capsys.readouterr().out
        assert "(max 2.5s)" in out
        assert fx.recorded == [2.5]

    def test_silent_transcript_emits_no_escuche_nada(self, cfg, capsys):
        fx = FakeVoice(transcript="")
        history = run_repl(
            cfg_path=cfg, lines=["", "", "", "/exit"],
            stream=stream_ok(), voice=True, voice_fx=fx)
        out = capsys.readouterr().out
        assert "(no escuché nada)" in out
        assert "heard nothing" not in out
        assert history == []  # no turn ran; back at the prompt
        assert fx.spoken == []

    def test_typed_turn_spoken_when_voice_on(self, cfg):
        fx = FakeVoice()
        run_repl(cfg_path=cfg, lines=["hola", "/exit"],
                 stream=stream_ok(), voice=True, voice_fx=fx)
        assert fx.spoken == ["hola!"]
        assert fx.recorded == []  # typed input skips the mic

    def test_voice_off_never_records_or_speaks(self, cfg):
        fx = FakeVoice()
        run_repl(cfg_path=cfg, lines=["", "hola", "/exit"],
                 stream=stream_ok(), voice=False, voice_fx=fx)
        assert fx.recorded == []
        assert fx.spoken == []

    def test_voice_command_toggles_and_reports_status(self, cfg, capsys):
        fx = FakeVoice()
        run_repl(cfg_path=cfg,
                 lines=["/voice status", "/voice on", "hola",
                        "/voice off", "/exit"],
                 stream=stream_ok(), voice=False, voice_fx=fx)
        out = capsys.readouterr().out
        assert "voice: off" in out
        assert "voice on" in out
        assert "voice off" in out
        assert fx.spoken == ["hola!"]  # spoken only while on

    def test_record_failure_degrades_to_text_mode(self, cfg, capsys):
        fx = FakeVoice(record_error=VoiceError(
            "arecord not found — install alsa-utils"))
        history = run_repl(
            cfg_path=cfg, lines=["", "hola", "/exit"], stream=stream_ok(),
            voice=True, voice_fx=fx)
        out = capsys.readouterr().out
        assert "(voice unavailable: arecord not found" in out
        # REPL survived; the typed turn still ran normally
        assert history[0] == {"role": "user", "content": "hola"}
        assert history[-1]["content"] == "hola!"

    def test_speak_failure_does_not_crash_repl(self, cfg, capsys):
        fx = FakeVoice(speak_error=VoiceError(
            "piper not installed — install piper"))
        history = run_repl(cfg_path=cfg, lines=["hola", "/exit"],
                           stream=stream_ok(), voice=True, voice_fx=fx)
        out = capsys.readouterr().out
        assert "(voice unavailable: piper not installed" in out
        assert history[-1] == {"role": "assistant", "content": "hola!"}


class TestMainVoiceFlag:
    def test_voice_flag_passed_to_repl(self, cfg, monkeypatch):
        import ana.cli as cli

        seen = {}

        def fake_run_repl(cfg_path, **kw):
            seen.update(kw)
            return []

        monkeypatch.setattr(cli, "run_repl", fake_run_repl)
        assert main(["--config", str(cfg), "--voice"]) == 0
        assert seen["voice"] is True

    def test_voice_flag_defaults_off(self, cfg, monkeypatch):
        import ana.cli as cli

        seen = {}
        monkeypatch.setattr(
            cli, "run_repl",
            lambda cfg_path, **kw: seen.update(kw) or [])
        assert main(["--config", str(cfg)]) == 0
        assert seen["voice"] is False


def _ok_proc():
    import subprocess
    return subprocess.CompletedProcess([], 0, stdout="", stderr="")


def _no_exec_sys(tmp_path):
    """sys stub whose executable lives in an empty dir (no venv piper)."""
    import types
    stub = types.SimpleNamespace()
    stub.executable = str(tmp_path / "nowhere" / "python")
    return stub


def test_piper_binary_falls_back_to_venv_bin(monkeypatch, tmp_path):
    """piper-tts installs into .venv/bin; PATH lookup may miss it."""
    import ana.voice.tts as tts_mod
    fake_piper = tmp_path / "piper"
    fake_piper.write_text("#!/bin/sh\n")
    fake_piper.chmod(0o755)
    monkeypatch.setattr(tts_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(tts_mod.subprocess, "run", lambda *a, **k: _ok_proc())
    monkeypatch.setattr(tts_mod.sys, "executable", str(tmp_path / "python"))
    tts_inst = tts_mod.PiperTTS(voice=str(tmp_path / "voice.onnx"))  # noqa: F841 (used below)
    assert tts_mod._piper_binary() == str(fake_piper)
