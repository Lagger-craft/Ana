"""Tests for Phase 2 voice (S9/S10): STT, TTS, and the CLI voice loop.

Covers ana.voice.stt (WhisperSTT, record_to_wav), ana.voice.tts
(PiperTTS, resolve_voice) and the run_repl voice seam (/voice, --voice).
"""

import os
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
from ana.voice.stt import (VoxtypeSTT, build_stt, capture_source,
                           default_max_seconds, mic_label, peak_db,
                           stdin_press)


class TestCliSttFlag:
    def test_flag_reaches_run_repl(self, monkeypatch):
        import ana.cli as cli

        seen = {}

        def fake_repl(cfg_path, **kw):
            seen.update(kw)
            return []

        monkeypatch.setattr(cli, "run_repl", fake_repl)
        assert main(["--voice", "--stt", "whisper"]) == 0
        assert seen["stt"] == "whisper"
        assert seen["voice"] is True


class TestMediaPauseAroundRecording:
    """_DefaultVoice.record pauses MPRIS players while capturing.

    Playing media dominates the mic feed and gets transcribed instead
    of the user (voxtype already does this: pause_media = true).
    """

    def _patch_voice(self, monkeypatch, fail=False):
        import ana.cli as cli
        import ana.voice.media as media_mod
        import ana.voice.stt as stt_mod

        events = []

        def fake_pause():
            events.append("pause")
            return ["org.mpris.MediaPlayer2.firefox"]

        def fake_resume(players):
            events.append(("resume", tuple(players)))

        def fake_record(path, max_seconds=None, press=None):
            events.append("record")
            if fail:
                raise VoiceError("boom")
            return path

        monkeypatch.setattr(media_mod, "pause_media", fake_pause)
        monkeypatch.setattr(media_mod, "resume_media", fake_resume)
        monkeypatch.setattr(stt_mod, "record_to_wav", fake_record)
        voice = cli._DefaultVoice()
        return voice, events

    def test_pause_record_resume_order(self, monkeypatch):
        voice, events = self._patch_voice(monkeypatch)
        voice.record(5.0, press=lambda timeout=None: True)
        assert events == ["pause", "record",
                          ("resume", ("org.mpris.MediaPlayer2.firefox",))]

    def test_resume_even_when_recording_fails(self, monkeypatch):
        voice, events = self._patch_voice(monkeypatch, fail=True)
        with pytest.raises(VoiceError):
            voice.record(5.0, press=lambda timeout=None: True)
        assert events == ["pause", "record",
                          ("resume", ("org.mpris.MediaPlayer2.firefox",))]

    def test_opt_out_env_skips_pause(self, monkeypatch):
        monkeypatch.setenv("ANA_VOICE_PAUSE_MEDIA", "0")
        voice, events = self._patch_voice(monkeypatch)
        voice.record(5.0, press=lambda timeout=None: True)
        assert events == ["record"]


class _Proc:
    """Minimal CompletedProcess stand-in for the voxtype _run seam."""

    def __init__(self, stdout="", stderr="", rc=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = rc


def _patch_run(monkeypatch, proc):
    import ana.voice.stt as stt
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return proc

    monkeypatch.setattr(stt, "_run", fake_run)
    return seen


def _which_missing(monkeypatch, names="voxtype"):
    import shutil as _shutil

    real = _shutil.which

    def fake_which(name):
        if name == names:
            return None
        return real(name)

    monkeypatch.setattr(_shutil, "which", fake_which)


class TestVoxtypeSTT:
    def test_transcribe_runs_voxtype_cli_and_returns_last_stdout_line(
            self, monkeypatch, tmp_path):
        # stdout mixes log lines (with ANSI color codes) and the exact
        # transcription, which is always the last non-empty line.
        proc = _Proc(
            stdout="\x1b[2m2026-11-02T15:01:35Z\x1b[0m \x1b[33m WARN\x1b[0m "
                   "Config section 'parakeet' could not be read.\n"
                   "Loading audio file: \"in.wav\"\n"
                   "Processing 48000 samples (3.00s)...\n"
                   "Hola, soy Ana.\n")
        seen = _patch_run(monkeypatch, proc)
        wav = _write_wav(tmp_path / "in.wav")

        assert VoxtypeSTT().transcribe(wav) == "Hola, soy Ana."
        cmd = seen["cmd"]
        assert cmd[0].endswith("voxtype") and cmd[1] == "transcribe"
        assert cmd[-1] == str(wav)

    def test_empty_wav_returns_empty_without_running_voxtype(
            self, monkeypatch, tmp_path):
        import ana.voice.stt as stt

        def boom(*a, **kw):
            raise AssertionError("voxtype must not run for empty audio")

        monkeypatch.setattr(stt, "_run", boom)
        wav = _write_wav(tmp_path / "empty.wav", nframes=0)
        assert VoxtypeSTT().transcribe(wav) == ""

    def test_missing_voxtype_binary_raises_actionable_voice_error(
            self, monkeypatch, tmp_path):
        _which_missing(monkeypatch)
        wav = _write_wav(tmp_path / "in.wav")
        with pytest.raises(VoiceError) as excinfo:
            VoxtypeSTT().transcribe(wav)
        msg = str(excinfo.value)
        assert "voxtype not found" in msg
        assert "ANA_STT=whisper" in msg  # documented fallback

    def test_nonzero_exit_raises_with_stderr_tail(self, monkeypatch,
                                                  tmp_path):
        proc = _Proc(stderr="model file missing\n", rc=1)
        _patch_run(monkeypatch, proc)
        wav = _write_wav(tmp_path / "in.wav")
        with pytest.raises(VoiceError) as excinfo:
            VoxtypeSTT().transcribe(wav)
        msg = str(excinfo.value)
        assert "voxtype transcribe failed (exit 1)" in msg
        assert "model file missing" in msg


class TestBuildSTT:
    def test_explicit_backend_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("ANA_STT", "whisper")
        assert isinstance(build_stt("voxtype"), VoxtypeSTT)

    def test_env_backend_wins_over_default(self, monkeypatch):
        monkeypatch.setenv("ANA_STT", "whisper")
        assert isinstance(build_stt(), WhisperSTT)

    def test_default_backend_is_voxtype(self, monkeypatch):
        monkeypatch.delenv("ANA_STT", raising=False)
        assert isinstance(build_stt(), VoxtypeSTT)

    def test_unknown_backend_raises_actionable(self, monkeypatch):
        monkeypatch.delenv("ANA_STT", raising=False)
        with pytest.raises(VoiceError) as excinfo:
            build_stt("muleto")
        msg = str(excinfo.value)
        assert "unknown STT backend" in msg
        assert "voxtype, whisper" in msg




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

    def test_module_imports_cleanly_without_faster_whisper(self, tmp_path):
        # a fresh interpreter import must succeed with the package absent
        # (in-process re-import replaced the cached module mid-session and
        # left every later test patching a stale copy — subprocess is clean)
        import subprocess

        code = (
            "import sys, importlib\n"
            "sys.modules['faster_whisper'] = None\n"
            "mod = importlib.import_module('ana.voice.stt')\n"
            "assert mod.WhisperSTT().model_size == 'small'\n")
        env = {**os.environ, "PYTHONPATH": "."}
        proc = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


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
        """Legacy arecord world: no pw-record, no PipeWire default query."""
        import ana.voice.stt as stt
        procs = []

        def fake_popen(cmd, **kw):
            proc = FakeProc(cmd, **kw)
            procs.append(proc)
            return proc

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "shutil.which",
            lambda name: None if name == "pw-record" else f"/usr/bin/{name}")
        monkeypatch.setattr(
            stt, "capture_source",
            lambda: os.environ.get("ANA_MIC_DEVICE", "").strip())
        return procs

    def test_ana_mic_device_targets_pipewire_source(self, monkeypatch,
                                                    tmp_path):
        """$ANA_MIC_DEVICE pins a PipeWire capture node explicitly."""
        import ana.voice.stt as stt
        monkeypatch.setenv("ANA_MIC_DEVICE",
                           "alsa_input.usb-ICT_5M_USB_2.0_Camera"
                           "_200901010001-02.analog-stereo")
        # pw-record exists here (the fixture hides it for legacy tests)
        monkeypatch.setattr(
            stt, "capture_source",
            lambda: "alsa_input.usb-ICT_5M_USB_2.0_Camera"
                    "_200901010001-02.analog-stereo")
        procs = self._patch_popen(monkeypatch)
        monkeypatch.setattr(
            "shutil.which",
            lambda name: None if name == "arecord" else f"/usr/bin/{name}")
        press = FakePress(results=(True, True))

        record_to_wav(tmp_path / "in.wav", press=press)

        cmd = procs[0].cmd
        assert cmd[0].endswith("pw-record"), cmd
        i = cmd.index("--target")
        assert cmd[i + 1] == ("alsa_input.usb-ICT_5M_USB_2.0_Camera"
                              "_200901010001-02.analog-stereo")
        assert cmd[-1].endswith(".wav")

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
        monkeypatch.setattr(
            "shutil.which",
            lambda name: None if name == "pw-record" else f"/usr/bin/{name}")
        import ana.voice.stt as stt
        monkeypatch.setattr(stt, "capture_source", lambda: "")

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

    def test_speak_feeds_text_to_piper_stdin(self, monkeypatch):
        """piper reads its text from stdin: speak() must pass it as input.

        Without it, piper phonemizes nothing, writes zero samples, and
        fails at wav close ('# channels not specified') — caught by the
        first live smoke with real deps.
        """
        import ana.voice.tts as tts

        calls = []
        monkeypatch.setattr(
            tts, "_run", lambda cmd, **kw: calls.append((cmd, kw)) or None)
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

        PiperTTS(voice="/models/es_AR-daniela-high.onnx").speak("¡Hola Ana!")

        kw = calls[0][1]
        assert kw["input"] == "¡Hola Ana!\n".encode("utf-8")

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
        assert "(🎙" in out and "Enter para empezar" in out
        assert "mic:" in out  # the hint shows the capture source
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


class TestCaptureSource:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("ANA_MIC_DEVICE", "hw:CARD=Mini")
        assert capture_source() == "hw:CARD=Mini"

    def test_pactl_default_source_query(self, monkeypatch):
        import ana.voice.stt as stt

        monkeypatch.delenv("ANA_MIC_DEVICE", raising=False)

        def fake_run(cmd, **kw):
            assert cmd[:2] == ["pactl", "get-default-source"]
            return types.SimpleNamespace(
                returncode=0,
                stdout="alsa_input.usb-0c76_Razer_Seiren_Mini-00"
                       ".mono-fallback\n",
                stderr="")

        monkeypatch.setattr(stt, "_run", fake_run)
        assert capture_source() == ("alsa_input.usb-0c76_Razer_Seiren_Mini"
                                    "-00.mono-fallback")

    def test_missing_pactl_falls_back_to_default(self, monkeypatch):
        import ana.voice.stt as stt

        monkeypatch.delenv("ANA_MIC_DEVICE", raising=False)

        def boom(cmd, **kw):
            raise OSError("no pactl")

        monkeypatch.setattr(stt, "_run", boom)
        assert capture_source() == "default"


class TestRecorderSelection:
    """Record must capture only ONE mic node — never a monitor mix."""

    def test_pw_record_targets_the_pipe_wire_source(self, monkeypatch,
                                                    tmp_path):
        import ana.voice.stt as stt

        monkeypatch.setattr("shutil.which", lambda n: f"/usr/bin/{n}")
        monkeypatch.setattr(
            stt, "capture_source",
            lambda: "alsa_input.usb-0c76_Razer_Seiren_Mini-00"
                    ".mono-fallback")
        procs = self._popen(monkeypatch)
        press = FakePress(results=(True, True))

        record_to_wav(tmp_path / "in.wav", press=press)

        cmd = procs[0].cmd
        assert cmd[0] == "pw-record"  # PATH lookup, not the which() value
        assert "--rate" in cmd and cmd[cmd.index("--rate") + 1] == "16000"
        assert "--target" in cmd
        assert cmd[-1] == str(tmp_path / "in.wav")

    def test_aplay_style_env_name_routes_via_arecord_minus_D(
            self, monkeypatch, tmp_path):
        import ana.voice.stt as stt

        monkeypatch.setattr("shutil.which", lambda n: f"/usr/bin/{n}")
        monkeypatch.setattr(stt, "capture_source", lambda: "hw:CARD=Mini")
        procs = self._popen(monkeypatch)
        press = FakePress(results=(True, True))

        record_to_wav(tmp_path / "in.wav", press=press)

        cmd = procs[0].cmd
        assert cmd[0] == "arecord"  # PATH lookup, not the which() value
        i = cmd.index("-D")
        assert cmd[i + 1] == "hw:CARD=Mini"

    def test_missing_both_recorders_raises_actionable(self, monkeypatch,
                                                      tmp_path):
        import ana.voice.stt as stt

        monkeypatch.setattr("shutil.which", lambda n: None)
        monkeypatch.setattr(stt, "capture_source", lambda: "default")
        press = FakePress(results=())
        with pytest.raises(VoiceError) as excinfo:
            record_to_wav(tmp_path / "in.wav", press=press)
        msg = str(excinfo.value)
        assert "alsa-utils" in msg and "pipewire-utils" in msg

    @staticmethod
    def _popen(monkeypatch):
        procs = []

        def fake_popen(cmd, **kw):
            Path(cmd[-1]).touch()
            proc = types.SimpleNamespace(
                cmd=cmd, signals=[],
                send_signal=lambda sig: procs[-1].signals.append(sig),
                wait=lambda timeout=None: 0, kill=lambda: None,
                returncode=0)
            procs.append(proc)
            return proc

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        return procs


class TestPeakDb:
    def test_loud_wav_measures_positive_db(self, tmp_path):
        # full-scale sine sample: 0.5 → -6.02 dBFS
        assert peak_db(_write_wav(tmp_path / "loud.wav",
                                  sample=16384)) == pytest.approx(-6.02,
                                                                  abs=0.01)

    def test_near_silent_wav_floors_at_minus_96(self, tmp_path):
        assert peak_db(_write_wav(tmp_path / "quiet.wav",
                                  sample=10)) < -30  # halved: tiny sample

    def test_empty_wav_is_below_any_threshold(self, tmp_path):
        assert peak_db(_write_wav(tmp_path / "empty.wav", nframes=0)) == -96.0


class TestMicLabel:
    def test_raw_pipe_wire_node_friendly(self):
        assert mic_label("alsa_input.usb-0c76_Razer_Seiren_Mini-00"
                         ".mono-fallback") == "Razer Seiren Mini"

    def test_camera_node(self):
        assert mic_label("alsa_input.usb-ICT_5M_USB_2.0_Camera_200901010001"
                         "-02.analog-stereo") == "ICT 5M USB 2.0 Camera"

    def test_default_falls_back(self):
        assert mic_label("default") == "default"


class TestVoiceFeedback:
    """The user asked: show the mic used and warn on near-silent takes."""

    def _run(self, cfg, sample, transcript="hola"):
        wav = _write_wav(cfg.parent / "take.wav", sample=sample)
        os.environ["ANA_MIC_DEVICE"] = "alsa_input.usb-RAZER.wav"
        try:
            class Fake:
                def record(self, max_seconds, press):
                    press()
                    press(timeout=max_seconds)
                    return wav

                def transcribe(self, path):
                    assert Path(path) == wav
                    return transcript

                def speak(self, text):
                    pass

            out = []
            run_repl(cfg, lines=["", "exit"],
                     stream=stream_ok(), voice=True, voice_fx=Fake(),
                     emit=out.append)
        finally:
            os.environ.pop("ANA_MIC_DEVICE", None)
        return "\n".join(out)

    def test_hint_shows_the_mic_name(self, cfg):
        text = self._run(cfg, sample=16000)
        assert "(🎙 mic: RAZER" in text
        assert "Enter para empezar" in text

    def test_near_silent_take_warns_with_meter(self, cfg):
        text = self._run(cfg, sample=10)
        assert "casi silenciosa" in text

    def test_loud_take_does_not_warn(self, cfg):
        text = self._run(cfg, sample=16000)
        assert "casi silenciosa" not in text
