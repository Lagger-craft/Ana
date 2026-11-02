"""Tests for Phase 2 voice (S9/S10): STT, TTS, and the CLI voice loop.

Covers ana.voice.stt (WhisperSTT, record_to_wav), ana.voice.tts
(PiperTTS, resolve_voice) and the run_repl voice seam (/voice, --voice).
"""

import sys
import types
from pathlib import Path

import pytest

from ana.cli import main, run_repl
from ana.config import write_template
from ana.voice import PiperTTS, VoiceError, WhisperSTT, record_to_wav


@pytest.fixture
def cfg(tmp_path) -> object:
    path = tmp_path / "config.toml"
    write_template(path)
    return path


# ---------------------------------------------------------------- stt


class TestWhisperSTT:
    def test_transcribe_wraps_faster_whisper_and_strips(self, monkeypatch):
        seen = {}
        fake = types.ModuleType("faster_whisper")

        class FakeWhisperModel:
            def __init__(self, size, device=None, compute_type=None):
                seen["init"] = (size, device, compute_type)

            def transcribe(self, path, language=None):
                seen["transcribe"] = (path, language)
                return iter([("  hola  ", None)]), None

        fake.WhisperModel = FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)

        stt = WhisperSTT()  # defaults: small / cpu / int8 / es
        assert stt.transcribe("/tmp/x.wav") == "hola"
        assert seen["init"] == ("small", "cpu", "int8")
        assert seen["transcribe"] == ("/tmp/x.wav", "es")

    def test_missing_faster_whisper_raises_actionable_voice_error(
            self, monkeypatch):
        # None in sys.modules makes `import faster_whisper` raise ImportError
        # even if the package were installed
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        stt = WhisperSTT()
        with pytest.raises(VoiceError) as excinfo:
            stt.transcribe("/tmp/x.wav")
        msg = str(excinfo.value)
        assert "faster-whisper not installed" in msg
        assert ".venv/bin/pip install -e '.[voice]'" in msg

    def test_module_imports_cleanly_without_faster_whisper(self, monkeypatch):
        # a fresh import of the module must succeed with the package absent
        monkeypatch.delitem(sys.modules, "ana.voice.stt", raising=False)
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        import importlib

        mod = importlib.import_module("ana.voice.stt")
        assert mod.WhisperSTT().model_size == "small"  # fresh module, no fw


class TestRecordToWav:
    def test_calls_arecord_with_expected_args(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kw: calls.append((cmd, kw)) or None)
        out = tmp_path / "in.wav"

        result = record_to_wav(2.7, out)

        assert result == out
        cmd, kw = calls[0]
        assert cmd == ["arecord", "-f", "S16_LE", "-r", "16000",
                       "-d", "2", "-q", str(out)]
        assert kw.get("check") is True

    def test_missing_arecord_raises_voice_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(VoiceError) as excinfo:
            record_to_wav(1, "/tmp/x.wav")
        msg = str(excinfo.value)
        assert "alsa-utils" in msg
        assert "pw-record" in msg


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

    def test_missing_piper_raises_actionable_voice_error(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
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
    """Injectable voice seam for run_repl (record/transcribe/speak)."""

    def __init__(self, transcript="hola ana", record_error=None,
                 speak_error=None):
        self.transcript = transcript
        self.record_error = record_error
        self.speak_error = speak_error
        self.recorded = []
        self.spoken = []

    def record(self, seconds):
        if self.record_error is not None:
            raise self.record_error
        self.recorded.append(seconds)
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
            cfg_path=cfg, lines=["", "/exit"], stream=stream_ok("ok!"),
            voice=True, voice_fx=fx)
        assert fx.recorded == [5.0]  # default N seconds
        assert history[0] == {"role": "user", "content": "di hola"}
        assert history[-1] == {"role": "assistant", "content": "ok!"}
        assert fx.spoken == ["ok!"]
        assert "di hola" in capsys.readouterr().out  # transcript echoed

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
