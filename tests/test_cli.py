"""Tests for ana.cli — REPL commands, one-shot mode, init and error paths."""

import json
import types
from pathlib import Path
import types
import pytest

from ana.cli import main, parse_command, run_repl
from ana.client import ProviderError
from ana.config import CONFIG_TEMPLATE, ConfigError, write_template


class TestParseCommand:
    def test_command_with_arg(self):
        assert parse_command("/model gpt-4o-mini") == ("model", "gpt-4o-mini")

    def test_command_without_arg(self):
        assert parse_command("/new") == ("new", "")

    def test_non_command_raises(self):
        with pytest.raises(ValueError):
            parse_command("hello ana")


@pytest.fixture
def cfg(tmp_path) -> object:
    path = tmp_path / "config.toml"
    write_template(path)
    return path


class TestRepl:
    def test_turn_appends_user_and_assistant_to_history(self, cfg):
        history = run_repl(
            cfg_path=cfg, lines=["hola"],
            stream=lambda msgs, model, system=None: iter(["hola!"]),
        )
        assert [(m["role"], m["content"]) for m in history] == [
            ("user", "hola"), ("assistant", "hola!")]

    def test_exit_ends_loop_without_api_call(self, cfg):
        history = run_repl(cfg_path=cfg, lines=["/exit"], stream=None)
        assert history == []

    def test_new_clears_history(self, cfg):
        history = run_repl(
            cfg_path=cfg,
            lines=["turn1", "/new", "turn2"],
            stream=lambda msgs, model, system=None: iter(["r"]),
        )
        # after /new only the second turn remains
        assert [m["content"] for m in history] == ["turn2", "r"]

    def test_failed_turn_is_rolled_back_and_loop_survives(self, cfg):
        replies = iter([ProviderError("boom"), iter(["ok"])])

        def stream(msgs, model, system=None):
            item = next(replies)
            if isinstance(item, Exception):
                raise item
            return item

        history = run_repl(cfg_path=cfg, lines=["q1", "q2"], stream=stream)
        # q1 rolled back on failure; only the successful turn stays
        assert [(m["role"], m["content"]) for m in history] == [
            ("user", "q2"), ("assistant", "ok")]

    def test_unknown_command_reported_not_sent(self, cfg, capsys):
        history = run_repl(cfg_path=cfg, lines=["/wat"], stream=None)
        assert history == []
        assert "unknown command" in capsys.readouterr().out

    def test_model_without_arg_shows_current(self, cfg, capsys):
        run_repl(cfg_path=cfg, lines=["/model"], stream=None)
        assert "kimi-k3" in capsys.readouterr().out

    def test_model_with_arg_switches_model(self, cfg):
        seen = []

        def stream(msgs, model, system=None):
            seen.append(model)
            return iter(["ok"])

        run_repl(cfg_path=cfg, lines=["/model my-model", "hi"], stream=stream)
        assert seen == ["my-model"]

    def test_provider_with_arg_switches_provider(self, cfg):
        seen = []

        def stream(msgs, model, system=None):
            seen.append(model)
            return iter(["ok"])

        history = run_repl(
            cfg_path=cfg, lines=["/provider openai", "hi"],
            stream=stream, models={"openai": "gpt-4o-mini"})
        assert seen == ["gpt-4o-mini"]
        assert history != []

    def test_history_grows_across_turns(self, cfg):
        def stream(msgs, model, system=None):
            return iter(["r"])

        history = run_repl(cfg_path=cfg, lines=["a", "b"], stream=stream)
        assert [m["content"] for m in history] == ["a", "r", "b", "r"]


class TestCloseOverlay:
    """Option B (default): closing the chat closes the overlay.

    Headless `ana talk` rounds pass close_overlay=False — they have no
    chat window, and the hotkey wrapper keeps the overlay up across
    rounds. The DEFAULT quit event stays pinned by
    tests/test_waifu.py::TestReplWaifuEvents.
    """

    def test_close_overlay_false_keeps_the_overlay_open(self, cfg,
                                                         monkeypatch):
        sent = []
        # run_repl imports waifu_send from ana.waifu at call time, so
        # that module is the patchable seam.
        monkeypatch.setattr("ana.waifu.waifu_send", sent.append)

        run_repl(cfg_path=cfg, lines=["/exit"], stream=None,
                 close_overlay=False)

        assert {"type": "quit"} not in sent


class TestMain:
    def test_one_shot_prints_stream_and_exits_zero(
            self, cfg, monkeypatch, capsys):
        monkeypatch.setattr(
            "ana.cli.build_stream",
            lambda cfg, provider=None: lambda msgs, model, system=None:
                iter(["hi", " there"]))
        rc = main(["--config", str(cfg), "--message", "hi", "--no-tools"])
        assert rc == 0
        assert "hi there" in capsys.readouterr().out

    def test_missing_config_exits_one_with_init_hint(self, tmp_path, capsys):
        rc = main(["--config", str(tmp_path / "missing.toml"),
                   "--message", "hi"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "ana init" in err

    def test_missing_api_key_env_exits_one_actionably(self, tmp_path,
                                                      monkeypatch, capsys):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            CONFIG_TEMPLATE.replace('default_provider = "opencode-go"',
                                    'default_provider = "openai"'))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rc = main(["--config", str(cfg_path), "--message", "hi"])
        assert rc == 1
        assert "OPENAI_API_KEY" in capsys.readouterr().err

    def test_provider_error_exits_one_with_message(self, cfg, capsys):
        def boom(msgs, model, system=None):
            raise ProviderError("500 'oops' body")
            yield  # pragma: no cover

        import ana.cli as cli_module
        cli_module.build_stream = lambda cfg, provider=None: boom
        rc = main(["--config", str(cfg), "--message", "hi", "--no-tools"])
        assert rc == 1
        assert "oops" in capsys.readouterr().err

    def test_init_writes_template_to_default_location(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        rc = main(["init"])
        out = tmp_path / "ana" / "config.toml"
        assert rc == 0
        assert out.read_text() == CONFIG_TEMPLATE


# ------------------------------------------------------ imágenes (T21)


class _RecordingFx:
    """Turn-level client that records every request like the real one."""

    api_style = "chat"

    def __init__(self):
        self.calls = []

    def stream_turn(self, messages, model, system=None, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "model": model})
        if self.calls[-1]["messages"][-1].get("images"):
            fake_events = [(("✓ imagen vista"),)]
        return iter([{"type": "text", "content": "ok"},
                     {"type": "finish", "finish_reason": "stop"}])


def _png(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    return p


class TestImageAttachment:
    def test_user_message_encodes_base64_with_mime(self, tmp_path):
        import base64
        from ana.cli import _user_message

        msg = _user_message("mira esto", [_png(tmp_path)])
        assert msg["content"] == "mira esto"
        img = msg["images"][0]
        assert img["media_type"] == "image/png"
        assert img["data"] == base64.b64encode(b"\x89PNG\r\n\x1a\nrest").decode()


    def test_rejects_non_image_file_with_actionable_error(self, tmp_path):
        from ana.cli import _user_message

        bad = tmp_path / "doc.txt"
        bad.write_text("texto")
        with pytest.raises(ValueError) as excinfo:
            _user_message("x", [bad])
        assert "no es una imagen" in str(excinfo.value)
        assert "doc.txt" in str(excinfo.value)

    def test_rejects_missing_file(self, tmp_path):
        from ana.cli import _user_message

        with pytest.raises(ValueError, match="no existe"):
            _user_message("x", [tmp_path / "fantasma.png"])

    def test_no_images_strips_the_key(self):
        from ana.cli import _user_message

        assert _user_message("hola", []) == {"role": "user", "content": "hola"}

    def test_img_queues_and_the_next_message_carries_them(self, cfg, tmp_path):
        img = _png(tmp_path)
        fx = _RecordingFx()

        def user_of(call):
            ms = [m for m in call["messages"] if m.get("role") == "user"]
            return ms[-1]

        run_repl(cfg_path=cfg, lines=[f"/img {img}", "mira",
                                      "otra cosa", "/exit"],
                 voice_fx=None, registry=None, client_fx=fx,
                 tools=True)
        first = user_of(fx.calls[0])
        assert first["images"][0]["media_type"] == "image/png"
        assert first["content"] == "mira"
        # second turn: queue was cleared by the first send
        assert "images" not in user_of(fx.calls[1])

    def test_img_without_arg_shows_pending(self, cfg, tmp_path, capsys):
        img = _png(tmp_path)
        run_repl(cfg_path=cfg, lines=[f"/img {img}", "/img", "hola", "/exit"],
                 stream=lambda msgs, model, system=None: iter(["ok"]))
        out = capsys.readouterr().out
        assert "pic.png" in out

    def test_img_missing_path_reports_and_sends_nothing(self, cfg, capsys):
        out_lines = []
        history = run_repl(cfg_path=cfg, lines=["/img /no/existe.png", "/exit"],
                           stream=lambda *a, **k: iter(["!"]),
                           emit=out_lines.append)
        assert history == []
        joined = "\n".join(out_lines)
        assert "no existe" in joined.lower()



class TestVoiceDefaultOn:
    """Voice is ON by default; --no-voice is the opt-out (user request)."""

    @staticmethod
    def _fake_repl(monkeypatch):
        import ana.cli as cli
        seen = {}

        def fake(cfg_path, **kw):
            seen.update(kw)
            return []

        monkeypatch.setattr(cli, "run_repl", fake)
        return seen

    def test_repl_starts_with_voice_on_by_default(self, cfg, monkeypatch):
        seen = self._fake_repl(monkeypatch)
        assert main(["--config", str(cfg)]) == 0
        assert seen["voice"] is True

    def test_no_voice_flag_starts_text_only(self, cfg, monkeypatch):
        seen = self._fake_repl(monkeypatch)
        assert main(["--config", str(cfg), "--no-voice"]) == 0
        assert seen["voice"] is False

    def test_tui_starts_with_voice_on_by_default(self, cfg, monkeypatch):
        import ana.tui as tui_mod
        seen = {}

        class FakeApp:
            def __init__(self, *args, **kw):
                seen.update(kw)

            def run(self):
                return None

        monkeypatch.setattr(tui_mod, "AnaApp", FakeApp)
        tui_mod.run_tui(str(cfg))
        assert seen["voice"] is True

    def test_tui_forwards_stt_and_tts_backends(self, cfg, monkeypatch):
        """--stt/--tts reach the TUI like they reach the REPL and talk."""
        import ana.tui as tui_mod
        seen = {}

        class FakeApp:
            def __init__(self, *args, **kw):
                seen.update(kw)

            def run(self):
                return None

        monkeypatch.setattr(tui_mod, "AnaApp", FakeApp)
        assert main(["--config", str(cfg), "--tui",
                     "--stt", "whisper", "--tts", "piper"]) == 0
        assert seen["stt"] == "whisper"
        assert seen["tts"] == "piper"


class TestVisionModel:
    def test_pick_model_returns_vision_model_when_images_present(self):
        from ana.cli import _model_for_turn
        from ana.config import Provider

        p = Provider(name="ollama", base_url="http://x/v1",
                     default_model="qwen3:8b", vision_model="qwen3-vl:4b")
        assert _model_for_turn(p, None, has_images=True) == "qwen3-vl:4b"
        assert _model_for_turn(p, None, has_images=False) == "qwen3:8b"
        # explicit user model always wins
        assert _model_for_turn(p, "m2", has_images=True) == "m2"

    def test_pick_model_without_vision_model_takes_default(self):
        from ana.cli import _model_for_turn
        from ana.config import Provider

        p = Provider(name="x", base_url="http://x/v1", default_model="m1")
        assert _model_for_turn(p, None, has_images=True) == "m1"

    def test_repl_images_auto_pick_vision_model(self, cfg, tmp_path):
        # the template ships an ollama vision_model (qwen3-vl:4b)
        img = _png(tmp_path)
        fx = _RecordingFx()
        run_repl(cfg_path=cfg, lines=[f"/img {img}", "mira", "/exit"],
                 provider_override="ollama", client_fx=fx, tools=True)
        assert fx.calls[0]["model"] == "qwen3-vl:4b"  # attached → vision


class TestImageDownscale:
    """OCR tokens come from the vision encoder's grid: big images are the
    slow path. _encode_image shrinks oversized shots to ANA_IMAGE_MAX_PX
    (default 1024) via ffmpeg before base64; smaller ones pass through."""

    def test_oversized_image_values_is_resized(self, monkeypatch, tmp_path):
        import ana.cli as cli

        big = tmp_path / "big.png"
        big.write_bytes(b"\x89PNG" + b"x" * 40)

        probes = {
            "ffprobe": types.SimpleNamespace(
                returncode=0, stdout=json.dumps(
                    {"streams": [{"width": 4096, "height": 3072}]},
                ), stderr=""),
            "ffmpeg": None,  # handled by the cmd seam callback below
        }

        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[0] == "ffprobe":
                return probes["ffprobe"]
            Path(cmd[-1]).write_bytes(b"\x89PNGsmall")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        result = cli._encode_image(big)
        ffmpeg_cmd = [c for c in calls if c[0] == "ffmpeg"][0]
        joined = " ".join(ffmpeg_cmd)
        assert "scale" in joined and "1024" in joined
        # result carries the SHRUNK bytes, mime stays png
        import base64 as b64mod
        assert result["media_type"] == "image/png"
        assert b64mod.b64decode(result["data"]) == b"\x89PNGsmall"

    def test_small_image_keeps_original_bytes(self, monkeypatch, tmp_path):
        import ana.cli as cli

        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        small = _png(tmp_path)
        result = cli._encode_image(small)
        assert [c[0] for c in calls] == ["ffprobe"]  # no resize executed
        import base64 as b64mod
        assert b64mod.b64decode(result["data"]) == small.read_bytes()

    def test_ana_image_max_px_env_overrides(self, monkeypatch, tmp_path):
        import ana.cli as cli

        monkeypatch.setenv("ANA_IMAGE_MAX_PX", "640")

        def fake_run(cmd, **kw):
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(
                {"streams": [{"width": 2048, "height": 2049}]}), stderr="")

        seen = []

        def fake_run(cmd, **kw):
            seen.append(cmd)
            if cmd[0] == "ffprobe":
                return types.SimpleNamespace(returncode=0, stdout=json.dumps(
                    {"streams": [{"width": 2048, "height": 2049}]}), stderr="")
            Path(cmd[-1]).write_bytes(b"\xff\xd8small")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        big = tmp_path / "big.jpg"
        big.write_bytes(b"\xff\xd8\xffII*")
        cli._encode_image(big)
        cmd = [c for c in seen if c[0] == "ffmpeg"][0]
        joined = " ".join(cmd)
        assert "scale" in joined and "640" in joined



# --------------------------------------------- memory-correction rule (prompt)


class TestSystemPromptMemoryRule:
    """The injected memory section must tell Ana to persist corrections
    (same title -> in-place update) in the SAME turn, before replying."""

    RULE = ("Before answering, when the user states, corrects or retracts "
            "a durable personal fact, call memory_save with the SAME title "
            "as the existing memory so it updates in place — do it in "
            "that turn, not later.")

    def test_prompt_includes_the_memory_correction_rule(self, cfg,
                                                        monkeypatch):
        from ana.cli import _system_prompt
        from ana.config import load_config
        import ana.memory as memory
        monkeypatch.setattr(memory, "session_context", lambda: "CTX-FAKE")
        prompt = _system_prompt(load_config(cfg))
        assert self.RULE in prompt
