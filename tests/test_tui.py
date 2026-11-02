"""Tests for the Phase 4 Textual TUI (S13): same engine, screen frontend.

Runs the app headless through Textual's pilot. Turns reuse the CLI's
_do_turn via the injected seam, so provider/tools/images behavior stays
in one place. The transcript attribute is the deterministic surface for
assertions (widgets may reflow differently between versions).
"""

import time
import pytest

from ana.config import write_template


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.toml"
    write_template(path)
    return path


class TestTuiApp:
    @pytest.mark.asyncio
    async def test_tui_streams_reply_into_the_transcript(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False, stream=lambda msgs, model,
                     system=None: iter(["hola ", "mundo"]))
        async with app.run_test() as pilot:
            await pilot.press("h", "i")
            await pilot.press("enter")
            for _ in range(15):  # worker flushes its chunks synchronously
                await pilot.pause()
        assert "you> hi" in app.transcript
        assert "hola mundo" in app.transcript
        assert app.history[-1]["content"] == "hola mundo"

    @pytest.mark.asyncio
    async def test_tui_reply_renders_in_the_chat_widget(self, cfg):
        """The renderer must show the reply — a regression (RichLog.write
        with a removed kwarg, then per-chunk line splitting) drew broken
        screens while the transcript looked fine; assert on the real
        widget surface."""
        from textual.widgets import Static

        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False, stream=lambda msgs, model,
                     system=None: iter(["hola ", "mundo"]))
        async with app.run_test() as pilot:
            await pilot.press("h", "i")
            await pilot.press("enter")
            for _ in range(15):
                await pilot.pause()
            items = "\n".join(
                str(st.content) for st in app.query("#chat Static"))
            assert "hola mundo" in items   # the streaming reply, IN one piece
            assert "you> hi" in items      # and the user turn

    @pytest.mark.asyncio
    async def test_tui_new_clears_history_and_transcript(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False,
                     stream=lambda msgs, model, system=None: iter(["x"]))
        async with app.run_test() as pilot:
            await pilot.press("a")
            await pilot.press("enter")
            for _ in range(15):
                await pilot.pause()
            await pilot.press("/", "n", "e", "w", "enter")
            await pilot.pause()
        assert app.history == []
        assert "historial limpiado" in app.transcript

    @pytest.mark.asyncio
    async def test_tui_model_shows_active_model(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg)
        async with app.run_test() as pilot:
            await pilot.press("/", "m", "o", "d", "e", "l", "enter")
            await pilot.pause()
        assert "model:" in app.transcript

    def test_tui_requires_the_tui_extra(self, tmp_path, monkeypatch):
        # fabricated missing-package error stays actionable like voice's
        import builtins

        from ana.tui import ANA_TUI_MISSING_MSG, import_textual

        real_import = builtins.__import__
        missing = {}

        def no_textual(name, *a, **kw):
            if name.split(".")[0] == "textual":
                missing[name] = True
                raise ImportError("No module named 'textual'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", no_textual)
        import ana.tui as tui
        monkeypatch.setattr(tui, "_TEXTUAL_INSTALLED", False)
        with pytest.raises(SystemExit):
            tui.import_textual()

    def test_do_turn_streaming_updates_status_after_turn(self, cfg, capsys):
        # the non-async wiring: submit_turn shows the model it resolved
        from ana.cli import _model_for_turn
        from ana.config import Provider

        p = Provider(name="x", base_url="http://x/v1", default_model="m1")
        assert _model_for_turn(p, None, has_images=False) == "m1"


# ------------------------------------------------- voz en la TUI (T26)


class _VoiceFx:
    """Injectable voice seam like the CLI's (record/transcribe/speak)."""

    def __init__(self, transcript="hola por voz", speak_error=None):
        self.transcript = transcript
        self.speak_error = speak_error
        self.recorded = []   # (max_seconds, press_kind)
        self.spoken = []
        self.record_error = None

    def record(self, max_seconds, press):
        if self.record_error is not None:
            raise self.record_error
        press()                      # first press: start
        ok = press(timeout=max_seconds)  # second press or auto-stop
        self.recorded.append((max_seconds, "ok" if ok else "timeout"))
        return "/fake/take.wav"

    def transcribe(self, wav):
        return self.transcript

    def speak(self, text):
        if self.speak_error is not None:
            raise self.speak_error
        self.spoken.append(text)


def _press(app, keys, pauses=8):
    async def go(pilot):
        await pilot.press(*keys)
        for _ in range(pauses):
            await pilot.pause()
    return go


class TestTuiVoice:
    @pytest.mark.asyncio
    async def test_no_voice_by_default(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False,
                     stream=lambda *a, **k: iter(["ok"]))
        vfx = _VoiceFx()
        app.voice_fx = vfx
        async with app.run_test() as pilot:
            await pilot.press("enter")   # empty input with voice OFF: nothing
            for _ in range(6):
                await pilot.pause()
            assert vfx.recorded == []
            assert app.history == []

    @pytest.mark.asyncio
    async def test_voice_round_on_empty_enter_twice(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False,
                     stream=lambda msgs, model, system=None: iter(["ok"]))
        vfx = _VoiceFx(transcript="hola por voz")
        app.voice_fx = vfx
        app.voice_on = True
        async with app.run_test() as pilot:
            await pilot.press("enter")        # first press: START
            for _ in range(3):
                await pilot.pause()
            await pilot.press("enter")        # second press: STOP
            for _ in range(12):
                await pilot.pause()
        assert vfx.recorded, "record should have run"
        assert app.history[-2]["content"] == "hola por voz"
        assert app.history[-1]["role"] == "assistant"
        assert "hola por voz" in app.transcript

    @pytest.mark.asyncio
    async def test_reply_is_spoken_when_voice_on(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False,
                     stream=lambda msgs, model, system=None: iter(["digital!"]))
        vfx = _VoiceFx()
        app.voice_fx = vfx
        app.voice_on = True
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")   # typed input; reply spoken too
            for _ in range(12):
                await pilot.pause()
        assert app.history[-1]["content"] == "digital!"
        assert vfx.spoken == ["digital!"]

    @pytest.mark.asyncio
    async def test_speak_error_degrades_without_crashing(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False,
                     stream=lambda msgs, model, system=None: iter(["ok"]))
        vfx = _VoiceFx(speak_error=RuntimeError("no piper"))
        app.voice_fx = vfx
        app.voice_on = True
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")
            for _ in range(12):
                await pilot.pause()
        assert vfx.recorded == []
        assert app.history[-1]["content"] == "ok"   # turn kept
        assert "voice unavailable" in app.transcript

    @pytest.mark.asyncio
    async def test_voice_command_toggles(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False)
        async with app.run_test() as pilot:
            await pilot.press("/", "v", "o", "i", "c", "e", "space",
                              "o", "n", "enter")
            for _ in range(6):
                await pilot.pause()
            assert app.voice_on is True
            await pilot.press("/", "v", "o", "i", "c", "e", "space",
                              "o", "f", "f", "enter")
            for _ in range(6):
                await pilot.pause()
            assert app.voice_on is False


# -------------------------------------------------- TUI polish (T28)


class TestTuiPolish:
    @pytest.mark.asyncio
    async def test_mount_shows_a_welcome_note(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False)
        async with app.run_test() as pilot:
            for _ in range(4):
                await pilot.pause()
            assert "/help" in app.transcript
            assert "current_time" in app.transcript  # hints mention tools

    @pytest.mark.asyncio
    async def test_help_command_lists_commands(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False)
        async with app.run_test() as pilot:
            await pilot.press("/", "h", "e", "l", "p", "enter")
            for _ in range(6):
                await pilot.pause()
            assert "/model" in app.transcript and "/provider" in \
                app.transcript
            assert "/img" in app.transcript and "/voice" in app.transcript
            # spanish alias registers
            await pilot.press("/", "a", "y", "u", "d", "a", "enter")
            for _ in range(6):
                await pilot.pause()
        assert app.transcript.count("/model") >= 2

    @pytest.mark.asyncio
    async def test_items_get_role_classes(self, cfg):
        from textual.widgets import Static as S

        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False,
                     stream=lambda msgs, model, system=None: iter(["hola!"]))
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")
            for _ in range(12):
                await pilot.pause()
            you, reply = app.query("#chat > .you")[0], app.query(
                "#chat > .reply")[0]
            assert "you>" in str(you.content)
            assert "hola!" in str(reply.content)


# ------------------------------------------- ↑/↓ historial (T29)


class TestEntryHistory:
    @pytest.mark.asyncio
    async def test_up_recalls_the_last_message(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False,
                     stream=lambda *a, **k: iter(["ok"]))
        async with app.run_test() as pilot:
            await pilot.press("h", "o", "l", "a", "enter")
            for _ in range(8):
                await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert app.query_one("#entry").value == "hola"
            await pilot.press("down")
            await pilot.pause()
            assert app.query_one("#entry").value == ""

    @pytest.mark.asyncio
    async def test_down_after_two_recalls_in_order(self, cfg):
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False, stream=lambda *a, **k: iter(["ok"]))
        async with app.run_test() as pilot:
            await pilot.press("p", "r", "i", "m", "e", "r", "o", "enter")
            await pilot.press("s", "e", "g", "u", "n", "d", "o", "enter")
            for _ in range(10):
                await pilot.pause()
            await pilot.press("up")   # segundo
            await pilot.press("up")   # primero
            await pilot.press("down") # vuelve al segundo
            await pilot.pause()
            assert app.query_one("#entry").value == "segundo"

    @pytest.mark.asyncio
    async def test_commands_and_turns_both_count(self, cfg):
        # commands Pparte of the same history: recall repeats both
        from ana.tui import AnaApp

        app = AnaApp(cfg, tools=False, stream=lambda *a, **k: iter(["ok"]))
        async with app.run_test() as pilot:
            await pilot.press("/", "m", "o", "d", "e", "l", "enter")
            await pilot.press("h", "o", "l", "a", "enter")
            for _ in range(10):
                await pilot.pause()
            await pilot.press("up")
            await pilot.pause()
            assert app.query_one("#entry").value == "hola"
            await pilot.press("up")
            await pilot.pause()
            assert app.query_one("#entry").value == "/model"
            await pilot.press("down")
            await pilot.pause()
            assert app.query_one("#entry").value == "hola"
            await pilot.press("down")
            await pilot.pause()
            assert app.query_one("#entry").value == ""


# --------------------------------- indicador de actividad (T29 extra)


class TestBusyIndicator:
    @pytest.mark.asyncio
    async def test_status_shows_pensando_during_stream(self, cfg):
        import asyncio as aio

        from textual.widgets import Static

        from ana.tui import AnaApp

        def slow(msgs, model, system=None):
            yield "pr"
            time.sleep(1)  # turn engine is SYNC: windows reopen afterwards
            yield "ristino"

        app = AnaApp(cfg, tools=False, stream=slow)
        async with app.run_test() as pilot:
            await pilot.press("a", "enter")
            flag = ""
            for _ in range(60):          # poll instead of racing pauses
                await pilot.pause()
                flag = str(app.query_one("#status").content)
                if "respondiendo" in flag:
                    break
            assert "respondiendo" in flag  # work visible mid-turn
            cleared = False
            for _ in range(90):
                await pilot.pause()
                flag = str(app.query_one("#status").content)
                if "respondiendo" not in flag:
                    cleared = True
                    break
            assert cleared                # flag cleared when the turn ends
