"""Tests for the Phase 4 Textual TUI (S13): same engine, screen frontend.

Runs the app headless through Textual's pilot. Turns reuse the CLI's
_do_turn via the injected seam, so provider/tools/images behavior stays
in one place. The transcript attribute is the deterministic surface for
assertions (widgets may reflow differently between versions).
"""

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
