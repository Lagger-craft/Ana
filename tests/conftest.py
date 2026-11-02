"""Shared test guards.

The waifu overlay is a LIVE desktop window: its FIFO
($XDG_RUNTIME_DIR/ana/waifu.fifo) exists while the overlay runs, and
every REPL/TUI turn sends a `say` event (S15b). Without this guard the
suite would paint test subtitles onto the real screen. Pointing
fifo_path at a path that never exists makes every waifu_send degrade
to its documented silent no-op. Tests that exercise the real send path
re-point fifo_path themselves (TestFifoSend) and win over this guard.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_live_overlay_fifo(tmp_path, monkeypatch):
    monkeypatch.setattr("ana.waifu.fifo_path",
                        lambda: tmp_path / "waifu.fifo")
