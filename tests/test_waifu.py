"""Phase 5 waifu overlay (S14): pure logic testable; the GTK shell is thin."""

import json

from ana.waifu import parse_event, resolve_asset, WaifuState


class TestEventParsing:
    def test_line_to_state_event(self):
        state, subtitle = parse_event('{"type": "state", "state": "talk"}')
        assert state == "talk"
        assert subtitle is None

    def test_say_event_carries_text(self):
        state, subtitle = parse_event(
            json.dumps({"type": "say", "text": "¡Hola!"}))
        assert state is None
        assert subtitle == "¡Hola!"

    def test_garbage_line_degrades_without_crash(self):
        state, subtitle = parse_event("not json at all")
        assert state is None and subtitle is None

    def test_unknown_type_ignored(self):
        state, subtitle = parse_event('{"type": "frobnicate"}')
        assert state is None and subtitle is None


class TestStateFallback:
    def test_unknown_name_falls_back_to_idle(self):
        assert WaifuState.coerce("dormir") is WaifuState.IDLE

    def test_states_are_the_pngtuber_set(self):
        assert {s.value for s in WaifuState} == {"idle", "talk", "blink"}


class TestAssets:
    def test_missing_assets_resolve_to_none_no_crash(self, tmp_path,
                                                     monkeypatch):
        monkeypatch.setattr("ana.waifu._WAIFU_DIR", tmp_path)
        assert resolve_asset("idle") is None
        assert resolve_asset("talk") is None

    def test_user_pngs_are_used(self, tmp_path, monkeypatch):
        (tmp_path / "idle.png").write_bytes(b"png")
        (tmp_path / "talk.png").write_bytes(b"png2")
        monkeypatch.setattr("ana.waifu._WAIFU_DIR", tmp_path)
        assert resolve_asset("idle") == tmp_path / "idle.png"
        assert resolve_asset("talk") == tmp_path / "talk.png"

    def test_ana_env_dir_wins_over_home(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "custom"
        env_dir.mkdir()
        (env_dir / "blink.png").write_bytes(b"png3")
        monkeypatch.setattr("ana.waifu._WAIFU_DIR", None)
        monkeypatch.setenv("ANA_WAIFU_DIR", str(env_dir))
        assert resolve_asset("blink") == env_dir / "blink.png"


# ------------------------------------------------ monitor choice (S14)


class TestMonitorChoice:
    def test_preferred_connector_wins(self):
        from ana.waifu import choose_monitor

        names = ["DP-1", "DP-2", "HDMI-A-1"]
        assert choose_monitor(names, "DP-2", "DP-1") == "DP-2"
        assert choose_monitor(names, "HDMI-A-1", "DP-1") == "HDMI-A-1"

    def test_default_avoids_dp1_by_knowledge_of_the_actual_setup(self):
        from ana.waifu import choose_monitor

        names = ["DP-1", "DP-2", "HDMI-A-1"]
        assert choose_monitor(names, None, "DP-1") == "DP-2"

    def test_single_monitor_falls_back_to_it(self):
        from ana.waifu import choose_monitor

        assert choose_monitor(["DP-1"], None, "DP-1") == "DP-1"

    def test_unknown_preferred_name_falls_back_to_avoid_rule(self):
        from ana.waifu import choose_monitor

        names = ["DP-1", "DP-2"]
        assert choose_monitor(names, "HDMI-A-9", "DP-1") == "DP-2"


# ------------------------------------------------------ fifo events


class TestFifoSend:
    def test_send_overlay_event_writes_json_line(self, tmp_path, monkeypatch):
        import os
        import threading

        from ana.waifu import fifo_path, waifu_send

        monkeypatch.setattr("ana.waifu.fifo_path", lambda: tmp_path / "w.fifo")
        path = tmp_path / "w.fifo"
        os.mkfifo(path)

        lines = []
        def reader():
            with open(path, "r") as fh:
                for line in fh:
                    lines.append(line.strip())
                    break

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        # wait until the reader has opened the FIFO (both sides)
        import time
        for _ in range(30):
            time.sleep(0.02)
            if lines:
                break

        waifu_send({"type": "state", "state": "talk"})
        t.join(timeout=2)
        assert lines == ['{"type": "state", "state": "talk"}']

    def test_missing_fifo_is_a_silent_noop(self, tmp_path, monkeypatch):
        from ana.waifu import fifo_path, waifu_send

        monkeypatch.setattr("ana.waifu.fifo_path", lambda: tmp_path / "no.fifo")
        waifu_send({"type": "state", "state": "talk"})  # no crash, no error
