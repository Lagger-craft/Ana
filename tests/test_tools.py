"""Tests for ana.tools — Tavily web search, hyprctl read-only, system info,
Engram-backed memory tools."""

import os
import shutil
import sys

import pytest


@pytest.fixture
def tools(monkeypatch):
    import ana.tools as tools_mod
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    return tools_mod.build_default_tools()


class TestWebSearch:
    def test_registered_and_schemas(self, tools, monkeypatch):
        reg = tools
        assert "web_search" in reg.names()
        spec = reg.to_spec("chat")
        ws = [t for t in spec if t["function"]["name"] == "web_search"][0]
        assert ws["function"]["parameters"]["properties"]["query"]["type"] == "string"
        assert ws["function"]["parameters"]["properties"]["count"]["type"] == "integer"

    def test_missing_key_degrades_to_error_string(self, tools, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = tools.call("web_search", {"query": "hyprland"})
        assert "TAVILY_API_KEY" in result
        assert "unavailable" in result

    def test_tavily_query_maps_results(self, tools, monkeypatch):
        fake = type("Resp", (), {})()
        fake.json = lambda: {"results": [
            {"title": "Hyprland", "url": "https://hypr.land", "content": "compositor"},
            {"title": "Omarchy", "url": "https://omarchy.org", "content": None},
        ]}
        fake.raise_for_status = lambda: None
        calls = []
        monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

        def fake_post(url, *, json=None, headers=None, timeout=None):
            calls.append((url, json, headers, timeout))
            return fake

        import ana.tools as tools_mod
        monkeypatch.setattr(tools_mod.httpx, "post", fake_post)
        result = tools.call("web_search", {"query": "wayland compositor", "count": 2})
        assert "1. Hyprland" in result
        assert "https://hypr.land" in result
        assert "2. Omarchy" in result
        assert "[no description]" in result
        url, payload, headers, timeout = calls[0]
        assert url == "https://api.tavily.com/search"
        assert payload["query"] == "wayland compositor"
        assert payload["max_results"] == 2
        assert payload["search_depth"] == "basic"
        assert headers.get("Authorization") == "Bearer tavily-key"
        assert headers.get("Content-Type") == "application/json"
        assert timeout == 15

    def test_tavily_error_degrades_to_message(self, tools, monkeypatch):
        class Fail(Exception):
            pass

        def fake_post(*a, **kw):
            raise Fail("boom")

        monkeypatch.setenv("TAVILY_API_KEY", "k")
        import ana.tools as tools_mod
        monkeypatch.setattr(tools_mod.httpx, "post", fake_post)
        result = tools.call("web_search", {"query": "x"})
        assert "unavailable" in result


class TestHyprctl:
    def test_registered_commands(self, tools):
        for name in ("hypr_active_window", "hypr_workspaces",
                     "hypr_monitors", "hypr_version", "hypr_devices"):
            assert name in tools.names()

    def test_runs_hyprctl_subcommand(self, tools, monkeypatch):
        import ana.tools as tools_mod
        ran = []

        def fake_run(cmd, **kw):
            ran.append(cmd)
            return type("Proc", (), {"stdout": " hyprctl info\n",
                                     "returncode": 0})()

        monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)
        result = tools.call("hypr_workspaces", {})
        assert result == "hyprctl info"
        assert ran == [["hyprctl", "workspaces", "-j"]]

    def test_missing_hyprctl_degrades_to_message(self, tools, monkeypatch,
                                                 tmp_path):
        """No Hyprland (labwc/KDE/Niri): the tool MUST answer a string,
        never raise — run_agent only traps AgentError, so an OSError here
        kills the whole turn.
        """
        monkeypatch.setenv("PATH", str(tmp_path))  # hyprctl no existe
        result = tools.call("hypr_monitors", {})
        assert isinstance(result, str)
        assert "hyprctl" in result
        assert "unavailable" in result.lower() or "no disponible" in result

    def test_timeout_degrades_to_message(self, tools, monkeypatch):
        """A hung hyprctl must not kill the turn either (TimeoutExpired is
        a SubprocessError, not an AgentError)."""
        import subprocess
        import ana.tools as tools_mod

        def fake_run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=10)

        monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)
        result = tools.call("hypr_devices", {})
        assert isinstance(result, str)
        assert "hyprctl" in result

    def test_no_log_too_deep_flag_allowlist(self, tools, monkeypatch, capsys):
        assert "hypr_dispatch" not in tools.names()  # destructive, not present
        assert "hypr_kill" not in tools.names()

    def test_invalid_subcommand_not_attempted(self, tools):
        # the tool set simply does not contain arbitrary dispatch
        with pytest.raises(Exception, match="unknown tool"):
            tools.call("hypr_dispatch", {"args": "exit"})


class TestSystemInfo:
    def test_registered_and_returns_fields(self, tools):
        assert "system_info" in tools.names()
        result = tools.call("system_info", {})
        assert result["os"] == sys.platform
        assert result["python"] == sys.version.split()[0]
        assert isinstance(result["kernel"], str)
        assert isinstance(result["cpu_count"], int)

    def test_search_registered_regardless_of_key(self, tools):
        assert "web_search" in tools.names()


class TestToolInstances:
    def test_build_default_tools_registers_all(self):
        import ana.tools as tools_mod
        reg = tools_mod.build_default_tools()
        assert set(reg.names()) == {
            "web_search", "hypr_active_window", "hypr_workspaces",
            "hypr_monitors", "hypr_version", "hypr_devices", "system_info",
            "current_time", "memory_save", "memory_search", "screenshot"}


# --------------------------------------------- screenshot (same-turn image)


class TestScreenshot:
    """spectacle captures the active monitor -> ToolResult image for the
    current turn. Tests patch the `_run_spectacle` subprocess seam — the
    suite never runs the real binary."""

    def test_registered_with_screen_description_and_empty_schema(self, tools):
        assert "screenshot" in tools.names()
        spec = [t for t in tools.to_spec("chat")
                if t["function"]["name"] == "screenshot"][0]["function"]
        assert spec["parameters"]["properties"] == {}
        assert spec["parameters"].get("required", []) == []
        desc = spec["description"].lower()
        assert "screen" in desc

    def test_capture_runs_spectacle_with_active_monitor_flags(
            self, tools, monkeypatch):
        import base64 as _b64
        import ana.tools as tools_mod
        monkeypatch.setattr(tools_mod.shutil, "which",
                            lambda name: "/usr/bin/spectacle")
        ran = []

        def fake_run(cmd, timeout=None):
            ran.append((cmd, timeout))
            out = cmd[cmd.index("-o") + 1]
            with open(out, "wb") as fh:
                fh.write(b"\x89PNG-fake-bytes")
            return type("Proc", (), {"returncode": 0, "stderr": b""})()

        monkeypatch.setattr(tools_mod, "_run_spectacle", fake_run)
        result = tools.call("screenshot", {})

        from ana.agent import ToolResult
        assert isinstance(result, ToolResult)
        cmd, timeout = ran[0]
        assert cmd[:5] == ["spectacle", "-b", "-m", "-n", "-o"]
        assert timeout == 10
        out = cmd[5]
        assert out.endswith(".png")
        assert not os.path.exists(out)      # tmp file never leaks
        assert result.image["media_type"] == "image/png"
        assert result.image["data"] == _b64.b64encode(
            b"\x89PNG-fake-bytes").decode("ascii")
        assert isinstance(result.text, str) and result.text

    def test_spectacle_failure_degrades_to_error_string(self, tools,
                                                        monkeypatch):
        import ana.tools as tools_mod
        monkeypatch.setattr(tools_mod.shutil, "which",
                            lambda name: "/usr/bin/spectacle")
        ran = []

        def fake_run(cmd, timeout=None):
            ran.append(cmd)
            return type("Proc", (), {"returncode": 1,
                                     "stderr": b"no compositor"})()

        monkeypatch.setattr(tools_mod, "_run_spectacle", fake_run)
        out = tools.call("screenshot", {})
        assert isinstance(out, str)          # honest note, never an exception
        assert "no compositor" in out        # carries spectacle's reason
        tmp_out = ran[0][ran[0].index("-o") + 1]
        assert not os.path.exists(tmp_out)   # no leftovers on failure either

    def test_missing_spectacle_degrades_to_error_string(self, tools,
                                                        monkeypatch,
                                                        tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path))   # no spectacle anywhere
        monkeypatch.setattr("ana.tools._run_spectacle",
                            lambda *a, **k: pytest.fail("must not run"))
        out = tools.call("screenshot", {})
        assert isinstance(out, str)
        assert "spectacle" in out
        assert "unavailable" in out

    def test_spectacle_timeout_degrades_to_error_string(self, tools,
                                                        monkeypatch):
        import subprocess
        import ana.tools as tools_mod
        monkeypatch.setattr(tools_mod.shutil, "which",
                            lambda name: "/usr/bin/spectacle")

        def fake_run(cmd, timeout=None):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(tools_mod, "_run_spectacle", fake_run)
        out = tools.call("screenshot", {})
        assert isinstance(out, str)          # a hung capture must not kill
        assert "timed out" in out            # the turn either

    @staticmethod
    def _ffmpeg_seam(w, h, color="red"):
        """Fake spectacle that writes a REAL w×h PNG via ffmpeg."""
        import subprocess

        def fake_run(cmd, timeout=None):
            out = cmd[cmd.index("-o") + 1]
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                 "-i", f"color=c={color}:s={w}x{h}",
                 "-frames:v", "1", out],
                check=True, capture_output=True)
            return type("Proc", (), {"returncode": 0, "stderr": b""})()

        return fake_run

    @pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
        reason="ffmpeg/ffprobe needed for the shrink pipeline")
    def test_oversized_capture_downscaled_in_injected_message(
            self, tools, monkeypatch):
        """A raw 2000×1080-class capture must NOT reach the provider:

        the shared /img pipeline (ANA_IMAGE_MAX_PX, default 1024) runs
        before base64, and the SHRUNK bytes are what the agent loop
        injects into the turn (image/png stays image/png).
        """
        import base64 as _b64
        import struct
        import ana.tools as tools_mod
        from tests.test_agent import FakeTurnClient, _final_round, _shot_round
        from ana.agent import run_agent

        monkeypatch.setenv("ANA_IMAGE_MAX_PX", "1024")
        monkeypatch.setattr(tools_mod.shutil, "which",
                            lambda name: "/usr/bin/spectacle")
        monkeypatch.setattr(tools_mod, "_run_spectacle",
                            self._ffmpeg_seam(2000, 1500))

        client = FakeTurnClient([_shot_round(), _final_round()])
        run_agent(client, [{"role": "user", "content": "what's on screen"}],
                  model="m1", registry=tools, emit=lambda _: None,
                  vision_model="v1")

        injected = client.calls[1][1][-1]     # synthetic user message
        assert injected["role"] == "user"
        image = injected["images"][0]
        assert image["media_type"] == "image/png"
        raw = _b64.b64decode(image["data"])
        w, h = struct.unpack(">II", raw[16:24])   # PNG IHDR
        assert w <= 1024 and h <= 1024            # ANA_IMAGE_MAX_PX

    @pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
        reason="ffmpeg/ffprobe needed for the shrink pipeline")
    def test_small_capture_passes_through_without_upscale(
            self, tools, monkeypatch):
        """Below the limit the shared pipeline must not touch the file:
        a 64×48 capture injects as the same 64×48 PNG — no upscale,
        no re-encode.
        """
        import base64 as _b64
        import struct
        import ana.tools as tools_mod
        from ana.agent import ToolResult

        monkeypatch.setenv("ANA_IMAGE_MAX_PX", "1024")
        monkeypatch.setattr(tools_mod.shutil, "which",
                            lambda name: "/usr/bin/spectacle")
        monkeypatch.setattr(tools_mod, "_run_spectacle",
                            self._ffmpeg_seam(64, 48, color="blue"))

        result = tools.call("screenshot", {})

        assert isinstance(result, ToolResult)
        assert result.image["media_type"] == "image/png"
        raw = _b64.b64decode(result.image["data"])
        assert struct.unpack(">II", raw[16:24]) == (64, 48)


# --------------------------------------------- current_time (S8+)


class TestCurrentTime:
    def test_registered_in_the_default_registry(self, tools):
        assert tools.has_tool("current_time") if hasattr(tools, "has_tool") \
            else "current_time" in tools.names()

    def test_returns_local_time_and_iso(self, tools, monkeypatch):
        monkeypatch.setenv("TZ", "America/Argentina/Buenos_Aires")
        out = tools.call("current_time", {})
        # both a machine-readable ISO and a human-readable local time
        assert "T" in out and ":" in out
        assert str(__import__("datetime").datetime.now().year) in out

    def test_accepts_timezone_override(self, tools, monkeypatch):
        monkeypatch.setenv("TZ", "UTC")
        out = tools.call("current_time", {"timezone": "Europe/Madrid"})
        assert "T" in out


# --------------------------------------------- memory tools (Engram)


class TestMemoryTools:
    def test_registered_with_schemas(self, tools):
        assert "memory_save" in tools.names()
        assert "memory_search" in tools.names()
        spec = {t["function"]["name"]: t["function"]
                for t in tools.to_spec("chat")}
        save = spec["memory_save"]
        assert save["parameters"]["required"] == ["title", "content"]
        props = save["parameters"]["properties"]
        assert props["type"]["enum"] == ["fact", "preference", "persona"]
        # the description must steer the model to FUTURE sessions
        assert "future" in save["description"].lower()
        search = spec["memory_search"]["parameters"]
        assert search["required"] == ["query"]
        assert "query" in search["properties"]

    def test_save_description_titles_are_keys_and_update_in_place(self, tools):
        """The schema stays the same, but the description must teach the
        model that an EXISTING title updates that memory in place."""
        spec = {t["function"]["name"]: t["function"]
                for t in tools.to_spec("chat")}
        desc = spec["memory_save"]["description"].lower()
        assert "existing title" in desc
        assert "updates that memory in place" in desc
        assert "never create a second record" in desc

    def test_save_posts_through_the_memory_seam(self, tools, monkeypatch):
        import ana.memory as memory
        calls = []

        def fake_post(path, body):
            calls.append((path, body))
            return {"id": 7, "status": "saved"}

        monkeypatch.setattr(memory, "_get", lambda *a, **k: [])  # no match
        monkeypatch.setattr(memory, "_post", fake_post)
        out = tools.call("memory_save", {"title": "coffee",
                                         "content": "likes flat white",
                                         "type": "preference"})
        assert calls[0][0] == "/sessions"
        assert calls[1][1]["title"] == "coffee"
        assert calls[1][1]["type"] == "preference"
        assert calls[1][1]["project"] == "ana-asistente"
        assert "coffee" in out

    def test_save_without_title_or_content_says_so(self, tools, monkeypatch):
        import ana.memory as memory
        monkeypatch.setattr(memory, "_post",
                            lambda *a, **k: pytest.fail("nothing to save"))
        out = tools.call("memory_save", {"title": "", "content": ""})
        assert isinstance(out, str)
        assert "title" in out and "content" in out

    def test_save_degrades_when_engram_is_down(self, tools, monkeypatch):
        import ana.memory as memory

        def down(path, body):
            raise memory.MemoryUnavailable("engram down")

        # resolution succeeds (no match) so the failure is the POST's
        monkeypatch.setattr(memory, "_get", lambda *a, **k: [])
        monkeypatch.setattr(memory, "_post", down)
        out = tools.call("memory_save", {"title": "t", "content": "c"})
        assert isinstance(out, str)               # a note, never an exception
        assert "memoria no disponible" in out

    def test_search_returns_formatted_hits(self, tools, monkeypatch):
        import ana.memory as memory

        def fake_get(path, **params):
            return [{"type": "fact", "title": "cat",
                     "content": "has a cat named Momo"}]

        monkeypatch.setattr(memory, "_get", fake_get)
        out = tools.call("memory_search", {"query": "pet"})
        assert out == "- [fact] **cat**: has a cat named Momo"

    def test_search_degrades_when_engram_is_down(self, tools, monkeypatch):
        import ana.memory as memory

        def down(path, **params):
            raise memory.MemoryUnavailable("engram down")

        monkeypatch.setattr(memory, "_get", down)
        out = tools.call("memory_search", {"query": "pet"})
        assert isinstance(out, str)
        assert "memoria no disponible" in out

    def test_disabled_memory_returns_one_line_note(self, tools, monkeypatch):
        monkeypatch.setenv("ANA_MEMORY", "0")
        for name, args in (("memory_save", {"title": "t", "content": "c"}),
                           ("memory_search", {"query": "q"})):
            out = tools.call(name, args)
            assert out.count("\n") == 0
            assert "memoria deshabilitada" in out


class TestTimezoneDefault:
    def test_ana_timezone_env_overrides_the_system_zone(self, monkeypatch,
                                                         tools):
        monkeypatch.setenv("ANA_TIMEZONE", "Pacific/Apia")
        out = tools.call("current_time", {})
        assert "+13" in out.splitlines()[0]  # UTC+13, unambiguous

    def test_without_env_uses_the_machine_zone(self, monkeypatch, tools):
        monkeypatch.delenv("ANA_TIMEZONE", raising=False)
        out = tools.call("current_time", {})
        first = out.splitlines()[0]
        assert "T" in first       # ISO line stays machine-readable
