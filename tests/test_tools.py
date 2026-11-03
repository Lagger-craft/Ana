"""Tests for ana.tools — Tavily web search, hyprctl read-only, system info."""

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
            "current_time"}


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
