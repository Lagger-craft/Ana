"""Tests for ana.tools — Brave web search, hyprctl read-only, system info."""

import sys

import pytest


@pytest.fixture
def tools(monkeypatch):
    import ana.tools as tools_mod
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
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
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        result = tools.call("web_search", {"query": "hyprland"})
        assert "BRAVE_API_KEY" in result
        assert "unavailable" in result

    def test_brave_query_maps_results(self, tools, monkeypatch):
        fake = type("Resp", (), {})()
        fake.json = lambda: {"web": {"results": [
            {"title": "Hyprland", "url": "https://hypr.land", "description": "compositor"},
            {"title": "Omarchy", "url": "https://omarchy.org", "description": None},
        ]}}
        fake.raise_for_status = lambda: None
        statuses = []
        monkeypatch.setenv("BRAVE_API_KEY", "brave-key")

        def fake_get(url, params=None, headers=None, timeout=None):
            statuses.append((url, params, headers))
            return fake

        import ana.tools as tools_mod
        monkeypatch.setattr(tools_mod.httpx, "get", fake_get)
        result = tools.call("web_search", {"query": "wayland compositor", "count": 2})
        assert "1. Hyprland" in result
        assert "https://hypr.land" in result
        assert "2. Omarchy" in result
        assert "[no description]" in result
        url, params, headers = statuses[0]
        assert url == "https://api.search.brave.com/res/v1/web/search"
        assert params["q"] == "wayland compositor"
        assert params["count"] == 2
        assert headers.get("X-Subscription-Token") == "brave-key"

    def test_brave_error_degrades_to_message(self, tools, monkeypatch):
        class Fail(Exception):
            pass

        def fake_get(*a, **kw):
            raise Fail("boom")

        monkeypatch.setenv("BRAVE_API_KEY", "k")
        import ana.tools as tools_mod
        monkeypatch.setattr(tools_mod.httpx, "get", fake_get)
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
            "hypr_monitors", "hypr_version", "hypr_devices", "system_info"}
