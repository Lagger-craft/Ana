"""Tests for ana.memory — Engram-backed memory with graceful degradation.

Every test fakes the HTTP seam (`_get` / `_post`); the real transport
also refuses to run under pytest, so the suite can never reach the live
Engram server on this machine.
"""

from pathlib import Path

import pytest

import ana.memory as memory
from ana.config import load_config, write_template


@pytest.fixture(autouse=True)
def fresh_memory(monkeypatch):
    """Per-process session state must not leak between tests."""
    monkeypatch.setattr(memory, "_session_id", None)
    monkeypatch.setattr(memory, "_session_registered", False)
    monkeypatch.setattr(memory, "_context_cache", None)
    monkeypatch.delenv("ANA_MEMORY", raising=False)
    monkeypatch.delenv("ANA_MEMORY_PROJECT", raising=False)
    monkeypatch.delenv("ANA_ENGRAM_URL", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.toml"
    write_template(path)
    return path


def _fake_post(calls, error=None):
    """Recording replacement for the `_post` transport seam."""
    def post(path, body):
        calls.append((path, body))
        if error is not None:
            raise error
        return {"id": 1, "status": "saved"}
    return post


def _fake_get(calls, answer, error=None):
    """Recording replacement for the `_get` transport seam."""
    def get(path, **params):
        calls.append((path, params))
        if error is not None:
            raise error
        return answer
    return get


# ------------------------------------------------------------- save (S1)


class TestSave:
    def test_payload_and_session_registered_exactly_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(memory, "_post", _fake_post(calls))
        first = memory.save("comida favorita", "prefiere la pizza",
                            type="preference")
        second = memory.save("mascota", "tiene un gato llamado Momo")
        assert [path for path, _ in calls] == [
            "/sessions", "/observations", "/observations"]
        session = calls[0][1]
        assert session["project"] == "ana-asistente"
        assert session["ownership_mode"] == "project_owned"
        assert session["directory"] == str(
            Path.home() / ".local" / "share" / "ana")
        assert session["id"].startswith("ana-")
        assert calls[1][1] == {
            "session_id": session["id"],
            "title": "comida favorita",
            "content": "prefiere la pizza",
            "type": "preference",
            "project": "ana-asistente",
            "scope": "project",
        }
        assert calls[2][1]["type"] == "fact"      # default kind
        assert "comida favorita" in first         # short confirmation
        assert isinstance(second, str)

    def test_registration_failure_raises_and_writes_nothing(self, monkeypatch):
        calls = []
        monkeypatch.setattr(memory, "_post", _fake_post(
            calls, error=memory.MemoryUnavailable("engram down")))
        with pytest.raises(memory.MemoryUnavailable):
            memory.save("t", "c")
        assert [path for path, _ in calls] == ["/sessions"]
        assert memory._session_registered is False  # retried on the next save

    def test_project_env_overrides_the_namespace(self, monkeypatch):
        monkeypatch.setenv("ANA_MEMORY_PROJECT", "otro-espacio")
        calls = []
        monkeypatch.setattr(memory, "_post", _fake_post(calls))
        memory.save("t", "c")
        assert calls[0][1]["project"] == "otro-espacio"
        assert calls[1][1]["project"] == "otro-espacio"


# ---------------------------------------------------------- shutdown (S1)


class TestShutdown:
    def test_shutdown_ends_the_session_once_best_effort(self, monkeypatch):
        calls = []
        monkeypatch.setattr(memory, "_post", _fake_post(calls))
        memory.save("t", "c")
        session_id = calls[0][1]["id"]
        memory.shutdown()
        assert calls[-1] == (f"/sessions/{session_id}/end", {"summary": ""})
        memory.shutdown()                       # already ended: no repeat
        assert len(calls) == 3

    def test_shutdown_without_session_posts_nothing(self, monkeypatch):
        calls = []
        monkeypatch.setattr(memory, "_post", _fake_post(calls))
        memory.shutdown()
        assert calls == []

    def test_shutdown_swallows_errors(self, monkeypatch):
        monkeypatch.setattr(memory, "_session_id", "ana-x")
        monkeypatch.setattr(memory, "_session_registered", True)

        def boom(path, body):
            raise RuntimeError("window is closing")

        monkeypatch.setattr(memory, "_post", boom)
        memory.shutdown()                        # best-effort: never raises


# ------------------------------------------------------------ search (S1)


class TestSearch:
    def test_formats_hits_as_markdown_lines(self, monkeypatch):
        calls = []
        answer = [
            {"type": "preference", "title": "comida",
             "content": "prefiere la pizza", "rank": 0.87},
            {"type": "fact", "title": "mascota",
             "content": "tiene un gato llamado Momo"},
        ]
        monkeypatch.setattr(memory, "_get", _fake_get(calls, answer))
        out = memory.search("comida")
        assert out.splitlines() == [
            "- [preference] **comida**: prefiere la pizza",
            "- [fact] **mascota**: tiene un gato llamado Momo",
        ]
        assert calls == [("/search", {"q": "comida",
                                      "project": "ana-asistente",
                                      "limit": 5})]

    def test_no_hits_is_an_honest_message_and_read_only(self, monkeypatch):
        post_calls = []
        monkeypatch.setattr(memory, "_get", _fake_get([], []))
        monkeypatch.setattr(memory, "_post", _fake_post(post_calls))
        out = memory.search("nada raro")
        assert "no encontré recuerdos" in out
        assert post_calls == []      # search never writes or registers


# ------------------------------------------------------- session context (S1)


class TestSessionContext:
    def test_fetched_once_and_cached_per_process(self, monkeypatch):
        calls = []
        answer = {"context": "## Memory from Previous Sessions\n\n- hola"}
        monkeypatch.setattr(memory, "_get", _fake_get(calls, answer))
        assert memory.session_context().startswith("## Memory")
        assert memory.session_context().startswith("## Memory")
        assert calls == [("/context", {"project": "ana-asistente",
                                       "max_bytes": 2000})]

    def test_unavailable_context_is_empty_and_attempted_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(memory, "_get", _fake_get(
            calls, None, error=memory.MemoryUnavailable("engram down")))
        assert memory.session_context() == ""
        assert memory.session_context() == ""   # cached: one attempt only
        assert len(calls) == 1


# ----------------------------------------------------------- disabled (S1)


class TestDisabledWithAnaMemoryZero:
    def test_context_is_empty_without_fetching(self, monkeypatch):
        monkeypatch.setenv("ANA_MEMORY", "0")
        monkeypatch.setattr(memory, "_get",
                            lambda *a, **k: pytest.fail("no HTTP when off"))
        assert memory.session_context() == ""

    def test_save_and_search_return_a_one_line_note(self, monkeypatch):
        monkeypatch.setenv("ANA_MEMORY", "0")
        monkeypatch.setattr(memory, "_post",
                            lambda *a, **k: pytest.fail("no HTTP when off"))
        monkeypatch.setattr(memory, "_get",
                            lambda *a, **k: pytest.fail("no HTTP when off"))
        for out in (memory.save("t", "c"), memory.search("q")):
            assert isinstance(out, str)
            assert out.count("\n") == 0                # one line
            assert "memoria deshabilitada" in out


# ----------------------------------------------------------- transport (S1)


class TestTransport:
    """The REAL `_get`/`_post` with httpx faked: wrap, never leak."""

    def test_post_httpx_errors_become_memoryunavailable(self, monkeypatch):
        import httpx as httpx_mod
        monkeypatch.setattr(memory, "_in_pytest", lambda: False)

        def boom(*a, **k):
            raise httpx_mod.ConnectError("connection refused")

        monkeypatch.setattr(memory.httpx, "post", boom)
        with pytest.raises(memory.MemoryUnavailable) as excinfo:
            memory.save("t", "c")
        assert not isinstance(excinfo.value, httpx_mod.HTTPError)
        assert "ConnectError" in str(excinfo.value)

    def test_get_httpx_errors_become_memoryunavailable(self, monkeypatch):
        import httpx as httpx_mod
        monkeypatch.setattr(memory, "_in_pytest", lambda: False)

        def boom(*a, **k):
            raise httpx_mod.ConnectTimeout("too slow")

        monkeypatch.setattr(memory.httpx, "get", boom)
        with pytest.raises(memory.MemoryUnavailable) as excinfo:
            memory.search("q")
        assert "ConnectTimeout" in str(excinfo.value)

    def test_transport_refuses_the_network_under_pytest(self):
        """No test may ever reach the live server: the guard fires first."""
        with pytest.raises(memory.MemoryUnavailable):
            memory._get("/context", project="ana-asistente")
        with pytest.raises(memory.MemoryUnavailable):
            memory._post("/observations", {"title": "x"})


# --------------------------------------------- system prompt injection (S3)


class _FakeClient:
    """Turn-level fake recording the system prompt it was sent."""

    api_style = "chat"

    def __init__(self):
        self.calls = []

    def stream_turn(self, messages, model, system=None, tools=None):
        self.calls.append({"system": system, "model": model})
        return iter([{"type": "text", "content": "ok"},
                     {"type": "finish", "finish_reason": "stop"}])


class TestSystemPromptInjection:
    def test_repl_prepends_context_to_the_system_prompt(self, cfg,
                                                        monkeypatch):
        from ana.cli import run_repl
        monkeypatch.setattr(memory, "session_context", lambda: "CTX-FAKE")
        client = _FakeClient()
        run_repl(cfg_path=cfg, lines=["hola"], tools=False, client_fx=client)
        base = load_config(cfg).system_prompt
        assert client.calls[0]["system"] == f"CTX-FAKE\n\n{base}"

    def test_one_shot_prepends_context(self, cfg, monkeypatch, capsys):
        from ana.cli import main
        monkeypatch.setattr(memory, "session_context", lambda: "CTX-FAKE")
        seen = {}

        def fake_build(cfg_arg, provider=None):
            def stream(messages, model, system=None):
                seen["system"] = system
                return iter(["hi"])
            return stream

        monkeypatch.setattr("ana.cli.build_stream", fake_build)
        assert main(["--config", str(cfg), "-m", "hola", "--no-tools"]) == 0
        base = load_config(cfg).system_prompt
        assert seen["system"] == f"CTX-FAKE\n\n{base}"

    def test_memory_off_keeps_the_prompt_untouched(self, cfg, monkeypatch):
        from ana.cli import run_repl
        monkeypatch.setenv("ANA_MEMORY", "0")
        monkeypatch.setattr(memory, "_get",
                            lambda *a, **k: pytest.fail("no HTTP when off"))
        client = _FakeClient()
        run_repl(cfg_path=cfg, lines=["hola"], tools=False, client_fx=client)
        assert client.calls[0]["system"] == load_config(cfg).system_prompt


# -------------------------------------------------- shutdown wiring (S4)


class TestShutdownWiring:
    def test_repl_exit_shuts_down_memory(self, cfg, monkeypatch):
        from ana.cli import run_repl
        calls = []
        monkeypatch.setattr(memory, "shutdown",
                            lambda: calls.append("shutdown"))
        run_repl(cfg_path=cfg, lines=["/exit"])
        assert calls == ["shutdown"]

    def test_tui_exit_shuts_down_memory(self, cfg, monkeypatch):
        import ana.tui as tui_mod
        calls = []
        monkeypatch.setattr(memory, "shutdown",
                            lambda: calls.append("shutdown"))

        class FakeApp:
            def __init__(self, *args, **kw):
                pass

            def run(self):
                return None

        monkeypatch.setattr(tui_mod, "AnaApp", FakeApp)
        tui_mod.run_tui(str(cfg))
        assert calls == ["shutdown"]
