"""Tests for Phase 1 REPL integration: tool turns, /tools, graceful fallback."""

import pytest

from ana.agent import AgentError, Tool, ToolRegistry
from ana.client import ProviderError
from ana.cli import main, run_repl
from ana.config import write_template


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "config.toml"
    write_template(path)
    return path


class FakeTurnClient:
    """Records stream_turn calls; serves canned event lists in order."""

    api_style = "chat"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def stream_turn(self, messages, model, system=None, tools=None):
        self.calls.append({
            "system": system,
            "roles": [m["role"] for m in messages],
            "tools": tools is not None,
        })
        items = self.responses.pop(0)
        return iter(items)

    @property
    def last(self):
        return self.calls[-1]


def _tool_call_event(name="echo_tool", text="hola", id="c1"):
    return {"type": "tool_call", "id": id, "name": name, "arguments": {"text": text}}


def echo_registry():
    reg = ToolRegistry()
    reg.register(Tool(name="echo_tool", description="echo",
                      parameters={"type": "object", "properties": {"text": {}}},
                      fn=lambda a: a.get("text", "")))
    return reg


class TestToolTurn:
    def test_tool_call_runs_then_final_reply(self, cfg, capsys):
        client = FakeTurnClient([
            [_tool_call_event()],
            [{"type": "text", "content": "aca va la respuesta"},
             {"type": "finish", "finish_reason": "stop"}],
        ])
        history = run_repl(cfg_path=cfg, lines=["investiga"],
                           stream=None, tools=True, registry=echo_registry(),
                           client_fx=client)
        assert client.last["tools"] is True
        assert "echo_tool" in capsys.readouterr().out
        assert history[-1] == {"role": "assistant",
                               "content": "aca va la respuesta"}
        # round 1 sends only the user msg; round 2 sees tool traffic
        assert client.calls[0]["roles"] == ["user"]
        assert client.calls[1]["roles"] == ["user", "assistant", "tool"]

    def test_tools_disabled_skips_agent(self, cfg):
        client = FakeTurnClient([
            [{"type": "text", "content": "plain"},
             {"type": "finish", "finish_reason": "stop"}],
        ])
        history = run_repl(cfg_path=cfg, lines=["hola"], tools=False,
                           registry=echo_registry(), client_fx=client)
        assert client.calls[0]["tools"] is False
        assert history[-1]["content"] == "plain"

    def test_tools_command_toggles(self, cfg, capsys):
        client = FakeTurnClient([
            [{"type": "text", "content": "x"},
             {"type": "finish", "finish_reason": "stop"}],
        ])
        run_repl(cfg_path=cfg, lines=["/tools off", "hola"],
                 tools=True, registry=echo_registry(), client_fx=client)
        assert client.calls[0]["tools"] is False


class TestFallback:
    def test_tools_payload_rejected_falls_back_once(self, cfg, capsys):
        class Rejecting(FakeTurnClient):
            def stream_turn(self, messages, model, system=None, tools=None):
                self.calls.append({"tools": tools is not None})
                if tools is not None:
                    raise ProviderError(
                        "provider returned HTTP 400: tools unsupported by model")
                return iter([{"type": "text", "content": "sin tools"},
                             {"type": "finish", "finish_reason": "stop"}])

        client = Rejecting([])
        history = run_repl(cfg_path=cfg, lines=["hola", "otra vez"],
                           tools=True, registry=echo_registry(),
                           client_fx=client)
        out = capsys.readouterr().out
        assert "sin tools" in out
        assert "tools disabled" in out
        # turn 1: agent call (fail) + plain retry; turn 2 stays plain
        assert len(client.calls) == 3
        assert history[-1]["content"] == "sin tools"

    def test_agent_round_limit_rolls_back_and_reports(self, cfg, capsys):
        class AlwaysTool(FakeTurnClient):
            def stream_turn(self, messages, model, system=None, tools=None):
                self.calls.append({"tools": tools is not None})
                if tools is None:
                    return iter([{"type": "text", "content": "dead"},
                                 {"type": "finish", "finish_reason": "stop"}])
                return iter([_tool_call_event(),
                             {"type": "finish", "finish_reason": "tool_calls"}])

        client = AlwaysTool([])
        history = run_repl(cfg_path=cfg, lines=["q1"], tools=True,
                           registry=echo_registry(), client_fx=client,
                           max_rounds=2)
        out = capsys.readouterr().out
        assert "tool rounds" in out
        assert history == [] or all(m["content"] != "dead" for m in history)


def capsys_line():
    import io
    from contextlib import redirect_stdout
    return ""
