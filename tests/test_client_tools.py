"""Tests for ana.client tool-calling: structured events in both dialects.

stream_turn() yields event dicts:
  {"type": "text", "content": str}
  {"type": "tool_call", "id": str, "name": str, "arguments": dict}  (after arg deltas)
  {"type": "finish", "finish_reason": str}  (final event)
"""

import json

import pytest

from ana.client import ProviderClient
from tests.test_client import FakeServer, _sse_chunks, _delta, make_chat_client

TOOLS_SPEC = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search the web",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        },
    },
}]


class TestChatToolCalls:
    def test_tool_call_deltas_accumulate_args(self):
        # split arguments JSON across two SSE chunks, as real providers do
        body = _sse_chunks(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "type": "function",
                 "function": {"name": "web_search", "arguments": '{"q": "arch'}},
            ]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ' linux"}'}},
            ]}}]},
            {"choices": [{"finish_reason": "tool_calls", "delta": {}}]},
        )
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = make_chat_client(server)
            events = list(client.stream_turn(
                [{"role": "user", "content": "hola"}], model="m1",
                tools=TOOLS_SPEC))
            calls = [e for e in events if e["type"] == "tool_call"]
            finishes = [e for e in events if e["type"] == "finish"]
            assert calls == [{"type": "tool_call", "id": "c1",
                              "name": "web_search", "arguments": {"q": "arch linux"}}]
            assert finishes and finishes[-1]["finish_reason"] == "tool_calls"
            assert not any(e["type"] == "text" for e in events)
            # tools were sent to the provider
            sent = json.loads(server.last_body)
            assert sent["tools"] == TOOLS_SPEC
        finally:
            server.stop()

    def test_plain_turn_still_yields_text_and_stop_finish(self):
        body = _sse_chunks(_delta("hola"), _delta("!"),
                           {"choices": [{"finish_reason": "stop", "delta": {}}]})
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = make_chat_client(server)
            events = list(client.stream_turn(
                [{"role": "user", "content": "hola"}], model="m1"))
            assert [e["type"] for e in events] == ["text", "text", "finish"]
            assert events[-1]["finish_reason"] == "stop"
        finally:
            server.stop()

    def test_no_tools_header_when_tools_absent(self):
        body = _sse_chunks(_delta("hi"))
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = make_chat_client(server)
            list(client.stream_turn([{"role": "user", "content": "hi"}], model="m1"))
            sent = json.loads(server.last_body)
            assert "tools" not in sent
        finally:
            server.stop()


ANTHROPIC_TOOLS = [{
    "name": "web_search",
    "description": "search the web",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}]




class TestAnthropicToolCalls:
    def test_tool_use_block_yields_structured_event(self):
        server = FakeServer(lambda p: (200, "text/event-stream",
                                       _anthropic_tool_body(), {}))
        try:
            client = ProviderClient(
                base_url=f"http://127.0.0.1:{server.port}",
                api_style="anthropic", api_key="ak")
            events = list(client.stream_turn(
                [{"role": "user", "content": "hola"}], model="claude",
                tools=ANTHROPIC_TOOLS))
            calls = [e for e in events if e["type"] == "tool_call"]
            finishes = [e for e in events if e["type"] == "finish"]
            assert calls == [{"type": "tool_call", "id": "tu_1",
                              "name": "web_search", "arguments": {"q": "arch linux"}}]
            assert finishes and finishes[-1]["finish_reason"] == "tool_use"
            sent = json.loads(server.last_body)
            assert sent["tools"] == ANTHROPIC_TOOLS
        finally:
            server.stop()


def _anthropic_tool_body():
    """Anthropic SSE with one tool_use block; args split across 2 deltas."""
    start = {
        "type": "content_block_start", "index": 1,
        "content_block": {"type": "tool_use", "id": "tu_1",
                          "name": "web_search", "input": {}},
    }
    d1 = {"type": "content_block_delta", "index": 1,
          "delta": {"type": "input_json_delta", "partial_json": '{"q": "arch'}}
    d2 = {"type": "content_block_delta", "index": 1,
          "delta": {"type": "input_json_delta", "partial_json": ' linux"}'}}
    stop = {"type": "content_block_stop", "index": 1}
    delta_msg = {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}
    parts = [
        'event: message_start\ndata: {"type":"message_start"}\n\n',
        'event: content_block_start\ndata: ' + json.dumps(start) + '\n\n',
        'event: content_block_delta\ndata: ' + json.dumps(d1) + '\n\n',
        'event: content_block_delta\ndata: ' + json.dumps(d2) + '\n\n',
        'event: content_block_stop\ndata: ' + json.dumps(stop) + '\n\n',
        'event: message_delta\ndata: ' + json.dumps(delta_msg) + '\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    return "".join(parts).encode()
