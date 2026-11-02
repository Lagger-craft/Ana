"""Tests for ana.agent — bounded tool-execution loop over stream_turn."""

import json

import pytest

from ana.agent import AgentError, Tool, ToolRegistry, run_agent
from ana.client import ProviderClient
from tests.test_client import FakeServer, _sse_chunks, _delta


def _chat_tool_call_chunks(name="echo_tool", args='{"text": "hola"}', id="c1"):
    tool_delta = {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": id, "type": "function",
         "function": {"name": name, "arguments": args}},
    ]}}]}
    finish = {"choices": [{"finish_reason": "tool_calls", "delta": {}}]}
    return _sse_chunks(tool_delta, finish)


def _chat_final_chunks(text="listo!"):
    return _sse_chunks(
        _delta(text),
        {"choices": [{"finish_reason": "stop", "delta": {}}]},
    )


class SeqServer:
    """FakeServer variant serving canned bodies in sequence per request."""

    def __init__(self, bodies):
        import http.server
        import threading

        seq = iter(bodies)
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = next(seq)
                outer.bodies_seen.append(body)
                outer.last_body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append(outer.last_body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.bodies_seen = []
        self.requests = []
        self.last_body = None

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestToolRegistry:
    def test_to_spec_chat_format(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="echo_tool", description="echo things",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda args: args["text"],
        ))
        spec = reg.to_spec("chat")
        assert spec[0]["type"] == "function"
        assert spec[0]["function"]["name"] == "echo_tool"
        assert spec[0]["function"]["parameters"]["properties"]["text"]["type"] == "string"

    def test_to_spec_anthropic_format(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="echo_tool", description="echo things", parameters={
                "type": "object", "properties": {"text": {"type": "string"}}},
            fn=lambda args: args["text"],
        ))
        spec = reg.to_spec("anthropic")
        assert spec[0]["name"] == "echo_tool"
        assert spec[0]["input_schema"]["properties"]["text"]["type"] == "string"

    def test_call_returns_result(self):
        reg = ToolRegistry()
        reg.register(Tool(name="double", description="d",
                          parameters={"type": "object", "properties": {"n": {}}},
                          fn=lambda args: args["n"] * 2))
        assert reg.call("double", {"n": 2}) == 4

    def test_call_unknown_tool_raises(self):
        reg = ToolRegistry()
        with pytest.raises(AgentError, match="unknown tool"):
            reg.call("nope", {})


class TestAgentLoopChat:
    def test_tool_round_then_final_answer(self):
        server = SeqServer([_chat_tool_call_chunks(), _chat_final_chunks()])
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat")
            reg = ToolRegistry()
            seen_calls = []
            reg.register(Tool(name="echo_tool", description="d",
                              parameters={"type": "object",
                                          "properties": {"text": {}}},
                              fn=lambda a: (seen_calls.append(a),
                                            a["text"])[1]))

            emitted = []
            history = run_agent(
                client, [{"role": "user", "content": "di hola"}], model="m1",
                registry=reg, emit=emitted.append, on_tool=lambda n, a: None)

            assert seen_calls == [{"text": "hola"}]
            assert "listo!" in history[-1]["content"]
            assert history[-1]["role"] == "assistant"
            # second request carried assistant tool_call + tool result msgs
            sent = json.loads(server.requests[1])
            roles = [m["role"] for m in sent["messages"]]
            assert roles == ["user", "assistant", "tool"]
            assert sent["messages"][1]["tool_calls"][0]["function"]["name"] == "echo_tool"
            assert sent["messages"][2]["content"] == "hola"
        finally:
            server.stop()

    def test_text_streamed_as_it_happens(self):
        server = SeqServer([_chat_final_chunks("hola mundo")])
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat")
            reg = ToolRegistry()
            emitted = []
            run_agent(client, [{"role": "user", "content": "hola"}],
                      model="m1", registry=reg, emit=emitted.append,
                      on_tool=lambda n, a: None)
            assert "".join(emitted) == "hola mundo"
        finally:
            server.stop()

    def test_round_limit_raises_agent_error(self):
        server = SeqServer([_chat_tool_call_chunks()] * 3)
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat")
            reg = ToolRegistry()
            reg.register(Tool(name="echo_tool", description="d",
                              parameters={"type": "object", "properties": {}},
                              fn=lambda a: "ok"))
            with pytest.raises(AgentError, match="tool rounds"):
                run_agent(client, [{"role": "user", "content": "x"}],
                          model="m1", registry=reg,
                          emit=lambda _: None, max_rounds=2)
        finally:
            server.stop()


class TestAgentLoopAnthropic:
    def test_tool_use_round_then_final(self):
        body_tools = "".join([
            'event: content_block_start\ndata: ' + json.dumps({
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "tool_use", "id": "tu_1",
                                  "name": "echo_tool", "input": {}}}) + "\n\n",
            'event: content_block_delta\ndata: ' + json.dumps({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "input_json_delta",
                          "partial_json": '{"text": "hola"}'}}) + "\n\n",
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
            'event: message_delta\ndata: ' + json.dumps({
                "type": "message_delta", "delta": {"stop_reason": "tool_use"}}) + "\n\n",
        ]).encode()
        body_final = "".join([
            'event: content_block_start\ndata: ' + json.dumps({
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}) + "\n\n",
            'event: content_block_delta\ndata: ' + json.dumps({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "listo!"}}) + "\n\n",
            'event: message_delta\ndata: ' + json.dumps({
                "type": "message_delta", "delta": {"stop_reason": "end_turn"}}) + "\n\n",
        ]).encode()

        bodies = iter([body_tools, body_final])
        seen_bodies = []

        def route(path):
            seen_bodies.append(None)
            return 200, "text/event-stream", next(bodies), {}

        server = FakeServer_with(path=route, bodies=seen_bodies)
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="anthropic", api_key="ak")
            reg = ToolRegistry()
            seen_calls = []
            reg.register(Tool(name="echo_tool", description="d",
                              parameters={"type": "object",
                                          "properties": {"text": {}}},
                              fn=lambda a: (seen_calls.append(a),
                                            a["text"])[1]))
            history = run_agent(
                client, [{"role": "user", "content": "di hola"}], model="claude",
                registry=reg, emit=lambda _: None)
            assert seen_calls == [{"text": "hola"}]
            assert history[-1]["role"] == "assistant"
            assert history[-1]["content"] == "listo!"
            # last request shows the tool_result user turn next to assistant blocks
            last = json.loads(server.last_body)
            roles = [m["role"] for m in last["messages"]]
            assert roles == ["user", "assistant", "user"]
            blocks = last["messages"][2]["content"]
            assert blocks[0]["type"] == "tool_result"
            assert blocks[0]["tool_use_id"] == "tu_1"
            assert blocks[0]["content"] == "hola"
        finally:
            server.stop()


def FakeServer_with(path, bodies=None):
    # thin helper: FakeServer from test_client expects route(path) tuple
    from tests.test_client import FakeServer as _FS
    return _FS(path)
