"""Tests for ana.client — chat-completions SSE streaming and Anthropic Messages."""

import http.server
import json
import threading

import pytest

from ana.client import ProviderClient, ProviderError


def _sse_chunks(*payloads, done=True):
    """Build an SSE body from OpenAI-style delta payloads."""
    lines = []
    for p in payloads:
        lines.append("data: " + json.dumps(p) + "\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _delta(text):
    return {"choices": [{"delta": {"content": text}}]}


def _anthropic_chunks(*texts):
    """Anthropic Messages SSE body for one assistant turn."""
    lines = [
        'event: message_start\ndata: {"type":"message_start"}\n\n'
    ]
    for t in texts:
        lines.append(
            "event: content_block_delta\n"
            "data: " + json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": t},
            }) + "\n\n")
    lines.append('event: message_stop\ndata: {"type":"message_stop"}\n\n')
    return "".join(lines).encode()


class FakeServer:
    """Threaded HTTP server with one canned SSE response per request.

    route(path) -> (status, content_type, body_bytes, request_headers)
    """

    def __init__(self, route):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                status, ctype, body, _ = route(self.path)
                outer.last_body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.last_headers = dict(self.headers)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.last_body = None
        self.last_headers = {}

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def chat_server():
    body = _sse_chunks(_delta("hol"), _delta("a mundo"))
    server = FakeServer(lambda p: (
        200, "text/event-stream",
        body if p == "/chat/completions" else b'{"error":"not found"}', {}))
    yield server
    server.stop()


def make_chat_client(server):
    return ProviderClient(
        base_url=f"http://127.0.0.1:{server.port}",
        api_style="chat", api_key="sk-test")


class TestChatCompletions:
    def test_streams_text_deltas_in_order(self, chat_server):
        client = make_chat_client(chat_server)
        chunks = list(client.stream_chat(
            [{"role": "user", "content": "hi"}], model="m1"))
        assert chunks == ["hol", "a mundo"]

    def test_sends_bearer_key_and_payload(self, chat_server):
        client = make_chat_client(chat_server)
        list(client.stream_chat([{"role": "user", "content": "hi"}], model="m1"))
        assert chat_server.last_headers["Authorization"] == "Bearer sk-test"
        sent = json.loads(chat_server.last_body)
        assert sent["model"] == "m1"
        assert sent["messages"] == [{"role": "user", "content": "hi"}]
        assert sent["stream"] is True

    def test_tolerates_missing_done_marker(self):
        body = _sse_chunks(_delta("ok"), done=False)
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat")
            assert list(client.stream_chat(
                [{"role": "user", "content": "hi"}], model="m1")) == ["ok"]
        finally:
            server.stop()

    def test_ignores_non_content_deltas_and_sse_comments(self):
        body = _sse_chunks(
            {"choices": [{"delta": {"role": "assistant"}}]},
            _delta("hi"),
            {"choices": [{"finish_reason": "stop", "delta": {}}]},
        )
        body = b': keep-alive comment\n\n' + body
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat")
            assert list(client.stream_chat(
                [{"role": "user", "content": "hi"}], model="m1")) == ["hi"]
        finally:
            server.stop()

    def test_http_error_raises_provider_error_with_body(self):
        body = json.dumps({"error": {"message": "bad key"}}).encode()
        server = FakeServer(lambda p: (401, "application/json", body, {}))
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat", api_key="wrong")
            with pytest.raises(ProviderError, match="bad key"):
                list(client.stream_chat(
                    [{"role": "user", "content": "hi"}], model="m1"))
        finally:
            server.stop()

    def test_data_error_json_raises_provider_error(self):
        body = _sse_chunks({"error": {"message": "model overloaded"}})
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat")
            with pytest.raises(ProviderError, match="model overloaded"):
                list(client.stream_chat(
                    [{"role": "user", "content": "hi"}], model="m1"))
        finally:
            server.stop()


class TestAnthropic:
    def test_streams_text_deltas_and_uses_messages_api(self):
        body = _anthropic_chunks("ho", "la")
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="anthropic", api_key="ak-test",
                                    api_version="2023-06-01")
            chunks = list(client.stream_chat(
                [{"role": "user", "content": "hi"}], model="claude",
                system="you are nice"))
            assert chunks == ["ho", "la"]
            assert server.last_headers.get("x-api-key") == "ak-test"
            assert server.last_headers.get("anthropic-version") == "2023-06-01"
            sent = json.loads(server.last_body)
            assert sent["model"] == "claude"
            assert sent["messages"] == [{"role": "user", "content": "hi"}]
            assert sent["system"] == "you are nice"
            assert sent["stream"] is True
        finally:
            server.stop()

    def test_error_response_raises_provider_error(self):
        body = json.dumps({
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid key"},
        }).encode()
        server = FakeServer(lambda p: (401, "application/json", body, {}))
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="anthropic", api_key="bad")
            with pytest.raises(ProviderError, match="invalid key"):
                list(client.stream_chat(
                    [{"role": "user", "content": "hi"}], model="claude"))
        finally:
            server.stop()
