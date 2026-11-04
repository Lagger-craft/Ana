"""Tests for ana.client — chat-completions SSE streaming and Anthropic Messages."""

import http.server
import json
import socket
import threading

import pytest

from ana.client import ProviderClient, ProviderError, probe_fallback
from ana.config import Config, Provider


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

    def test_sends_custom_provider_headers(self, chat_server):
        client = ProviderClient(
            base_url=f"http://127.0.0.1:{chat_server.port}",
            api_style="chat", api_key="sk-test",
            provider_headers={"x-opencode-session": "ana-1"})
        list(client.stream_chat(
            [{"role": "user", "content": "hi"}], model="m1"))
        assert chat_server.last_headers["x-opencode-session"] == "ana-1"
        assert chat_server.last_headers["Authorization"] == "Bearer sk-test"

    def test_tolerates_empty_choices_chunks(self):
        # mimo emits usage/ping chunks with an explicit empty choices list
        body = _sse_chunks(
            {"choices": []},
            _delta("hola"),
            {"choices": [{"finish_reason": "stop", "delta": {}}]})
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat")
            assert list(client.stream_chat(
                [{"role": "user", "content": "hi"}], model="m1")) == ["hola"]
        finally:
            server.stop()

    def test_tolerates_explicit_null_tool_calls(self):
        # mimo (opencode-go) sends "tool_calls": null on plain text deltas
        body = _sse_chunks(
            {"choices": [{"delta": {"content": "hola",
                                    "tool_calls": None}}]},
            {"choices": [{"finish_reason": "stop", "delta": {}}]})
        server = FakeServer(lambda p: (200, "text/event-stream", body, {}))
        try:
            client = ProviderClient(base_url=f"http://127.0.0.1:{server.port}",
                                    api_style="chat")
            assert list(client.stream_chat(
                [{"role": "user", "content": "hi"}], model="m1")) == ["hola"]
        finally:
            server.stop()

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


# --------------------------------------------------- multimodal (T20)


def _img(media_type="image/png", data="QUJD"):
    return {"media_type": media_type, "data": data}


class TestMultimodalImages:
    def test_chat_request_builds_image_url_parts_and_strips_images_key(
            self, chat_server):
        client = make_chat_client(chat_server)
        messages = [{"role": "user", "content": "mira",
                     "images": [_img()]}]
        list(client.stream_chat(messages, model="m1"))
        sent = json.loads(chat_server.last_body)
        assert sent["stream"] is True  # control: normal request succeeded
        msg = sent["messages"][0]
        assert "images" not in msg  # provider-facing key is stripped
        assert msg["content"][0] == {"type": "text", "text": "mira"}
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"] == {
            "url": "data:image/png;base64,QUJD"}

    def test_chat_request_keeps_plain_messages_untouched(self, chat_server):
        client = make_chat_client(chat_server)
        list(client.stream_chat([{"role": "user", "content": "hola"}],
                                model="m1"))
        assert json.loads(chat_server.last_body)["messages"] == [
            {"role": "user", "content": "hola"}]

    def test_anthropic_request_builds_base64_source_parts(self):
        body = _anthropic_chunks("ba", "ba")
        server = FakeServer(lambda p: (
            200, "text/event-stream", body,
            {}) if p == "/messages" else (404, "t", b"", {}))
        try:
            client = ProviderClient(
                base_url=f"http://127.0.0.1:{server.port}",
                api_style="anthropic", api_key="sk-ant")
            messages = [{"role": "user", "content": "mira",
                         "images": [_img(media_type="image/jpeg",
                                         data="WFla")]}]
            list(client.stream_chat(messages, model="m1"))
            sent = json.loads(server.last_body)
        finally:
            server.stop()
        msg = sent["messages"][0]
        assert msg["content"][0] == {"type": "text", "text": "mira"}
        assert msg["content"][1]["type"] == "image"
        assert msg["content"][1]["source"] == {
            "type": "base64", "media_type": "image/jpeg", "data": "WFla"}


class TestReasoningDeltas:
    def test_chat_reasoning_delta_yields_reasoning_event_and_not_text(self):
        body = _sse_chunks(
            {"choices": [{"delta": {"content": "", "reasoning": "hmm"}}]},
            {"choices": [{"delta": {"reasoning": "ora"}}]},
            _delta("p evidently"),
        )
        server = FakeServer(lambda p: (
            200, "text/event-stream", body,
            {}) if p == "/chat/completions" else (404, "t", b"", {}))
        try:
            client = make_chat_client(server)
            events = list(client.stream_turn(
                [{"role": "user", "content": "hi"}], model="m1"))
            assert [e["type"] for e in events] == [
                "reasoning", "reasoning", "text", "finish"]
            assert events[0]["content"] == "hmm"
            assert events[2]["content"] == "p evidently"
        finally:
            server.stop()

    def test_chat_reasoning_content_field_also_yields_events(self):
        body = _sse_chunks(
            {"choices": [{"delta": {"reasoning_content": "pienso"}}]},
        )
        server = FakeServer(lambda p: (
            200, "text/event-stream", body,
            {}) if p == "/chat/completions" else (404, "t", b"", {}))
        try:
            client = make_chat_client(server)
            events = list(client.stream_turn(
                [{"role": "user", "content": "hi"}], model="m1"))
            reasoning = [e for e in events if e["type"] == "reasoning"]
            assert reasoning == [{"type": "reasoning", "content": "pienso"}]
        finally:
            server.stop()

    def test_anthropic_thinking_delta_yields_reasoning_event(self):
        lines = [
            "event: content_block_start\ndata: " + json.dumps({
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking"},
            }) + "\n\n",
            "event: content_block_delta\ndata: " + json.dumps({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta",
                          "thinking": "dame un segundo"},
            }) + "\n\n",
            "event: content_block_stop\ndata: " + json.dumps(
                {"type": "content_block_stop", "index": 0}) + "\n\n",
            "event: content_block_start\ndata: " + json.dumps({
                "type": "content_block_start", "index": 1,
                "content_block": {"type": "text"},
            }) + "\n\n",
            "event: content_block_delta\ndata: " + json.dumps({
                "type": "content_block_delta", "index": 1,
                "delta": {"type": "text_delta", "text": "respuesta"},
            }) + "\n\n",
        ]
        body = ("".join(lines) + "event: message_stop\n"
                'data: {"type":"message_stop"}\n\n').encode()
        server = FakeServer(lambda p: (
            200, "text/event-stream", body,
            {}) if p == "/messages" else (404, "t", b"", {}))
        try:
            client = ProviderClient(
                base_url=f"http://127.0.0.1:{server.port}",
                api_style="anthropic", api_key="k")
            events = list(client.stream_turn(
                [{"role": "user", "content": "hi"}], model="m1"))
            kinds = [e["type"] for e in events if e["type"] != "finish"]
            assert kinds[0] == "reasoning"
            assert kinds[1] == "text"
        finally:
            server.stop()


# --------------------------------------------- speed levers (T23)


class TestPayloadSpeedFields:
    def test_keep_alive_added_when_provider_sets_it(self, chat_server):
        client = make_chat_client(chat_server)
        client.keep_alive = "1h"
        list(client.stream_chat([{"role": "user", "content": "hi"}], "m1"))
        assert json.loads(chat_server.last_body)["keep_alive"] == "1h"

    def test_vision_max_tokens_only_applies_to_image_turns(self,
                                                           chat_server):
        client = make_chat_client(chat_server)
        client.vision_max_tokens = 1024

        # no images: no cap
        list(client.stream_chat([{"role": "user", "content": "hi"}], "m1"))
        assert json.loads(chat_server.last_body) == {
            "model": "m1", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}

        # images: capped (probed: 256 empties answers, 1024 keeps stops)
        list(client.stream_chat(
            [{"role": "user", "content": "mira", "images": [_img()]}], "m1"))
        body = json.loads(chat_server.last_body)
        assert body["max_tokens"] == 1024
        # keep_alive not set by default on this client
        assert "keep_alive" not in body


# ------------------------------------------- free-tier startup probe


def _probe_config(base_url: str) -> Config:
    """Minimal config pointing one provider at ``base_url``."""
    return Config(default_provider="p", providers={"p": Provider(
        name="p", base_url=base_url, default_model="m1")})


class TestFreeTierProbe:
    """probe_fallback: classify a tiny request against the free endpoint.

    Fallback reason = dead free tier (402/403, funds, quota...);
    None = healthy OR transient (timeouts, connection errors, 5xx must
    never switch the session off its provider).
    """

    @staticmethod
    def _json_server(status: int, message: str):
        body = json.dumps({"error": {"message": message}}).encode()
        return FakeServer(lambda p: (status, "application/json", body, {}))

    def test_http_402_insufficient_funds_returns_reason(self):
        server = self._json_server(402, "Insufficient account funds")
        try:
            reason = probe_fallback(
                _probe_config(f"http://127.0.0.1:{server.port}"), "p")
        finally:
            server.stop()
        assert reason is not None
        assert "402" in reason and "Insufficient" in reason

    def test_http_400_unavailable_model_returns_reason(self):
        server = self._json_server(400, "Model is unavailable")
        try:
            reason = probe_fallback(
                _probe_config(f"http://127.0.0.1:{server.port}"), "p")
        finally:
            server.stop()
        assert reason is not None
        assert "unavailable" in reason.lower()

    def test_healthy_stream_returns_none(self):
        body = _sse_chunks(_delta("hi"))
        server = FakeServer(lambda p: (
            200, "text/event-stream", body, {}))
        try:
            reason = probe_fallback(
                _probe_config(f"http://127.0.0.1:{server.port}"), "p")
        finally:
            server.stop()
        assert reason is None  # stream started -> healthy, no switch

    def test_connection_refused_returns_none(self):
        server = FakeServer(lambda p: (200, "text/event-stream", b"", {}))
        server.stop()  # closed port -> connection refused
        reason = probe_fallback(
            _probe_config(f"http://127.0.0.1:{server.port}"), "p")
        assert reason is None  # transient: must NOT switch providers

    def test_timeout_returns_none(self):
        # listener that completes the handshake but never answers:
        # the short probe timeout classifies as transient, not fallback
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        try:
            cfg = _probe_config(f"http://127.0.0.1:{srv.getsockname()[1]}")
            reason = probe_fallback(cfg, "p", timeout=0.3)
        finally:
            srv.close()
        assert reason is None  # transient: must NOT switch providers

    def test_http_5xx_returns_none_even_when_body_says_unavailable(self):
        # 5xx is transient even though "unavailable" is a fallback
        # keyword: a flaky backend must not silently move Ana off her
        # provider
        server = self._json_server(503, "service unavailable, retry")
        try:
            reason = probe_fallback(
                _probe_config(f"http://127.0.0.1:{server.port}"), "p")
        finally:
            server.stop()
        assert reason is None
