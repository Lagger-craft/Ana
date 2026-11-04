"""Streaming chat client for OpenAI-compatible and Anthropic APIs.

ProviderClient talks to one backend endpoint. Supported styles:

- ``chat``: OpenAI Chat Completions (OpenAI, OpenCode Zen, MiMo, Ollama
  local endpoints all accept this dialect).
- ``anthropic``: Anthropic Messages API with the natively different
  auth headers and SSE event names.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import httpx

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

# Startup free-tier probe: SHORT budget so a dead endpoint can't stall
# session startup (the whole probe is one tiny streaming request).
PROBE_TIMEOUT = 8.0


def _payload_messages(messages, anthropic: bool) -> list[dict]:
    """Provider-facing copy of the history, with images per dialect.

    A message like {role, content, images: [{media_type, data(base64)}]}
    becomes {role, content: [text, image parts]} for either dialect:
    chat-completions wants image_url data-URIs, Anthropic wants a base64
    source. The provider-facing copy never carries the internal `images`
    key; plain messages pass through untouched.
    """
    out: list[dict] = []
    for msg in messages:
        if "images" not in msg:
            out.append(msg)
            continue
        content: list[dict] = [
            {"type": "text", "text": str(msg.get("content", ""))}]
        for image in msg["images"]:
            mime = str(image.get("media_type", "image/png"))
            data = str(image.get("data", ""))
            if anthropic:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": mime, "data": data}})
            else:
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{data}"}})
        out.append({"role": msg.get("role", "user"), "content": content})
    return out



class ProviderError(Exception):
    """Raised when the provider returns an error response."""


def _error_detail(body: bytes, status: int) -> str:
    try:
        data = json.loads(body)
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str) and err:
            return err
        return body.decode("utf-8", "replace")[:200]
    except (json.JSONDecodeError, AttributeError):
        return body.decode("utf-8", "replace")[:200]


class ProviderClient:
    """Streams chat completions from one provider endpoint."""

    def __init__(
        self,
        base_url: str,
        api_style: str = "chat",
        api_key: str | None = None,
        api_version: str = DEFAULT_ANTHROPIC_VERSION,
        timeout: float = 120.0,
        keep_alive: str | None = None,
        vision_max_tokens: int | None = None,
        provider_headers: dict[str, str] | None = None,
    ) -> None:
        if api_style not in ("chat", "anthropic"):
            raise ValueError(f"unknown api_style: {api_style!r}")
        self.base_url = base_url.rstrip("/")
        self.api_style = api_style
        self.api_key = api_key
        self.api_version = api_version
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.vision_max_tokens = vision_max_tokens
        self.provider_headers = provider_headers or {}

    @classmethod
    def from_provider(cls, config, provider, **kwargs) -> "ProviderClient":
        return cls(
            base_url=provider.base_url,
            api_style=provider.api_style,
            api_key=config.resolve_key(provider),
            keep_alive=getattr(provider, "keep_alive", None),
            vision_max_tokens=getattr(provider, "vision_max_tokens", None),
            provider_headers=getattr(provider, "headers", None),
            **kwargs,
        )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        system: str | None = None,
    ) -> Iterator[str]:
        """Yield assistant text deltas in order."""
        for event in self.stream_turn(messages, model, system=system):
            if event["type"] == "text":
                yield event["content"]

    def stream_turn(
        self,
        messages: list[dict[str, str]],
        model: str,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[dict]:
        """Yield structured turn events (see tests/test_client_tools.py)."""
        if self.api_style == "anthropic":
            url, payload = self._anthropic_request(messages, model, system, tools)
        else:
            url, payload = self._chat_request(messages, model, system, tools)
        yield from self._open_stream_turn(url, payload)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self.provider_headers)
        if self.api_key is not None:
            if self.api_style == "anthropic":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = self.api_version
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _apply_speed_fields(self, payload: dict, has_images: bool) -> None:
        """keep_alive always; the vision token cap only on image turns.

        Probed on ollama /v1 (RX 7600): keep_alive extends the loaded
        model window without sudo; max_tokens caps thinking+answer and
        scales time ~2× at 1024, but low caps can empty the answer —
        so it stays opt-in per provider and per config.
        """
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if self.vision_max_tokens and has_images:
            payload["max_tokens"] = self.vision_max_tokens

    def _chat_request(
        self, messages: list[dict[str, str]], model: str, system: str | None,
        tools: list[dict] | None = None,
    ) -> tuple[str, dict]:
        has_images = any("images" in m for m in messages)
        payload_messages = _payload_messages(messages, anthropic=False)
        if system:
            payload_messages.insert(0, {"role": "system", "content": system})
        payload: dict = {
            "model": model, "messages": payload_messages, "stream": True,
        }
        self._apply_speed_fields(payload, has_images)
        if tools:
            payload["tools"] = tools
        return self.base_url + "/chat/completions", payload

    def _anthropic_request(
        self, messages: list[dict[str, str]], model: str, system: str | None,
        tools: list[dict] | None = None,
    ) -> tuple[str, dict]:
        has_images = any("images" in m for m in messages)
        payload_messages = _payload_messages(messages, anthropic=True)
        payload: dict = {"model": model, "messages": payload_messages,
                         "stream": True}
        self._apply_speed_fields(payload, has_images)
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        return self.base_url + "/messages", payload

    def _open_stream_turn(self, url: str, payload: dict) -> Iterator[dict]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST", url, json=payload, headers=self._headers(),
                ) as response:
                    if response.status_code >= 400:
                        body = response.read()
                        raise ProviderError(
                            f"provider returned HTTP {response.status_code}: "
                            f"{_error_detail(body, response.status_code)}")
                    if self.api_style == "anthropic":
                        yield from self._anthropic_turn(response)
                    else:
                        yield from self._chat_turn(response)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"cannot reach {url}: {exc.__class__.__name__}: {exc}") from exc


    def _parse_json_obj(self, raw: str) -> dict:
        """Parse tool-call argument JSON; fall back to an error marker."""
        try:
            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {"__value__": parsed}
        except json.JSONDecodeError:
            return {"__raw__": raw}

    def _chat_turn(self, response: httpx.Response) -> Iterator[dict]:
        """OpenAI Chat Completions SSE -> structured events, after full read."""
        pending: dict[int, dict] = {}
        finish_reason = "stop"
        for raw in response.iter_lines():
            line = raw.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:") or line.startswith("id:"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"malformed SSE chunk from provider: {exc}") from exc
            if "error" in event:
                err = event["error"]
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise ProviderError(f"provider error: {msg or event}")
            # mimo (opencode-go) can emit chunks with "choices": []
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                # reasoning models (qwen3*, deepseek...) stream thinking in a
                # separate field while content stays empty: surface it so the
                # caller can show progress instead of a frozen prompt.
                yield {"type": "reasoning", "content": str(reasoning)}
            text = delta.get("content")
            if text:
                yield {"type": "text", "content": text}
            # providers like mimo send "tool_calls": null explicitly
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        for idx in sorted(pending):
            slot = pending[idx]
            yield {
                "type": "tool_call", "id": slot["id"] or str(idx),
                "name": slot["name"],
                "arguments": self._parse_json_obj(slot["args"]),
            }
        yield {"type": "finish", "finish_reason": finish_reason}

    def _anthropic_turn(self, response: httpx.Response) -> Iterator[dict]:
        """Anthropic Messages SSE -> structured events, after full read."""
        blocks: dict[int, dict] = {}
        current_text: list[str] | None = None
        stop_reason = "end_turn"
        for raw in response.iter_lines():
            line = raw.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                continue
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"malformed SSE chunk from provider: {exc}") from exc
            etype = event.get("type")
            if etype == "error":
                err = event.get("error", {})
                raise ProviderError(f"provider error: {err.get('message', event)}")
            if etype == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "text":
                    current_text = []
                elif block.get("type") == "tool_use":
                    blocks[event.get("index", 0)] = {
                        "id": block.get("id", ""), "name": block.get("name", ""),
                        "args": "",
                    }
            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                dtype = delta.get("type")
                if dtype == "thinking_delta" and delta.get("thinking"):
                    yield {"type": "reasoning",
                           "content": str(delta["thinking"])}
                elif dtype == "text_delta" and delta.get("text"):
                    if current_text is None:  # no start event seen (lenient)
                        current_text = []
                    current_text.append(delta["text"])
                    yield {"type": "text", "content": delta["text"]}
                elif dtype == "input_json_delta" and delta.get("partial_json"):
                    slot = blocks.get(event.get("index", -1))
                    if slot is not None:
                        slot["args"] += delta["partial_json"]
            elif etype == "content_block_stop":
                if current_text is not None:
                    current_text = None
                idx = event.get("index", -1)
                slot = blocks.pop(idx, None)
                if slot is not None:
                    yield {
                        "type": "tool_call", "id": slot["id"],
                        "name": slot["name"],
                        "arguments": self._parse_json_obj(slot["args"]),
                    }
            elif etype == "message_delta":
                reason = event.get("delta", {}).get("stop_reason")
                if reason:
                    stop_reason = reason
        yield {"type": "finish", "finish_reason": stop_reason}


# ------------------------------------------- free-tier startup probe

# Messages that mean "the free tier is gone" (case-insensitive match).
_FALLBACK_MARKERS = ("HTTP 402", "HTTP 403", "insufficient", "funds",
                     "payment", "paid", "upgrade", "credit", "quota",
                     "unavailable", "not found")

# Transient failures win over the markers above: a flaky network or a
# 5xx blip (even a body saying "service unavailable") must never
# silently move the session off its provider.
_TRANSIENT_RE = re.compile(
    r"HTTP 5\d\d|timeout|timed out|cannot reach|connect|network", re.I)


def _classify_probe_error(message: str) -> str | None:
    """Short fallback reason for a probe error; None = no switch."""
    if _TRANSIENT_RE.search(message):
        return None
    if any(marker.lower() in message.lower() for marker in _FALLBACK_MARKERS):
        reason = message.removeprefix("provider returned ")
        return reason if len(reason) <= 120 else reason[:119] + "…"
    return None


def probe_fallback(cfg, provider_name: str,
                   timeout: float = PROBE_TIMEOUT) -> str | None:
    """Cheap startup health probe of one provider's free endpoint.

    Sends one tiny streaming chat request with a SHORT timeout and
    consumes at most the first event:

    - a dead free tier (HTTP 402/403, insufficient funds, payment,
      upgrade, credit, quota, unavailable, not found) returns a short
      fallback reason;
    - a healthy endpoint (200 + stream start) returns None;
    - transient failures (timeouts, connection errors, HTTP 5xx) also
      return None — a flaky network must not silently move the session
      off its provider.

    Blind spot: a model that keeps returning 200 while silently billing
    cannot be detected client-side.

    Tests must patch the ana.cli._free_tier_probe seam instead of ever
    reaching real endpoints; classification tests run against a local
    FakeServer only.
    """
    provider = cfg.provider(provider_name)
    if not provider.default_model:
        return None  # nothing to probe with: leave the decision alone
    try:
        client = ProviderClient.from_provider(cfg, provider,
                                              timeout=timeout)
        stream = client.stream_chat(
            [{"role": "user", "content": "ping"}],
            provider.default_model)
        for _ in stream:
            return None  # first event arrived: the stream is alive
        return None  # reachable (stream had no text): healthy
    except ProviderError as exc:
        return _classify_probe_error(str(exc))
