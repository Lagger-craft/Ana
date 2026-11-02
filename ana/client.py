"""Streaming chat client for OpenAI-compatible and Anthropic APIs.

ProviderClient talks to one backend endpoint. Supported styles:

- ``chat``: OpenAI Chat Completions (OpenAI, OpenCode Zen, MiMo, Ollama
  local endpoints all accept this dialect).
- ``anthropic``: Anthropic Messages API with the natively different
  auth headers and SSE event names.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


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
    ) -> None:
        if api_style not in ("chat", "anthropic"):
            raise ValueError(f"unknown api_style: {api_style!r}")
        self.base_url = base_url.rstrip("/")
        self.api_style = api_style
        self.api_key = api_key
        self.api_version = api_version
        self.timeout = timeout

    @classmethod
    def from_provider(cls, config, provider, **kwargs) -> "ProviderClient":
        return cls(
            base_url=provider.base_url,
            api_style=provider.api_style,
            api_key=config.resolve_key(provider),
            **kwargs,
        )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        system: str | None = None,
    ) -> Iterator[str]:
        """Yield assistant text deltas in order."""
        if self.api_style == "anthropic":
            url, payload = self._anthropic_request(messages, model, system)
        else:
            url, payload = self._chat_request(messages, model, system)
        yield from self._open_stream(url, payload)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            if self.api_style == "anthropic":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = self.api_version
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat_request(
        self, messages: list[dict[str, str]], model: str, system: str | None,
    ) -> tuple[str, dict]:
        payload_messages = list(messages)
        if system:
            payload_messages.insert(0, {"role": "system", "content": system})
        return (
            self.base_url + "/chat/completions",
            {"model": model, "messages": payload_messages, "stream": True},
        )

    def _anthropic_request(
        self, messages: list[dict[str, str]], model: str, system: str | None,
    ) -> tuple[str, dict]:
        payload: dict = {"model": model, "messages": messages, "stream": True}
        if system:
            payload["system"] = system
        return self.base_url + "/messages", payload

    def _open_stream(self, url: str, payload: dict) -> Iterator[str]:
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
                    yield from self._iter_deltas(response)
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"cannot reach {url}: {exc.__class__.__name__}: {exc}") from exc

    def _iter_deltas(self, response: httpx.Response) -> Iterator[str]:
        """Yield text deltas from an SSE response body, per api_style."""
        if self.api_style == "anthropic":
            yield from self._anthropic_deltas(response)
        else:
            yield from self._chat_deltas(response)

    def _chat_deltas(self, response: httpx.Response) -> Iterator[str]:
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
            delta = event.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content")
            if text:
                yield text

    def _anthropic_deltas(self, response: httpx.Response) -> Iterator[str]:
        for raw in response.iter_lines():
            line = raw.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"malformed SSE chunk from provider: {exc}") from exc
            etype = event.get("type")
            if etype == "error":
                err = event.get("error", {})
                raise ProviderError(f"provider error: {err.get('message', event)}")
            if etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]
