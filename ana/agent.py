"""Agent loop: run a chat turn, executing tools until a final answer.

run_agent() drives provider stream_turn() through ToolRegistry-backed
tool calls, respecting per-dialect message shapes:
- chat style: assistant tool_calls[] + role="tool" results
- anthropic: tool_use blocks + tool_result blocks in a user message

Tool result convention (NON-BREAKING): a tool fn returns its result
as before — a plain str (or any JSON-serializable value) becomes the
ordinary tool-result text and nothing else changes. A fn may instead
return a ``ToolResult``: its ``text`` fills the normal tool-result
slot, and its ``image`` (``{media_type, data(base64)}`` — the same
shape client.py already sends) is delivered as a synthetic
``{"role": "user", "content": ..., "images": [...]}`` message right
after that round's tool results, so the model answers from the picture
in the SAME run. Once an image is injected, later rounds stream with
the caller's ``vision_model`` (``model`` when absent — best effort).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class AgentError(Exception):
    """Raised for agent-level failures (unknown tool, round limit)."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[dict], Any]

    def __post_init__(self) -> None:
        if not self.name or " " in self.name:
            raise AgentError(f"invalid tool name: {self.name!r}")


@dataclass(frozen=True)
class ToolResult:
    """Tool result that can also carry an image into the current turn.

    ``text`` is what the model reads in the tool-result slot (exactly
    what a plain str return would produce). When ``image`` is set —
    ``{"media_type", "data(base64)}`` — run_agent appends a synthetic
    user message carrying it and switches the run to the caller's
    ``vision_model``. Plain str/dict returns keep today's behavior.
    """

    text: str
    image: dict | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def to_spec(self, style: str) -> list[dict]:
        """Tool schemas in the dialect's wire format."""
        tools = sorted(self._tools.values(), key=lambda t: t.name)
        if style == "anthropic":
            return [{
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            } for t in tools]
        return [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        } for t in tools]

    def call(self, name: str, arguments: dict) -> Any:
        if name not in self._tools:
            raise AgentError(f"unknown tool: {name}")
        return self._tools[name].fn(arguments)

    def __len__(self) -> int:
        return len(self._tools)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def run_agent(
    client,
    messages: list[dict],
    model: str,
    registry: ToolRegistry,
    emit: Callable[[str], None],
    on_tool: Callable[[str, dict], None] | None = None,
    system: str | None = None,
    max_rounds: int = 4,
    vision_model: str | None = None,
) -> list[dict]:
    """Run one user turn with tools; return the updated history.

    ``messages`` must END with the user message. Text is streamed to
    ``emit`` as it arrives; ``on_tool(name, args)`` is called just
    before each tool runs so callers can surface progress.

    ``vision_model``: after a ``ToolResult`` with an image is injected,
    every later ``stream_turn`` in this run uses it (when set); without
    an injection — or when it is None — ``model`` is used unchanged.
    See the module docstring for the tool-result convention.
    """
    style = client.api_style
    tools = registry.to_spec(style) if len(registry) else None
    if messages[-1]["role"] != "user":
        raise AgentError("messages must end with the user turn")

    current_model = model

    for _round in range(max_rounds):
        reply_text: list[str] = []
        calls: list[dict] = []
        reasoning_noted = False
        for event in client.stream_turn(messages, current_model, system=system,
                                        tools=tools):
            if event["type"] == "text":
                reply_text.append(event["content"])
                emit(event["content"])
            elif event["type"] == "reasoning":
                if not reasoning_noted:
                    # reasoning models stream thinking silently for seconds;
                    # one note prevents the "frozen prompt" feeling
                    reasoning_noted = True
                    emit("(🤔 razonando...)")
            elif event["type"] == "tool_call":
                calls.append(event)
            elif event["type"] == "finish":
                break

        if not calls:
            text = "".join(reply_text)
            messages.append({"role": "assistant", "content": text})
            return messages

        # one shared executor for both dialects: runs the tool, collects
        # any image for same-turn injection, returns the result text
        shots: list[dict] = []

        def _invoke(call: dict) -> str:
            if on_tool:
                on_tool(call["name"], call["arguments"])
            try:
                value = registry.call(call["name"], call["arguments"])
            except AgentError as exc:
                return f"tool error: {exc}"
            if isinstance(value, ToolResult):
                if value.image:
                    shots.append(value.image)
                return value.text
            return _stringify(value)

        if style == "anthropic":
            blocks: list[dict] = []
            if "".join(reply_text):
                blocks.append({"type": "text", "text": "".join(reply_text)})
            for call in calls:
                blocks.append({
                    "type": "tool_use", "id": call["id"],
                    "name": call["name"], "input": call["arguments"],
                })
            messages.append({"role": "assistant", "content": blocks})
            results: list[dict] = [{
                "type": "tool_result", "tool_use_id": call["id"],
                "content": _invoke(call),
            } for call in calls]
            messages.append({"role": "user", "content": results})
        else:
            assistant_msg: dict = {"role": "assistant",
                                   "content": "".join(reply_text) or None,
                                   "tool_calls": [{
                                       "type": "function",
                                       "id": c["id"],
                                       "function": {
                                           "name": c["name"],
                                           "arguments": json.dumps(
                                               c["arguments"],
                                               ensure_ascii=False),
                                       },
                                   } for c in calls]}
            messages.append(assistant_msg)
            for call in calls:
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": _invoke(call),
                })

        if shots:
            # same-turn injection: the picture rides back as a synthetic
            # user message BEFORE the next round streams, and that round
            # (and every later one in this run) uses the vision model
            messages.append({
                "role": "user",
                "content": "Screenshot attached — answer from this image.",
                "images": shots,
            })
            if vision_model:
                current_model = vision_model

    raise AgentError(
        f"model kept requesting tool rounds (limit {max_rounds}); stopping")
