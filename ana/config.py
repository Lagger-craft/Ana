"""Configuration for Ana: TOML file with provider definitions.

Config location defaults to ~/.config/ana/config.toml. Create it with
``ana init``. API keys live in environment variables (``api_key_env``),
never in the config file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VALID_API_STYLES = ("chat", "anthropic")

CONFIG_TEMPLATE = '''\
# Ana configuration. Create with: ana init
# API keys are read from environment variables, never stored here.

# Name of the provider to use when --provider is not given.
default_provider = "opencode-go"

# Optional persona prompt sent with every request.
system_prompt = "You are Ana, a friendly virtual assistant. Answer in the user's language."

[providers.openai]
base_url = "https://api.openai.com/v1"
api_style = "chat"
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-4o-mini"

# Anthropic uses its own Messages API (api_style = "anthropic").
[providers.anthropic]
base_url = "https://api.anthropic.com/v1"
api_style = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-sonnet-4-5"

# OpenCode Go (console subscription, $10/mo). Create a Personal API key
# in the OpenCode Console (Account > Personal API keys) and export it:
#   export OPENCODE_API_KEY="oc_sk_..."
# The gateway serves models by API family. Open models (kimi, glm,
# minimax, deepseek, qwen3.8+) live on this Chat Completions endpoint:
[providers.opencode-go]
base_url = "https://opencode.ai/inference/openai/v1"
api_style = "chat"
api_key_env = "OPENCODE_API_KEY"
default_model = "kimi-k3"

# Claude models on the same OpenCode Go key use the Messages API path.
# (GPT* models need the Responses API, not supported yet.)
[providers.opencode-go-claude]
base_url = "https://opencode.ai/inference/anthropic/v1"
api_style = "anthropic"
api_key_env = "OPENCODE_API_KEY"
default_model = "claude-sonnet-5-5"

# Xiaomi MiMo - adjust base_url/model to match your Xiaomi API docs.
[providers.mimo]
base_url = "https://api.mimo.xiaomi.com/v1"
api_style = "chat"
api_key_env = "MIMO_API_KEY"
default_model = "mimo-vl-7b"

# Local Ollama (no key needed). Requires `ollama serve`.
[providers.ollama]
base_url = "http://127.0.0.1:11434/v1"
api_style = "chat"
api_key_env = ""
default_model = "qwen3.5:4b"
# Model used when you attach images (vision). Leave it empty to always
# use default_model for everything.
vision_model = "qwen3-vl:4b"
# Keeps this model loaded between requests (default unload is 5m).
keep_alive = "1h"
# Output token budget for image turns (max_tokens). Measured on RX 7600:
# 256 ≈ 8s but can answer EMPTY (thinking eats the budget); 1024 ≈ 9s
# with complete thinking; unset ≈ 18s. Remove the line to disable.
vision_max_tokens = 1024
'''


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_style: str = "chat"
    api_key_env: str | None = None
    default_model: str = ""
    vision_model: str | None = None  # used automatically when images are attached
    keep_alive: str | None = None    # stay-loaded window for local servers
    vision_max_tokens: int | None = None  # output budget for image turns
    headers: dict[str, str] | None = None  # extra request headers
    fallback_provider: str | None = None  # switch here when free tier dies

    def __post_init__(self) -> None:
        if self.api_style not in VALID_API_STYLES:
            raise ConfigError(
                f"provider '{self.name}': unknown api_style '{self.api_style}' "
                f"(expected one of {', '.join(VALID_API_STYLES)})")
        if not self.base_url:
            raise ConfigError(f"provider '{self.name}': base_url is required")


@dataclass(frozen=True)
class Config:
    default_provider: str
    providers: dict[str, Provider] = field(default_factory=dict)
    system_prompt: str = ""

    def provider(self, name: str | None = None) -> Provider:
        key = name or self.default_provider
        try:
            return self.providers[key]
        except KeyError:
            known = ", ".join(sorted(self.providers)) or "(none)"
            raise ConfigError(
                f"provider '{key}' is not defined in the config "
                f"(known: {known})") from None

    def resolve_key(self, provider: Provider) -> str | None:
        """Return the provider API key from the environment, or None."""
        if not provider.api_key_env:
            return None
        value = os.environ.get(provider.api_key_env)
        if not value:
            raise ConfigError(
                f"provider '{provider.name}': environment variable "
                f"{provider.api_key_env} is not set. "
                f"Export it, then try again.")
        return value


def default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(xdg).expanduser() / "ana" / "config.toml"


def write_template(path: Path) -> None:
    """Write the starter config template, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE)


def load_config(path: str | Path) -> Config:
    """Parse config.toml into a Config, validating structure."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"config file not found: {path}\n"
            f"Run `ana init` to create a starter config, then edit it.")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    known_top = {"default_provider", "providers", "system_prompt"}
    unknown = sorted(set(data) - known_top)
    if unknown:
        raise ConfigError(
            f"{path}: unknown top-level keys: {', '.join(unknown)}")

    default_provider = data.get("default_provider")
    if not default_provider or not isinstance(default_provider, str):
        raise ConfigError(f"{path}: default_provider must be a string")

    providers_block = data.get("providers", {})
    providers: dict[str, Provider] = {}
    for name, raw in providers_block.items():
        known_prov = {"base_url", "api_style", "api_key_env",
                      "default_model", "vision_model", "keep_alive",
                      "vision_max_tokens", "headers",
                      "fallback_provider"}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: provider '{name}' must be a table")
        bad = sorted(set(raw) - known_prov)
        if bad:
            raise ConfigError(
                f"{path}: unknown keys in [providers.{name}]: {', '.join(bad)}")
        max_tokens = raw.get("vision_max_tokens")
        hdrs = raw.get("headers")
        if hdrs is not None and (
                not isinstance(hdrs, dict)
                or not all(isinstance(k, str) and isinstance(v, str)
                           for k, v in hdrs.items())):
            raise ConfigError(
                f"{path}: provider '{name}': headers must be a table "
                "of string values")
        providers[name] = Provider(
            name=name,
            base_url=str(raw.get("base_url", "")),
            api_style=str(raw.get("api_style", "chat")),
            api_key_env=raw.get("api_key_env") or None,
            default_model=str(raw.get("default_model", "")),
            vision_model=raw.get("vision_model") or None,
            keep_alive=raw.get("keep_alive") or None,
            vision_max_tokens=int(max_tokens) if max_tokens else None,
            headers=dict(hdrs) if hdrs else None,
            fallback_provider=raw.get("fallback_provider") or None,
        )

    cfg = Config(
        default_provider=default_provider,
        providers=providers,
        system_prompt=str(data.get("system_prompt", "")),
    )
    cfg.provider()  # validates default_provider exists
    for name, prov in providers.items():
        # mirror default_provider validation: the fallback target must
        # exist in THIS config, and the error names the bad target
        target = prov.fallback_provider
        if target is None:
            continue
        if not isinstance(target, str):
            raise ConfigError(
                f"{path}: provider '{name}': fallback_provider must be "
                "a string")
        try:
            cfg.provider(target)
        except ConfigError as exc:
            raise ConfigError(
                f"{path}: provider '{name}': fallback_provider: {exc}"
            ) from None
    return cfg
