"""Tests for ana.config — TOML provider config, env-var key indirection, template creation."""

import os

import pytest

from ana.config import (
    CONFIG_TEMPLATE,
    Config,
    ConfigError,
    Provider,
    load_config,
    write_template,
)


class TestTemplate:
    def test_write_template_creates_toml_with_known_providers(self, tmp_path):
        target = tmp_path / "config.toml"
        write_template(target)
        text = target.read_text()
        assert text == CONFIG_TEMPLATE
        for name in (
            "openai",
            "anthropic",
            "opencode-go",
            "opencode-go-claude",
            "mimo",
            "ollama",
        ):
            assert f"[providers.{name}]" in text, f"missing provider {name}"

    def test_write_template_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "nested" / "deep" / "config.toml"
        write_template(target)
        assert target.exists()

    def test_opencode_go_points_at_inference_gateway(self, tmp_path):
        target = tmp_path / "config.toml"
        write_template(target)
        text = target.read_text()
        assert 'base_url = "https://opencode.ai/inference/openai/v1"' in text
        assert 'base_url = "https://opencode.ai/inference/anthropic/v1"' in text
        assert 'api_key_env = "OPENCODE_API_KEY"' in text


class TestLoad:
    def test_load_missing_file_raises_with_hint(self, tmp_path):
        with pytest.raises(ConfigError, match="ana init"):
            load_config(tmp_path / "missing.toml")

    def test_load_parses_providers_and_default(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        write_template(cfg_path)
        cfg = load_config(cfg_path)
        assert "opencode-go" in cfg.providers
        assert cfg.providers["openai"].base_url == "https://api.openai.com/v1"
        assert cfg.default_provider in cfg.providers

    def test_load_rejects_default_provider_unknown(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(CONFIG_TEMPLATE.replace(
            'default_provider = "opencode-go"', 'default_provider = "nope"'))
        with pytest.raises(ConfigError, match="provider 'nope' is not defined"):
            load_config(cfg_path)

    def test_load_parses_provider_headers(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            'default_provider = "oc"\n\n'
            '[providers.oc]\n'
            'base_url = "https://example.invalid/v1"\n'
            'api_key_env = ""\n'
            'default_model = "m"\n'
            'headers = { "x-opencode-session" = "ana-1" }\n')
        cfg = load_config(cfg_path)
        assert cfg.providers["oc"].headers == {
            "x-opencode-session": "ana-1"}

    def test_provider_headers_must_be_a_string_table(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            'default_provider = "oc"\n\n'
            '[providers.oc]\n'
            'base_url = "https://example.invalid/v1"\n'
            'headers = "nope"\n')
        with pytest.raises(ConfigError, match="headers"):
            load_config(cfg_path)

    def test_unknown_provider_keys_rejected(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        write_template(cfg_path)
        text = cfg_path.read_text()
        cfg_path.write_text(text + "\nnot_a_key = 1\n")
        with pytest.raises(ConfigError, match="not_a_key"):
            load_config(cfg_path)


class TestApiKey:
    def test_resolve_key_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-secret")
        cfg = Config(
            default_provider="p",
            providers={"p": Provider(
                name="p", base_url="https://example.invalid/v1",
                api_key_env="TEST_API_KEY", default_model="m1")},
        )
        assert cfg.resolve_key(cfg.providers["p"]) == "sk-secret"

    def test_resolve_key_missing_env_raises_actionable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        cfg = Config(
            default_provider="p",
            providers={"p": Provider(
                name="p", base_url="https://example.invalid/v1",
                api_key_env="TEST_API_KEY", default_model="m1")},
        )
        with pytest.raises(ConfigError, match="TEST_API_KEY"):
            cfg.resolve_key(cfg.providers["p"])

    def test_resolve_key_not_required_ok(self, tmp_path):
        cfg = Config(
            default_provider="p",
            providers={"p": Provider(
                name="p", base_url="http://127.0.0.1:11434/v1",
                api_key_env=None, default_model="m1")},
        )
        assert cfg.resolve_key(cfg.providers["p"]) is None


def _write_config(tmp_path, text):
    p = tmp_path / "config.toml"
    p.write_text(text)
    import re
    return p


class TestVisionSpeedConfig:
    def test_provider_parses_keep_alive_and_vision_max_tokens(self, tmp_path):
        from ana.config import load_config

        cfg = load_config(_write_config(tmp_path, """
default_provider = "x"
[providers.x]
base_url = "http://x/v1"
default_model = "m1"
keep_alive = "1h"
vision_max_tokens = 1024
"""))
        p = cfg.provider("x")
        assert p.keep_alive == "1h"
        assert p.vision_max_tokens == 1024

    def test_missing_speed_fields_are_none(self, tmp_path):
        from ana.config import load_config

        cfg = load_config(_write_config(tmp_path, """
default_provider = "x"
[providers.x]
base_url = "http://x/v1"
default_model = "m1"
"""))
        p = cfg.provider("x")
        assert p.keep_alive is None
        assert p.vision_max_tokens is None

    def test_template_ships_speed_defaults_for_ollama(self):
        import tomllib
        from ana.config import CONFIG_TEMPLATE

        data = tomllib.loads(CONFIG_TEMPLATE)
        ollama = data["providers"]["ollama"]
        assert ollama.get("keep_alive") == "1h"
        assert ollama.get("vision_max_tokens") == 1024


class TestFallbackProvider:
    """`fallback_provider = "<provider name>"` — free-tier safety net."""

    def test_fallback_provider_parses_and_defaults_to_none(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, """
default_provider = "beam"
[providers.beam]
base_url = "https://beam.invalid/v1"
api_key_env = ""
default_model = "m"
fallback_provider = "ollama"
[providers.ollama]
base_url = "http://127.0.0.1:11434/v1"
api_key_env = ""
default_model = "q"
"""))
        # forward reference works: the target is declared AFTER beam
        assert cfg.provider("beam").fallback_provider == "ollama"
        # no key declared -> None (default)
        assert cfg.provider("ollama").fallback_provider is None

    def test_unknown_fallback_target_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError) as excinfo:
            load_config(_write_config(tmp_path, """
default_provider = "beam"
[providers.beam]
base_url = "https://beam.invalid/v1"
api_key_env = ""
default_model = "m"
fallback_provider = "nope"
"""))
        # mirrors default_provider validation: names the bad target
        assert "provider 'nope' is not defined" in str(excinfo.value)
        assert "fallback_provider" in str(excinfo.value)

    def test_non_string_fallback_target_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError,
                           match="fallback_provider must be a string"):
            load_config(_write_config(tmp_path, """
default_provider = "beam"
[providers.beam]
base_url = "https://beam.invalid/v1"
api_key_env = ""
default_model = "m"
fallback_provider = 123
"""))
