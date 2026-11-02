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
        for name in ("openai", "anthropic", "opencode", "mimo", "ollama"):
            assert f"[providers.{name}]" in text, f"missing provider {name}"

    def test_write_template_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "nested" / "deep" / "config.toml"
        write_template(target)
        assert target.exists()


class TestLoad:
    def test_load_missing_file_raises_with_hint(self, tmp_path):
        with pytest.raises(ConfigError, match="ana init"):
            load_config(tmp_path / "missing.toml")

    def test_load_parses_providers_and_default(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        write_template(cfg_path)
        cfg = load_config(cfg_path)
        assert "opencode" in cfg.providers
        assert cfg.providers["openai"].base_url == "https://api.openai.com/v1"
        assert cfg.default_provider in cfg.providers

    def test_load_rejects_default_provider_unknown(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(CONFIG_TEMPLATE.replace(
            'default_provider = "opencode"', 'default_provider = "nope"'))
        with pytest.raises(ConfigError, match="provider 'nope' is not defined"):
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
