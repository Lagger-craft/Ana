"""Tests for ana.cli — REPL commands, one-shot mode, init and error paths."""

import pytest

from ana.cli import main, parse_command, run_repl
from ana.client import ProviderError
from ana.config import CONFIG_TEMPLATE, ConfigError, write_template


class TestParseCommand:
    def test_command_with_arg(self):
        assert parse_command("/model gpt-4o-mini") == ("model", "gpt-4o-mini")

    def test_command_without_arg(self):
        assert parse_command("/new") == ("new", "")

    def test_non_command_raises(self):
        with pytest.raises(ValueError):
            parse_command("hello ana")


@pytest.fixture
def cfg(tmp_path) -> object:
    path = tmp_path / "config.toml"
    write_template(path)
    return path


class TestRepl:
    def test_turn_appends_user_and_assistant_to_history(self, cfg):
        history = run_repl(
            cfg_path=cfg, lines=["hola"],
            stream=lambda msgs, model, system=None: iter(["hola!"]),
        )
        assert [(m["role"], m["content"]) for m in history] == [
            ("user", "hola"), ("assistant", "hola!")]

    def test_exit_ends_loop_without_api_call(self, cfg):
        history = run_repl(cfg_path=cfg, lines=["/exit"], stream=None)
        assert history == []

    def test_new_clears_history(self, cfg):
        history = run_repl(
            cfg_path=cfg,
            lines=["turn1", "/new", "turn2"],
            stream=lambda msgs, model, system=None: iter(["r"]),
        )
        # after /new only the second turn remains
        assert [m["content"] for m in history] == ["turn2", "r"]

    def test_failed_turn_is_rolled_back_and_loop_survives(self, cfg):
        replies = iter([ProviderError("boom"), iter(["ok"])])

        def stream(msgs, model, system=None):
            item = next(replies)
            if isinstance(item, Exception):
                raise item
            return item

        history = run_repl(cfg_path=cfg, lines=["q1", "q2"], stream=stream)
        # q1 rolled back on failure; only the successful turn stays
        assert [(m["role"], m["content"]) for m in history] == [
            ("user", "q2"), ("assistant", "ok")]

    def test_unknown_command_reported_not_sent(self, cfg, capsys):
        history = run_repl(cfg_path=cfg, lines=["/wat"], stream=None)
        assert history == []
        assert "unknown command" in capsys.readouterr().out

    def test_model_without_arg_shows_current(self, cfg, capsys):
        run_repl(cfg_path=cfg, lines=["/model"], stream=None)
        assert "kimi-k3" in capsys.readouterr().out

    def test_model_with_arg_switches_model(self, cfg):
        seen = []

        def stream(msgs, model, system=None):
            seen.append(model)
            return iter(["ok"])

        run_repl(cfg_path=cfg, lines=["/model my-model", "hi"], stream=stream)
        assert seen == ["my-model"]

    def test_provider_with_arg_switches_provider(self, cfg):
        seen = []

        def stream(msgs, model, system=None):
            seen.append(model)
            return iter(["ok"])

        history = run_repl(
            cfg_path=cfg, lines=["/provider openai", "hi"],
            stream=stream, models={"openai": "gpt-4o-mini"})
        assert seen == ["gpt-4o-mini"]
        assert history != []

    def test_history_grows_across_turns(self, cfg):
        def stream(msgs, model, system=None):
            return iter(["r"])

        history = run_repl(cfg_path=cfg, lines=["a", "b"], stream=stream)
        assert [m["content"] for m in history] == ["a", "r", "b", "r"]


class TestMain:
    def test_one_shot_prints_stream_and_exits_zero(
            self, cfg, monkeypatch, capsys):
        monkeypatch.setattr(
            "ana.cli.build_stream",
            lambda cfg, provider=None: lambda msgs, model, system=None:
                iter(["hi", " there"]))
        rc = main(["--config", str(cfg), "--message", "hi"])
        assert rc == 0
        assert "hi there" in capsys.readouterr().out

    def test_missing_config_exits_one_with_init_hint(self, tmp_path, capsys):
        rc = main(["--config", str(tmp_path / "missing.toml"),
                   "--message", "hi"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "ana init" in err

    def test_missing_api_key_env_exits_one_actionably(self, tmp_path,
                                                      monkeypatch, capsys):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            CONFIG_TEMPLATE.replace('default_provider = "opencode-go"',
                                    'default_provider = "openai"'))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rc = main(["--config", str(cfg_path), "--message", "hi"])
        assert rc == 1
        assert "OPENAI_API_KEY" in capsys.readouterr().err

    def test_provider_error_exits_one_with_message(self, cfg, capsys):
        def boom(msgs, model, system=None):
            raise ProviderError("500 'oops' body")
            yield  # pragma: no cover

        import ana.cli as cli_module
        cli_module.build_stream = lambda cfg, provider=None: boom
        rc = main(["--config", str(cfg), "--message", "hi"])
        assert rc == 1
        assert "oops" in capsys.readouterr().err

    def test_init_writes_template_to_default_location(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        rc = main(["init"])
        out = tmp_path / "ana" / "config.toml"
        assert rc == 0
        assert out.read_text() == CONFIG_TEMPLATE
