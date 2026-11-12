"""The command line of the service, `python -m sbmlsim.serve`.

The models of the command line are references of projects, which `main` loads
into a `ModelRegistry`; the tests use the toy project of `tests/project_toy`,
which imports no simulator, and never start a server: `serve` is replaced, so
a test runs the whole command up to the point where uvicorn would take over.
"""

from pathlib import Path
from typing import Any

import pytest

from sbmlsim.serve import cli

#: the project of the models of the tests, which loads nothing heavy
TOY = "tests.project_toy:PROJECT"


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record what `main` would serve instead of running a server."""
    recorded: dict[str, Any] = {}

    def record(app: Any, **kwargs: Any) -> None:
        recorded["app"] = app
        recorded.update(kwargs)

    monkeypatch.setattr(cli, "serve", record)
    return recorded


@pytest.fixture
def created(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record the arguments `main` creates the application with."""
    recorded: dict[str, Any] = {}

    def record(registry: Any, **kwargs: Any) -> Any:
        recorded["registry"] = registry
        recorded.update(kwargs)
        return object()

    monkeypatch.setattr("sbmlsim.serve.app.create_app", record)
    return recorded


def test_parse_models_reads_id_and_reference() -> None:
    assert cli.parse_models(["hctz=examples.hctz_fitting.project:PROJECT"]) == {
        "hctz": "examples.hctz_fitting.project:PROJECT"
    }


def test_the_models_keep_the_order_of_the_command_line() -> None:
    models = cli.parse_models([f"toy={TOY}", "hctz=hctz.project:PROJECT"])
    assert list(models) == ["toy", "hctz"]


def test_a_model_without_an_equals_sign_exits_with_the_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli.parse_models(["examples.hctz_fitting.project:PROJECT"])
    assert exit_.value.code == cli.EXIT_USAGE
    message = capsys.readouterr().err
    assert "usage" in message
    assert "id=module:attribute" in message


def test_a_model_without_an_id_or_a_reference_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for value in ("=module:attribute", "hctz="):
        with pytest.raises(SystemExit) as exit_:
            cli.parse_models([value])
        assert exit_.value.code == cli.EXIT_USAGE
    assert "usage" in capsys.readouterr().err


def test_a_duplicate_model_id_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli.parse_models([f"toy={TOY}", "toy=other.project:PROJECT"])
    assert exit_.value.code == cli.EXIT_USAGE
    message = capsys.readouterr().err
    assert "toy" in message


def test_main_without_a_model_exits_with_code_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_main_serves_the_application_of_its_models(served: dict[str, Any]) -> None:
    from fastapi import FastAPI

    assert cli.main(["--model", f"toy={TOY}"]) == cli.EXIT_OK
    assert isinstance(served["app"], FastAPI)
    assert served["host"] == cli.DEFAULT_HOST
    assert served["port"] == cli.DEFAULT_PORT
    assert served["open_browser"] is False


def test_the_service_binds_the_loopback_address(served: dict[str, Any]) -> None:
    assert cli.DEFAULT_HOST == "127.0.0.1"
    assert cli.main(["--model", f"toy={TOY}"]) == cli.EXIT_OK
    assert served["host"] == "127.0.0.1"


def test_the_host_and_the_port_reach_the_server(served: dict[str, Any]) -> None:
    code = cli.main(["--model", f"toy={TOY}", "--host", "0.0.0.0", "--port", "9001"])
    assert code == cli.EXIT_OK
    assert served["host"] == "0.0.0.0"
    assert served["port"] == 9001


def test_open_asks_for_the_browser(served: dict[str, Any]) -> None:
    assert cli.main(["--model", f"toy={TOY}", "--open"]) == cli.EXIT_OK
    assert served["open_browser"] is True


def test_the_static_directory_and_the_origins_reach_the_application(
    tmp_path: Path, served: dict[str, Any], created: dict[str, Any]
) -> None:
    code = cli.main(
        [
            "--model",
            f"toy={TOY}",
            "--static",
            str(tmp_path),
            "--allow-origin",
            "https://example.org",
            "--allow-origin",
            "https://slides.example.org",
        ]
    )
    assert code == cli.EXIT_OK
    assert created["static"] == tmp_path
    assert created["allow_origins"] == (
        "https://example.org",
        "https://slides.example.org",
    )
    assert created["registry"].ids() == ("toy",)


def test_a_service_without_origins_allows_none(
    served: dict[str, Any], created: dict[str, Any]
) -> None:
    assert cli.main(["--model", f"toy={TOY}"]) == cli.EXIT_OK
    assert created["static"] is None
    assert created["allow_origins"] == ()


def test_a_static_path_which_is_no_directory_exits_with_the_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope"
    assert cli.main(["--model", f"toy={TOY}", "--static", str(missing)]) == 2
    assert "usage" in capsys.readouterr().err


def test_a_model_which_is_no_project_is_a_configuration_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["--model", "toy=sbmlsim.no_such_module:PROJECT"])
    assert code == cli.EXIT_CONFIGURATION
    message = capsys.readouterr().err
    assert "toy" in message
    assert "sbmlsim.no_such_module" in message


class _Server:
    """A server which is up when its `started` is set, see `_open_when_up`."""

    started = False
    should_exit = False


def test_the_browser_is_opened_once_the_server_is_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []

    def open_(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(cli.webbrowser, "open", open_)
    server = _Server()
    thread = cli._open_when_up(server, "http://127.0.0.1:8000/")
    assert opened == []
    server.started = True
    thread.join(timeout=10.0)
    assert opened == ["http://127.0.0.1:8000/"]


def test_a_server_which_stops_opens_no_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(url: str) -> bool:
        pytest.fail("the browser was opened without a server")

    monkeypatch.setattr(cli.webbrowser, "open", refuse)
    server = _Server()
    server.should_exit = True
    cli._open_when_up(server, "http://127.0.0.1:8000/").join(timeout=10.0)


def test_the_command_imports_no_server_and_no_application_until_it_runs() -> None:
    """`sbmlsim.serve.cli` imports uvicorn and fastapi when it serves."""
    import subprocess
    import sys

    code = (
        "import sys, sbmlsim.serve.cli; "
        "print(sorted(m for m in ('uvicorn', 'fastapi', 'starlette',"
        " 'sbmlsim.serve.app', 'sbmlsim.project') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
