"""The setting of the progress of a run on the command line."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from sbmlsim import progress
from sbmlsim.run import cli

DUMMY = "tests.run.kinds_dummy:DUMMY"


@pytest.fixture
def spec(tmp_path: Path) -> Path:
    """The spec of a run of the dummy kind, which reports its units."""
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"n": 4, "seed": 1}), encoding="utf-8")
    return path


def _progress_events(out: Path) -> list[dict]:
    """The progress events of a run."""
    return [
        record
        for line in (out / "progress.jsonl").read_text().splitlines()
        if (record := json.loads(line))["event"] == "progress"
    ]


def test_the_run_cli_records_the_progress_with_progress_none(
    spec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`none` renders nothing, the events are recorded all the same."""
    monkeypatch.delenv("SBMLSIM_PROGRESS", raising=False)
    out = tmp_path / "out"
    assert (
        cli.main(["run", DUMMY, str(spec), "--out", str(out), "--progress", "none"])
        == 0
    )
    records = _progress_events(out)
    assert records[0]["state"] == "started"
    assert records[-1]["state"] == "finished"
    assert records[-1]["kind"] == "dummy"
    assert (records[-1]["completed"], records[-1]["total"]) == (4, 4)
    assert records[-1]["resources"]["cores"] >= 1


def test_the_run_cli_logs_the_progress(
    spec: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`log` writes the lines of the progress, here without an interval."""
    monkeypatch.setattr(progress.reporters, "LOG_INTERVAL", 0.0)
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger="sbmlsim.progress"):
        assert (
            cli.main(["run", DUMMY, str(spec), "--out", str(out), "--progress", "log"])
            == 0
        )
    lines = [record.message for record in caplog.records]
    assert any("dummy units" in line and "cores" in line for line in lines)


def test_the_run_cli_reads_the_environment(
    spec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the argument the environment decides, see `resolve`."""
    monkeypatch.setenv("SBMLSIM_PROGRESS", "none")
    out = tmp_path / "out"
    assert cli.main(["run", DUMMY, str(spec), "--out", str(out)]) == 0
    assert _progress_events(out)


def test_the_run_cli_refuses_an_unknown_progress(spec: Path, tmp_path: Path) -> None:
    """The choices of the argument are the renderings, see the usage of the CLI."""
    assert (
        cli.main(
            ["run", DUMMY, str(spec), "--out", str(tmp_path), "--progress", "bars"]
        )
        == cli.EXIT_USAGE
    )


def test_the_names_of_the_progress_are_those_of_the_setting() -> None:
    assert progress.NAMES == ("auto", "bar", "log", "none")
