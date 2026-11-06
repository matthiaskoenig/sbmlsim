"""The command line `python -m sbmlsim.run`, run as a process as simrun runs it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sbmlsim.run import cli

REPOSITORY = Path(__file__).parents[2]
DUMMY = "tests.run.kinds_dummy:DUMMY"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in (*THREAD_VARIABLES, "MPLBACKEND")
        and not name.startswith("SIMRUN_")
    }
    env["PYTHONPATH"] = str(REPOSITORY)
    return subprocess.run(
        [sys.executable, "-m", "sbmlsim.run", *args],
        cwd=REPOSITORY,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _spec(tmp_path: Path, content: dict[str, object]) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_cli_runs_a_kind(tmp_path: Path) -> None:
    spec = _spec(tmp_path, {"n": 4, "seed": 1})
    fit = tmp_path / "fit"
    fit.mkdir()
    out = tmp_path / "out"
    result = _run(DUMMY, str(spec), "--out", str(out), f"--input=fit={fit}")
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1 + 4 + 9 + 16
    planned = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    assert planned["fit"] == str(fit.resolve())


def test_cli_sets_thread_variables(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = _run(DUMMY, str(_spec(tmp_path, {"n": 1})), "--out", str(out))
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["omp_num_threads"] == "1"
    assert summary["mplbackend"] == "Agg"


def test_cli_invalid_spec_exits_4(tmp_path: Path) -> None:
    spec = _spec(tmp_path, {"n": 1, "unknown": 2})
    result = _run(DUMMY, str(spec), "--out", str(tmp_path / "out"))
    assert result.returncode == 4
    assert "unknown" in result.stderr


def test_cli_failing_kind_exits_3(tmp_path: Path) -> None:
    spec = _spec(tmp_path, {"n": 1, "fail": True})
    result = _run(DUMMY, str(spec), "--out", str(tmp_path / "out"))
    assert result.returncode == 3
    assert "the dummy kind fails" in result.stderr


def test_cli_unknown_kind_exits_4(tmp_path: Path) -> None:
    spec = _spec(tmp_path, {"n": 1})
    result = _run("unknown", str(spec), "--out", str(tmp_path / "out"))
    assert result.returncode == 4
    assert "unknown" in result.stderr


def test_cli_plan_prints_units(tmp_path: Path) -> None:
    result = _run("plan", DUMMY, str(_spec(tmp_path, {"n": 130})))
    assert result.returncode == 0, result.stderr
    planned = json.loads(result.stdout)
    assert planned["units"] == 44
    assert isinstance(planned["seed"], int)


def test_cli_list_json() -> None:
    result = _run("list", "examples.hctz_fitting.project:PROJECT", "--json")
    assert result.returncode == 0, result.stderr
    analyses = json.loads(result.stdout)
    assert ("fit", "PK") in {(a["kind"], a["name"]) for a in analyses}


# the commands in the process: the same code without the cost of a process


def test_main_plan_into_a_file_unit_and_merge(tmp_path: Path) -> None:
    planned = tmp_path / "planned.json"
    assert (
        cli.main(["plan", DUMMY, str(_spec(tmp_path, {"n": 3})), "-o", str(planned)])
        == 0
    )
    out = tmp_path / "out"
    for index in range(3):
        assert (
            cli.main(["unit", DUMMY, str(planned), str(index), "--out", str(out)]) == 0
        )
    assert cli.main(["merge", DUMMY, str(planned), "--out", str(out)]) == 0
    assert (out / "summary.json").exists()


def test_main_writes_the_summary_into_a_file(tmp_path: Path) -> None:
    """`--summary` writes the summary of a run or a merge also into a file."""
    spec = _spec(tmp_path, {"n": 3})
    out = tmp_path / "out"
    summary = tmp_path / "task" / "summary.json"
    assert (
        cli.main([DUMMY, str(spec), "--out", str(out), "--summary", str(summary)]) == 0
    )
    assert summary.read_bytes() == (out / "summary.json").read_bytes()

    planned = tmp_path / "planned.json"
    assert cli.main(["plan", DUMMY, str(spec), "-o", str(planned)]) == 0
    split = tmp_path / "split"
    for index in range(3):
        assert (
            cli.main(["unit", DUMMY, str(planned), str(index), "--out", str(split)])
            == 0
        )
    merged = tmp_path / "merged.json"
    assert (
        cli.main(
            [
                "merge",
                DUMMY,
                str(planned),
                "--out",
                str(split),
                "--summary",
                str(merged),
            ]
        )
        == 0
    )
    assert merged.read_bytes() == (split / "summary.json").read_bytes()


def test_main_failed_run_writes_no_summary_file(tmp_path: Path) -> None:
    spec = str(_spec(tmp_path, {"n": 1, "fail": True}))
    summary = tmp_path / "summary.json"
    out = str(tmp_path / "out")
    assert cli.main([DUMMY, spec, "--out", out, "--summary", str(summary)]) == 3
    assert not summary.exists()


def test_main_unit_outside_the_range_exits_4(tmp_path: Path) -> None:
    planned = tmp_path / "planned.json"
    assert (
        cli.main(["plan", DUMMY, str(_spec(tmp_path, {"n": 3})), "-o", str(planned)])
        == 0
    )
    code = cli.main(["unit", DUMMY, str(planned), "3", "--out", str(tmp_path / "out")])
    assert code == 4


def test_main_list_as_a_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list", "examples.hctz_fitting.project:PROJECT"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.split()[:2] == ["fit", "PK"] for line in lines)


def test_main_list_with_the_experiments(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list", "tests.project_toy:PROJECT", "--json"]) == 0
    names = [entry["name"] for entry in json.loads(capsys.readouterr().out)]
    assert "studies" not in names
    assert "rates" in names
    assert cli.main(["list", "tests.project_toy:PROJECT", "--experiments"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.split()[:2] == ["simulate", "studies"] for line in lines)
    assert any(line.split()[:2] == ["simulate", "ToyDecaySlow"] for line in lines)


def test_main_invalid_object_of_the_analysis_exits_3(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the validation of the spec is a configuration error."""
    spec = str(_spec(tmp_path, {"n": 1, "invalid": True}))
    assert cli.main([DUMMY, spec, "--out", str(tmp_path / "out")]) == 3
    assert "ValidationError" in caplog.text


def test_main_resume(tmp_path: Path) -> None:
    spec = str(_spec(tmp_path, {"n": 2}))
    out = str(tmp_path / "out")
    assert cli.main([DUMMY, spec, "--out", out]) == 0
    assert cli.main([DUMMY, spec, "--out", out]) == 4
    assert cli.main([DUMMY, spec, "--out", out, "--resume"]) == 0


@pytest.mark.parametrize(
    "args",
    [
        [],
        [DUMMY, "spec.json"],
        [DUMMY, "spec.json", "--out", "x", "--input", "no-equals"],
        [DUMMY, "spec.json", "--out", "x", "--workers", "0"],
        ["unit", DUMMY, "spec.json", "one", "--out", "x"],
    ],
)
def test_main_bad_command_line_exits_2(args: list[str]) -> None:
    assert cli.main(args) == 2


def test_input_pairs_split_at_the_first_equals() -> None:
    assert cli.input_pair("fit=/a=b") == ("fit", Path("/a=b"))
