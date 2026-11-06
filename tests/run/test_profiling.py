"""The profile of a run, `python -m sbmlsim.run ... --profile`.

The profiled runs need the sampling profiler of python 3.15 and are skipped
before it.
"""

import json
import os
import re
from pathlib import Path

import pytest

from sbmlsim.run import cli, profiling

DUMMY = "tests.run.kinds_dummy:DUMMY"
REPOSITORY = Path(__file__).parents[2]

needs_profiler = pytest.mark.skipif(
    not profiling.profiler_available(),
    reason="the sampling profiler is part of python 3.15",
)


def _spec(tmp_path: Path, content: dict[str, object]) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_the_folder_of_a_profile(tmp_path: Path) -> None:
    folder = profiling.profile_dir(tmp_path, "run", DUMMY)
    assert folder.parent == tmp_path / "profiling"
    assert re.fullmatch(r"run_tests\.run\.kinds_dummy_DUMMY_\d{8}_\d{6}", folder.name)
    assert not folder.exists()


def test_the_exit_code_of_a_profiled_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "exit_code"
    monkeypatch.delenv(profiling.EXIT_CODE_FILE, raising=False)
    profiling.record_exit_code(3)
    assert not path.exists()
    monkeypatch.setenv(profiling.EXIT_CODE_FILE, str(path))
    for code, written in [(None, "0"), (4, "4"), ("a message", "1")]:
        profiling.record_exit_code(code)
        assert path.read_text(encoding="utf-8") == written


def test_without_the_profiler_a_profile_is_a_bad_command_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(profiling, "profiler_available", lambda: False)
    with pytest.raises(profiling.ProfilerUnavailable, match=r"python 3\.15"):
        profiling.run_profiled(["list", "a:b"], tmp_path / "profile")
    out = tmp_path / "out"
    spec = _spec(tmp_path, {"n": 1})
    assert cli.main([DUMMY, str(spec), "--out", str(out), "--profile"]) == 2
    assert "needs the sampling profiler of python 3.15" in capsys.readouterr().err
    assert not out.exists()


def test_only_a_command_with_an_output_is_profiled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _spec(tmp_path, {"n": 1})
    assert cli.main(["--profile", "plan", DUMMY, str(spec)]) == 2
    assert "not 'plan'" in capsys.readouterr().err


@needs_profiler
def test_a_profiled_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPOSITORY)
    out = tmp_path / "out"
    spec = _spec(tmp_path, {"n": 4, "seed": 1})
    assert cli.main(["--profile", DUMMY, str(spec), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1 + 4 + 9 + 16
    # the child runs with the thread variables of the command line
    assert summary["omp_num_threads"] == os.environ.get("OMP_NUM_THREADS", "1")
    [folder] = (out / "profiling").iterdir()
    assert folder.name.startswith("run_tests.run.kinds_dummy_DUMMY_")
    for name in ("flamegraph.html", "heatmap", "profile.pstats", "profile.bin"):
        assert (folder / name).exists(), name
    assert "Profile Stats" in (folder / "summary.txt").read_text(encoding="utf-8")


@needs_profiler
def test_a_profiled_run_keeps_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPOSITORY)
    out = tmp_path / "out"
    spec = _spec(tmp_path, {"n": 1, "fail": True})
    assert cli.main([DUMMY, str(spec), "--out", str(out), "--profile"]) == 3
    assert list((out / "profiling").iterdir())


def test_a_sample_with_a_lone_surrogate_is_not_written() -> None:
    """A name read while it changed, which the writer of a profile cannot encode."""
    frames = [(1, [(2, 0, [("file.py", (1, 1, 0, 0), "f\ud87d", None)])])]
    assert not profiling._encodable(frames)
    assert profiling._encodable([(1, [(2, 0, [("file.py", (1, 1), "f", None)])])])
