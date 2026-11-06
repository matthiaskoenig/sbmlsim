"""The profile of a run, `python -m sbmlsim.run ... --profile`.

The profiled runs need the sampling profiler of python 3.15 and are skipped
before it.
"""

import json
import os
import re
import sys
import types
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


def test_the_folder_of_the_profile_of_a_unit(tmp_path: Path) -> None:
    folder = profiling.profile_dir(tmp_path, "unit", "fit", 3)
    assert re.fullmatch(r"unit_fit_3_\d{8}_\d{6}", folder.name)


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


class _Collector:
    """The binary collector of the profiler, as `profiling.sampling` has it."""

    def __init__(self) -> None:
        self.samples: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def collect(self, stack_frames: object, *args: object, **kwargs: object) -> str:
        self.samples.append((stack_frames, args, kwargs))
        return "written"


def _fake_profiler(
    monkeypatch: pytest.MonkeyPatch, collector: type | None
) -> types.ModuleType:
    """Put a module `profiling.sampling.binary_collector` with the collector."""
    module = types.ModuleType("profiling.sampling.binary_collector")
    if collector is not None:
        module.BinaryCollector = collector  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_the_samples_with_a_lone_surrogate_are_skipped_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = type("BinaryCollector", (_Collector,), {})
    monkeypatch.setattr(collector, "collect", _Collector.collect)
    _fake_profiler(monkeypatch, collector)
    skipped = profiling._patched_collector()
    assert skipped is not None
    instance = collector()
    assert instance.collect([("a.py", "f")], 7, flag=True) == "written"
    assert instance.collect([("a.py", "f\ud87d")], 8) is None
    assert instance.samples == [([("a.py", "f")], (7,), {"flag": True})]
    assert skipped() == 1


def test_a_profiler_without_the_collector_runs_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_profiler(monkeypatch, None)
    assert profiling._patched_collector() is None
    assert "runs unchanged" in capsys.readouterr().err


@needs_profiler
@pytest.mark.parametrize(("record", "code"), [(True, 5), (False, 1)])
def test_another_entry_point_is_profiled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    record: bool,
    code: int,
) -> None:
    monkeypatch.chdir(REPOSITORY)
    folder = tmp_path / "profile"
    arguments = ["5", "record"] if record else ["5"]
    returned = profiling.run_profiled(
        arguments, folder, module="tests.run.profiled_entry"
    )
    assert returned == code
    assert (folder / "flamegraph.html").exists()
    assert ("recorded no exit code" in caplog.text) is not record


@needs_profiler
def test_a_failed_conversion_keeps_the_code_of_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(binary: Path, folder: Path) -> None:
        raise RuntimeError("replay --flamegraph failed")

    monkeypatch.chdir(REPOSITORY)
    monkeypatch.setattr(profiling, "_convert", fail)
    returned = profiling.run_profiled(
        ["0", "record"], tmp_path / "profile", module="tests.run.profiled_entry"
    )
    assert returned == 0
    assert "was not converted" in caplog.text
    assert (tmp_path / "profile" / "profile.bin").exists()
