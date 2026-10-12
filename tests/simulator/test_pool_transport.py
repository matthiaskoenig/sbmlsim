"""Parallel runs share snapshots and bound the amount of pending work."""

import multiprocessing
import os
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import ScanError, Simulator
from sbmlsim.simulator.worker import ModelSpec, ScanPointError
from tests.simulator.models import PROBE, sbml


@pytest.fixture(autouse=True)
def _spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_: "spawn")


@pytest.fixture(autouse=True)
def _worker_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake pool runs the worker in this process, which keeps its models."""
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())


def _scan() -> Scan:
    return Scan(
        Simulation(end=1, steps=1),
        [Dimension("rate", values={"k2": np.linspace(0.1, 0.8, 24)})],
    )


def test_reused_pool_gets_small_snapshot_descriptors_and_correct_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submissions: list[ModelSpec] = []
    submit = ProcessPoolExecutor.submit

    def recording(
        self: ProcessPoolExecutor, function: Callable, *args: Any, **kwargs: Any
    ) -> Future:
        if args and isinstance(args[0], ModelSpec):
            spec = args[0]
            path = Path(spec.source)
            assert path.is_file()
            if os.name != "nt":
                assert path.parent.stat().st_mode & 0o077 == 0
            submissions.append(spec)
        return submit(self, function, *args, **kwargs)

    monkeypatch.setattr(ProcessPoolExecutor, "submit", recording)
    path = tmp_path / "model.xml"
    path.write_text(sbml())
    model = RoadrunnerSBMLModel(path, selections=["time", "A", "k1"])
    simulator = Simulator(n_workers=2)
    serial = Simulator(n_workers=1).run(model, _scan())
    first = simulator.run(model, _scan())
    executor = parallel.pool(2)
    path.unlink()
    second = simulator.run(model, _scan())
    assert parallel.pool(2) is executor
    xr.testing.assert_identical(first.ds, serial.ds)
    xr.testing.assert_identical(second.ds, serial.ds)
    assert len({spec.key for spec in submissions}) == 1
    assert len({spec.source for spec in submissions}) == 2
    assert all(not Path(spec.source).exists() for spec in submissions)
    other = RoadrunnerSBMLModel(
        sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")),
        selections=["time", "A", "k1"],
    )
    changed = simulator.run(other, _scan())
    assert parallel.pool(2) is executor
    np.testing.assert_array_equal(changed["k1"], 0.1)
    assert len({spec.key for spec in submissions}) == 2
    assert all(not Path(spec.source).exists() for spec in submissions)


def test_duplicate_payloads_share_one_file_and_cleanup_after_error() -> None:
    with (
        pytest.raises(RuntimeError, match="failure"),
        parallel.shared_payloads(["one", "one", "two"]) as paths,
    ):
        assert paths[0] == paths[1]
        assert len(list(Path(paths[0]).parent.iterdir())) == 2
        raise RuntimeError("failure")
    assert all(not Path(path).exists() for path in paths)


def test_only_a_bounded_number_of_chunks_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sbmlsim.simulator.simulator as module

    class ImmediatePool:
        def submit(self, function: Callable, *args: Any) -> Future:
            future: Future = Future()
            try:
                future.set_result(function(*args))
            except Exception as error:
                future.set_exception(error)
            return future

    pool = ImmediatePool()
    monkeypatch.setattr(parallel, "pool", lambda _: pool)
    wait = module.wait
    counts: list[int] = []

    def recording(futures: set[Future], **kwargs: Any) -> Any:
        counts.append(len(futures))
        return wait(futures, **kwargs)

    monkeypatch.setattr(module, "wait", recording)
    model = RoadrunnerSBMLModel(sbml(), selections=["time", "A"])
    result = Simulator(n_workers=2).run(model, _scan())
    assert result["A"].shape == (24, 2)
    assert max(counts) <= 4
    assert len(counts) >= 2


def test_an_earlier_failure_in_unsubmitted_chunks_still_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk may hold late points while later chunks hold earlier points."""
    simulator = Simulator(n_workers=2)
    model = RoadrunnerSBMLModel(sbml(), selections=["time", "A"])
    compiled = simulator._compile(model, _scan(), None)
    template = compiled.chunks(2, "raise")[0]
    groups = [[0, 10], *[[k] for k in range(1, 24) if k != 10]]
    chunks = [
        replace(
            template,
            indices=np.array(indices),
            values={"k2": np.linspace(0.1, 0.8, 24)[indices]},
        )
        for indices in groups
    ]
    submitted: list[int] = []

    class FailingPool:
        def submit(self, function: Callable, spec: ModelSpec, chunk: Any) -> Future:
            future: Future = Future()
            index = int(chunk.indices[0])
            submitted.append(index)
            if index in (0, 5):
                failure = 10 if index == 0 else 5
                future.set_exception(ScanPointError(failure, "injected failure"))
            else:
                future.set_result(function(spec, chunk))
            return future

    monkeypatch.setattr(parallel, "pool", lambda _: FailingPool())
    with pytest.raises(ScanError, match="rate=5"):
        simulator._run_pool(compiled, chunks, 2, lambda _: None)
    assert 5 in submitted
    assert 11 not in submitted


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_workers_stop_before_snapshot_cleanup_on_unexpected_errors(
    error: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    import sbmlsim.simulator.simulator as module

    paths: list[Path] = []
    stopped: list[bool] = []

    class PendingPool:
        def submit(self, function: Callable, spec: ModelSpec, chunk: Any) -> Future:
            paths.append(Path(spec.source))
            return Future()

    monkeypatch.setattr(parallel, "pool", lambda _: PendingPool())

    def fail_wait(*args: Any, **kwargs: Any) -> Any:
        raise error("interrupted")

    def stop(executor: Any) -> None:
        assert all(path.exists() for path in paths)
        stopped.append(True)

    monkeypatch.setattr(module, "wait", fail_wait)
    monkeypatch.setattr(parallel, "stop", stop)
    with pytest.raises(error, match="interrupted"):
        Simulator(n_workers=2).run(sbml(), _scan(), keep=["A"])
    assert stopped == [True]
    assert all(not path.exists() for path in paths)


def test_serial_execution_creates_no_shared_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_payloads(*args: Any) -> Any:
        raise AssertionError("serial execution created files")

    monkeypatch.setattr(parallel, "shared_payloads", no_payloads)
    result = Simulator(n_workers=1).run(sbml(), _scan(), keep=["A"])
    assert result["A"].shape == (24, 2)
