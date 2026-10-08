"""The speed of the scan core: `pytest -m benchmark -n 0 -s tests/simulator/test_benchmark.py`.

The times are those of the machine they were measured on: a benchmark fails
when the scan core became slower there than 0.8.5, or the pool does not pay.
"""

import os
import time

import numpy as np
import pytest

import sbmlsim.simulator.simulator as simulator_module
from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator

pytestmark = pytest.mark.benchmark

#: ms per `SimulatorSerial.simulate` of the repressilator (end=100, steps=100)
#: in 0.8.5, measured in the pre-flight of the scan core on the machine of the
#: benchmarks
BASE_SIMULATE_MS = 0.711

#: s per `SimulatorSerial.run_scan` of the 1e3 points of `_scan(1000)` in
#: 0.8.5, measured in the same pre-flight
BASE_SCAN_1E3_S = 0.81


def _model() -> RoadrunnerSBMLModel:
    model = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML)
    model.set_selections(["time", "PX", "PY", "PZ"])
    return model


def _scan(n: int) -> Scan:
    return Scan(
        Simulation(end=100, steps=100),
        [Dimension("dim_n", values={"n": np.linspace(1.5, 4.0, n)})],
    )


def _pid_after(seconds: float) -> int:
    """Answer the process id of the worker after a while."""
    time.sleep(seconds)
    return os.getpid()


def _start_every_worker(n_workers: int) -> None:
    """Start the kept pool of a number of workers with every one of its workers.

    The pool starts a worker when a task finds no idle one, so as many tasks
    which take a while start every worker, and each runs one of them.
    """
    pids = set(parallel.pool(n_workers).map(_pid_after, [0.5] * n_workers))
    assert len(pids) == n_workers


def test_a_simulation() -> None:
    """`Simulator.simulate` is not slower than `SimulatorSerial.simulate` of 0.8.5."""
    simulator, model = Simulator(n_workers=1), _model()
    simulation = Simulation(end=100, steps=100)
    simulator.simulate(model, simulation)
    # the best of five rounds, as timeit, so that the load of the machine
    # does not count
    rounds = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(100):
            simulator.simulate(model, simulation)
        rounds.append((time.perf_counter() - start) / 100)
    elapsed = min(rounds)
    print(
        f"\nSimulator.simulate: {elapsed * 1e3:.3f} ms per simulation "
        f"(0.8.5: {BASE_SIMULATE_MS:.3f} ms)"
    )
    assert elapsed * 1e3 <= 1.1 * BASE_SIMULATE_MS


def test_the_serial_time_of_a_scan_of_1e3_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scan is compiled once; its time is reported against 0.8.5."""
    calls: list[int] = []
    compile_simulation = simulator_module.compile_simulation

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return compile_simulation(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(simulator_module, "compile_simulation", counting)
    model = _model()
    start = time.perf_counter()
    Simulator(n_workers=1).run(model, _scan(1000))
    elapsed = time.perf_counter() - start
    print(
        f"\na serial scan of 1e3 points: {elapsed:.2f} s "
        f"(0.8.5 run_scan: {BASE_SCAN_1E3_S:.2f} s)"
    )
    assert len(calls) == 1
    # a regression of the serial path, beyond the noise of the machine
    assert elapsed <= 1.25 * BASE_SCAN_1E3_S


def test_a_scan_of_1e4_points_is_faster_on_4_workers() -> None:
    model, scan = _model(), _scan(10_000)
    start = time.perf_counter()
    serial = Simulator(n_workers=1).run(model, scan)
    t1 = time.perf_counter() - start
    # the start of the workers is paid once per process
    _start_every_worker(4)
    start = time.perf_counter()
    pooled = Simulator(n_workers=4).run(model, scan)
    t4 = time.perf_counter() - start
    print(f"\n1e4 points: {t1:.2f} s on 1 worker, {t4:.2f} s on 4, {t1 / t4:.2f}x")
    np.testing.assert_array_equal(pooled["PX"].values, serial["PX"].values)
    assert t1 / t4 >= 2.5


def test_the_pool_pays_from_the_threshold() -> None:
    """A scan of `POOL_THRESHOLD` points is not slower in the pool than serially.

    `n_workers=None` runs such a scan on every CPU. The pool and its workers
    are started, which is paid once per process, but the workers do not hold
    the model yet: each loads it in the first run, as for every new model. The
    best of three runs, each on a new pool, is compared.
    """
    n = parallel.POOL_THRESHOLD
    workers = parallel.resolve_workers(None, n)
    if workers == 1:
        pytest.skip("a machine with one CPU runs every scan serially")
    model, scan = _model(), _scan(n)
    Simulator(n_workers=1).run(model, scan)
    serial: list[float] = []
    loading: list[float] = []
    loaded: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        expected = Simulator(n_workers=1).run(model, scan)
        serial.append(time.perf_counter() - start)
        parallel.shutdown()
        _start_every_worker(workers)
        for times in (loading, loaded):
            start = time.perf_counter()
            pooled = Simulator(n_workers=None).run(model, scan, progress=False)
            times.append(time.perf_counter() - start)
            np.testing.assert_array_equal(pooled["PX"].values, expected["PX"].values)
    print(
        f"\n{n} points: {min(serial) * 1e3:.0f} ms on 1 worker; on {workers} "
        f"workers {min(loading) * 1e3:.0f} ms with the load of the model, "
        f"{min(loaded) * 1e3:.0f} ms in the next run"
    )
    assert min(loading) <= min(serial)
