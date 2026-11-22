"""The speed of the scan core: `pytest -m benchmark -n 0 -s tests/simulator/test_benchmark.py`.

A benchmark compares times measured in the same process, so it does not
depend on the speed of the machine: it fails when the scan core costs more
than the work it does, or the pool does not pay. The times of 0.8.5 are
reported.
"""

import os
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import sbmlsim.simulator.simulator as simulator_module
from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import MIDAZOLAM_SBML, REPRESSILATOR_SBML
from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.executor import execute
from tests.simulator.models import roadrunner_model, roadrunner_simulator

pytestmark = pytest.mark.benchmark

#: ms per 0.8.5 `simulate` of the repressilator (end=100, steps=100),
#: measured in the pre-flight of the scan core on the machine of the
#: benchmarks; reported, the time of the machine is not asserted
BASE_SIMULATE_MS = 0.711

#: the most `Simulator.simulate` may cost more than the work of 0.8.5
#: `simulate`, as a fraction of it: the settings of the integrator, which it
#: compares with the ones of roadrunner, and the check of the plan against the
#: units of the model, about 15 to 30 µs per call. Measured 2.0 to 2.5 % for
#: the repressilator and 2.6 to 3.2 % for midazolam, a simulation of 0.7 and
#: 0.9 ms; the limit leaves room for the noise of the machine and fails for a
#: check which computes the units of the model again on every call, 0.1 ms or
#: 10 to 16 % for midazolam
SIMULATE_OVERHEAD = 0.05

#: s per 0.8.5 `run_scan` of the 1e3 points of `_scan(1000)`,
#: measured in the same pre-flight; reported, not asserted
BASE_SCAN_1E3_S = 0.81

#: the most a serial scan may cost more than the execution of the plans of
#: its points, i.e. the chunks, the stacking and the assembly of the result
SCAN_OVERHEAD = 0.05


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


#: the models of `test_a_simulation`: a model, its selections and a simulation
SIMULATIONS = {
    "repressilator": (
        REPRESSILATOR_SBML,
        ["time", "PX", "PY", "PZ"],
        Simulation(end=100, steps=100),
    ),
    "midazolam": (
        MIDAZOLAM_SBML,
        ["time", "[Cve_mid]", "[Cve_mid1oh]", "Aurine_mid1oh"],
        Simulation(end=24 * 60, steps=100, preinit_changes={"PODOSE_mid": 10.0}),
    ),
}


def _medians(
    functions: dict[str, Callable[[], object]], n: int = 600
) -> dict[str, float]:
    """Get the median time of every function, called in turn.

    The functions are called one after the other, n times, so that a change of
    the load of the machine hits every one of them alike.
    """
    for function in functions.values():
        function()
    times: dict[str, list[float]] = {key: [] for key in functions}
    for _ in range(n):
        for key, function in functions.items():
            start = time.perf_counter()
            function()
            times[key].append(time.perf_counter() - start)
    return {key: statistics.median(values) for key, values in times.items()}


@pytest.mark.parametrize("name", list(SIMULATIONS))
def test_a_simulation(name: str) -> None:
    """`Simulator.simulate` costs little more than the execution of its plan.

    0.8.5 `simulate` compiled the simulation and executed the plan,
    `Simulator.simulate` adds the settings of the integrator, which it sets
    only when they changed, and the check of the plan against the model,
    whose units it computes once per model: a simulation costs at most
    `SIMULATE_OVERHEAD` more than the compile and the execution of its plan,
    and a plan at most that more than its execution. The times are measured
    in the same process, the result does not depend on the speed of the
    machine.

    The comparison is against `executor.execute`, the executor of roadrunner,
    so the simulator is roadrunner's and not the default engine, which is the
    engine core since #607 and which answers the same simulation with its own
    executor (`roadrunner_simulator`).
    """
    source, selections, simulation = SIMULATIONS[name]
    model = RoadrunnerSBMLModel(source=source)
    model.set_selections(selections)
    simulator = roadrunner_simulator(n_workers=1)
    plan = simulator.compile(simulator.load(model), simulation)
    selected = model.selections or []
    times = _medians(
        {
            "simulation": lambda: simulator.simulate(model, simulation),
            "0.8.5": lambda: execute(
                simulator.compile(model, simulation), model, selected
            ),
            "plan": lambda: simulator.simulate(model, plan),
            "execute": lambda: execute(plan, model, selected),
        }
    )
    simulation_overhead = times["simulation"] / times["0.8.5"] - 1
    plan_overhead = times["plan"] / times["execute"] - 1
    base = f" (0.8.5: {BASE_SIMULATE_MS:.3f} ms)" if name == "repressilator" else ""
    print(
        f"\nSimulator.simulate of the {name}: {times['simulation'] * 1e3:.3f} ms"
        f"{base}, {simulation_overhead * 100:+.1f} % against compile and execute; "
        f"of its plan {times['plan'] * 1e3:.3f} ms, {plan_overhead * 100:+.1f} % "
        f"against execute ({(times['plan'] - times['execute']) * 1e6:+.0f} µs)"
    )
    assert simulation_overhead <= SIMULATE_OVERHEAD
    assert plan_overhead <= SIMULATE_OVERHEAD


def test_the_serial_time_of_a_scan_of_1e3_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serial scan is compiled once and costs little more than its points.

    The work of a point is its plan with the value of the point,
    `Plan.with_values`, and its execution, `execute`: a serial scan of 1e3
    points costs at most `SCAN_OVERHEAD` more than that work for every point,
    measured in turn in the same process. The time of 0.8.5 is reported.

    The work of a point is `executor.execute`, so the simulator is
    roadrunner's and not the default engine core, whose scan is one batch of
    the engine and not a call of `execute` per point (#607).
    """
    calls: list[int] = []
    compile_simulation = simulator_module.compile_simulation

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return compile_simulation(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(simulator_module, "compile_simulation", counting)
    model, scan = _model(), _scan(1000)
    simulator = roadrunner_simulator(n_workers=1)
    simulator.run(model, scan)
    assert len(calls) == 1
    monkeypatch.undo()

    loaded = roadrunner_model(simulator.load(model))
    plan = simulator.compile(loaded, scan.simulation)
    selections = loaded.selections or []
    values = [float(v) for v in scan.dimensions[0].values["n"]]

    def points() -> None:
        for value in values:
            execute(plan.with_values({"n": value}), loaded, selections)

    times = _medians(
        {"scan": lambda: simulator.run(model, scan), "points": points}, n=7
    )
    overhead = times["scan"] / times["points"] - 1
    print(
        f"\na serial scan of 1e3 points: {times['scan']:.2f} s, "
        f"{overhead * 100:+.1f} % against the execution of its points "
        f"({times['points']:.2f} s); 0.8.5 run_scan: {BASE_SCAN_1E3_S:.2f} s"
    )
    assert overhead <= SCAN_OVERHEAD


def test_a_scan_of_1e4_points_is_faster_on_4_workers() -> None:
    model, scan = _model(), _scan(10_000)
    start = time.perf_counter()
    serial = Simulator(n_workers=1).run(model, scan)
    t1 = time.perf_counter() - start
    # the start of the workers is paid once per process
    _start_every_worker(4)
    start = time.perf_counter()
    pooled = Simulator(n_workers=4).run(model, scan, progress=False)
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


#: the least a batch of the engine core on four threads must do against one
#: thread; the batch of the measurement of #612 reached 3.1 of 4 threads on
#: performance cores of their own and 2.9 unpinned, and a machine of four
#: busy cores is still well above this
BATCH_SPEEDUP_4 = 2.0


def _core_batch(n_points: int, n_times: int = 101) -> tuple[Any, list[Any]]:
    """Get a model of the engine core and the requests of a batch of it.

    The requests are built once and handed to the executors of the thread
    counts under test, so what is measured is the batch of the engine and
    none of the python of a run.
    """
    core = pytest.importorskip("sbmlsim_core")
    assert core is not None
    from sbmlsim.engine import Engine
    from sbmlsim.model.model_sbml import load_model
    from sbmlsim.resources import HCTZ_SBML
    from sbmlsim.simulator import executor_core
    from sbmlsim.simulator.plan import compile_simulation

    model = load_model(HCTZ_SBML, Engine.CORE)
    # load_model gives the model of the engine, whose requests the executor
    # of the engine core builds; the bindings are Any, so is the model here
    core_model: Any = model
    end = 240.0
    simulation = Simulation(
        start=0, end=end, times=np.linspace(0, end, n_times).tolist()
    )
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    columns = executor_core.output_columns(["time", "Cve_hctz", "Aurine_hctz"])
    reference = float(model.value("BW"))
    requests = [
        executor_core.request_of(
            plan.with_values({"BW": reference * (1.0 + 1e-6 * k)}),
            core_model,
            columns,
        )
        for k in range(n_points)
    ]
    return core_model, requests


def test_the_batch_of_the_engine_core_scales_over_its_threads() -> None:
    """The threads of one batch of the engine core, measured on the batch alone.

    The parallel efficiency of a batch is the efficiency of every scan, fit
    and propagation of the engine core, which run in one process on the
    threads of the engine. The measurement is of the batch and not of a run:
    the requests are built once, the same batch runs on one, two, four and
    eight threads, and the rate is the best of three timed stretches, so a
    stretch which the operating system interrupts does not count.

    Measured for #612 on eight performance cores of a 12th Gen Core i7-12700,
    pinned to a core each: 88 % of two threads, 78 % of four and 58 % of
    eight, while eight processes of one thread each on the same cores reach
    91 % and eight threads of python driving a batch of one thread each 68 %,
    so the loss is in the batch and in sharing the process, not in the
    machine (sbmlsim-core#63).
    """
    model, requests = _core_batch(1_000)
    rates: dict[int, float] = {}
    for threads in (1, 2, 4, 8):
        executor = model.batch_executor(threads)
        # the instance of a thread is made at its first point and kept
        executor.simulate_many(requests[:4])
        best = 0.0
        for _ in range(3):
            done, start = 0, time.perf_counter()
            while time.perf_counter() - start < 1.0:
                executor.simulate_many(requests)
                done += len(requests)
            best = max(best, done / (time.perf_counter() - start))
        rates[threads] = best
    one = rates[1]
    print(
        f"\na batch of {len(requests)} points of the HCTZ model, "
        f"{os.process_cpu_count()} CPUs:"
    )
    for threads, rate in rates.items():
        print(
            f"  {threads:>2} thread(s): {rate:>8.0f} points/s, "
            f"{rate / one:4.2f}x, efficiency {rate / one / threads:4.0%}"
        )
    if parallel.resolve_workers(None, len(requests)) < 4:
        pytest.skip("a machine of fewer than four CPUs does not spread a batch")
    assert rates[4] / one >= BATCH_SPEEDUP_4
