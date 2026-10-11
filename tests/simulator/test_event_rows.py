"""The rows of an event of the model do not depend on the history of the process.

libroadrunner 2.10 reports an event of the model in the steps of the
integrator as two rows, the state before it at `root - roottol` and the state
after it at `root + roottol`, where `roottol` is a `static const` of
`CVODEIntegrator::integrate` which the first integration of the process fixes
to `100 * 32 * eps * (|t0| + |h|)`. The tests run in a fresh process, since
`roottol` cannot be reset within one: the first integration of the process is
one over `0..10000` of an unrelated model, either with roadrunner before
sbmlsim loaded a model or with sbmlsim.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

#: the root of the repository, which the script imports the tests from
ROOT = Path(__file__).resolve().parents[2]

#: seconds a script may take, a pool which hangs fails the test
TIMEOUT = 600

#: the time of the reset of `LATE`, closer to the end 2.5 than the 7.1e-9 of
#: `roottol` after a first integration over `0..10000` of roadrunner, farther
#: than the resolution of the time, 2.5e-9
LATE_TIME = 2.5 - 3e-9

#: the times of the resets of `TWO`, both closer to the end 2.5 than `roottol`
TWO_TIMES = (2.5 - 6.5e-9, 2.5 - 3e-9)

#: the times of the doses of `SHORT` and of the changes, in simulations of one
#: time unit after them, whose resolution of the time is less than `roottol`;
#: CVODE locates the dose a few ulps after the end of the segment before the
#: change, roadrunner's last row is then `roottol` after its row before it
SHORT_TIMES = {"short_": 1.04, "short2_": 1.015}

SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np

from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml
from tests.simulator.test_executor import SAWTOOTH

DECAY = "model decay\\n  S = 10\\n  S' = -0.1 * S\\nend"
LATE = (
    "model late\\n  var X = 0\\n  var Y = 0\\n  k = 1\\n  X' = k\\n"
    "  E: at (time >= LATE_TIME): X = 0, Y = Y + 1\\nend"
)
# two resets less than `roottol` before the end
TWO = (
    "model two\\n  var X = 0\\n  var Y = 0\\n  k = 1\\n  X' = k\\n  Y' = k\\n"
    "  E1: at (time >= TWO_A): X = 0\\n  E2: at (time >= TWO_B): Y = 0\\nend"
)
# a dose of the model at the time of a change of the plan
DOSE = (
    "model dose\\n  var A = 0\\n  k = 0.1\\n  amount = 10\\n  A' = -k * A\\n"
    "  E: at (time >= 2.0): A = A + amount\\nend"
)
# the dose at a change in a short simulation, with a second state
SHORT = (
    "model short\\n  var A = 0\\n  var B = 1\\n  k = 0.1\\n  amount = 10\\n"
    "  A' = -k * A\\n  B' = -B\\n  E: at (time >= {c!r}): A = A + amount\\nend"
)


def rows(path: Path, simulation: Simulation, variables: list[str]) -> dict:
    simulator = Simulator(n_workers=1)
    result = simulator.simulate(simulator.load(path), simulation)
    return {"time": result.time.tolist()} | {
        v: np.asarray(result[v]).tolist() for v in variables
    }


def scans(
    path: Path,
    simulation: Simulation,
    values: dict,
    grid: list[float],
    variables: list[str],
    out: dict,
    key: str,
) -> None:
    scan = Scan(simulation, [Dimension("rep", values=values)])
    for n in (1, 2):
        simulator = Simulator(n_workers=n)
        model = simulator.load(path)
        native = simulator.run(model, scan, keep=variables)
        gridded = simulator.run(model, scan, time=np.array(grid), keep=variables)
        out[f"{key}native{n}"] = {"time": native.ds["time"].values.tolist()} | {
            v: native.ds[v].values.tolist() for v in variables
        }
        out[f"{key}grid{n}"] = {v: gridded.ds[v].values.tolist() for v in variables}


def main(first: str, directory: Path) -> dict:
    if first == "roadrunner":
        import roadrunner

        r = roadrunner.RoadRunner(sbml(DECAY))
        r.getIntegrator().setValue("variable_step_size", True)
        r.simulate(0, 10000)
    else:
        simulator = Simulator(n_workers=1)
        simulator.simulate(simulator.load(sbml(DECAY)), Simulation(end=10000))
    paths = {}
    for name, text in (("sawtooth", SAWTOOTH), ("late", LATE), ("two", TWO), ("dose", DOSE)):
        paths[name] = directory / f"{name}.xml"
        paths[name].write_text(sbml(text))
    steps = Simulation(end=2.5)
    dosing = Simulation(end=4.0, changes=[Change(2.0, {"k": 0.2})])
    out = {
        "rows": rows(paths["sawtooth"], steps, ["X"]),
        "late_rows": rows(paths["late"], steps, ["X"]),
        "two_rows": rows(paths["two"], steps, ["X", "Y"]),
    }
    ones = {"k": [1.0, 1.0]}
    scans(paths["sawtooth"], steps, ones, [0.5, 1.0, 1.5], ["X"], out, "")
    scans(paths["late"], steps, ones, [1.0, LATE_TIME, 2.5], ["X", "Y"], out, "late_")
    scans(paths["two"], steps, ones, [1.0, TWO_A, TWO_B, 2.5], ["X", "Y"], out, "two_")
    scans(paths["dose"], dosing, {"amount": [10.0, 10.0]}, [2.0, 3.0, 4.0], ["A"], out, "dose_")
    for key, c in SHORT_TIMES.items():
        path = directory / f"{key}.xml"
        path.write_text(sbml(SHORT.format(c=c)))
        short = Simulation(end=c + 1.0, changes=[Change(c, {"k": 0.2})])
        values = {"amount": [10.0, 10.0]}
        scans(path, short, values, [c, c + 0.5, c + 1.0], ["A"], out, key)
    from tests.fit.test_data_at_change import sawtooth_prediction

    for task in ("task_steps", "task_past"):
        folder = directory / task
        folder.mkdir()
        out[task] = sawtooth_prediction(folder, task, "at (X >= 1)").tolist()
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], Path(sys.argv[2]))))
"""
SCRIPT = SCRIPT.replace("SHORT_TIMES", repr(SHORT_TIMES))
for _name, _time in (
    ("LATE_TIME", LATE_TIME),
    ("TWO_A", TWO_TIMES[0]),
    ("TWO_B", TWO_TIMES[1]),
):
    SCRIPT = SCRIPT.replace(_name, repr(_time))

#: the results of the script by the first integration of its process
_RESULTS: dict[str, dict[str, Any]] = {}


def _run(first: str, directory: Path) -> dict[str, Any]:
    """Run the script in a fresh process, see the module."""
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, first, str(directory)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _results(first: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Get the results of the script, run once per first integration."""
    if first not in _RESULTS:
        _RESULTS[first] = _run(first, tmp_path_factory.mktemp(first)) | {"first": first}
    return _RESULTS[first]


@pytest.fixture(scope="module", params=["roadrunner", "sbmlsim"])
def after_a_long_first_integration(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """The results of the models after a first integration over `0..10000`."""
    return _results(request.param, tmp_path_factory)


def _event(time: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Get the indices of the rows of the first reset of `x`, before and after it."""
    after = int(np.flatnonzero((time > 0.5) & (x == 0.0))[0])
    return np.array([after - 1, after])


@pytest.mark.parametrize(
    ("key", "variable", "at"),
    [
        ("rows", "X", 1.0),
        ("late_rows", "X", LATE_TIME),
        ("two_rows", "X", TWO_TIMES[0]),
        ("two_rows", "Y", TWO_TIMES[1]),
    ],
)
def test_the_rows_of_an_event_have_the_time_of_the_event(
    after_a_long_first_integration: dict[str, Any], key: str, variable: str, at: float
) -> None:
    """The states before and after a reset are two rows of one time.

    CVODE locates a reset a little after its time; before, the rows were at
    `root -/+ roottol`, 7.1e-9 from it after a first integration over
    `0..10000`, and the first step of CVODE after the reset was dropped. The
    late resets are closer to the end than `roottol`: roadrunner folded the
    row after a reset into its last row and left the row before it alone.
    """
    out = after_a_long_first_integration[key]
    time, x = np.asarray(out["time"]), np.asarray(out[variable])
    rows = _event(time, x)
    assert time[rows[0]] == time[rows[1]]
    assert abs(time[rows[0]] - at) < 1e-12
    np.testing.assert_allclose(x[rows], [at, 0.0], rtol=1e-9)
    assert np.all(np.diff(time) >= 0.0)


#: the values on the grids of the scans: the sawtooth, the late reset, the
#: two resets and the dose at the change at 2, once, then decaying with 0.2
EXPECTED = {
    "": {"X": [0.5, 0.0, 0.5]},
    "late_": {"X": [1.0, 0.0, 3e-9], "Y": [0.0, 1.0, 1.0]},
    "two_": {"X": [1.0, 0.0, 3.5e-9, 6.5e-9], "Y": [1.0, TWO_TIMES[0], 0.0, 3e-9]},
    "dose_": {"A": [10.0, 10.0 * np.exp(-0.2), 10.0 * np.exp(-0.4)]},
    **{
        key: {"A": [10.0, 10.0 * np.exp(-0.1), 10.0 * np.exp(-0.2)]}
        for key in SHORT_TIMES
    },
}


@pytest.mark.parametrize("key", list(EXPECTED))
def test_a_time_at_an_event_sees_the_state_after_it_in_a_run(
    after_a_long_first_integration: dict[str, Any], key: str
) -> None:
    """A time at an event sees the state after it, and a dose at a change fires once.

    roadrunner applied the dose at 2 `roottol` after its row before it, after
    the end of the segment before the change at 2; the integration back to
    the end did not see it, and the event fired a second time after the
    change.
    """
    out = after_a_long_first_integration
    for n in (1, 2):
        for variable, expected in EXPECTED[key].items():
            np.testing.assert_allclose(
                out[f"{key}grid{n}"][variable],
                [expected] * 2,
                rtol=1e-3,
                atol=1e-9,
            )


def test_a_time_at_an_event_sees_the_state_after_it_in_a_fit(
    after_a_long_first_integration: dict[str, Any],
) -> None:
    out = after_a_long_first_integration
    for task in ("task_steps", "task_past"):
        np.testing.assert_allclose(out[task], [0.5, 0.0, 0.5], atol=1e-6)


@pytest.mark.parametrize("key", list(EXPECTED))
def test_the_result_does_not_depend_on_the_number_of_workers(
    after_a_long_first_integration: dict[str, Any], key: str
) -> None:
    """A scan with two workers has the results of a scan in one process.

    A worker's first integration is its own. After a first integration with
    sbmlsim every process has the root tolerance of the probe and the results
    are equal. After one with roadrunner before sbmlsim loaded a model, the
    process of the run lacks the first step of CVODE after the reset of the
    sawtooth, which is shorter than its root tolerance and which a worker
    has: the times of the events agree to rounding, and so do the values of
    the sawtooth on its grid, which is linear between the steps. The events
    near the end continue with new steps, which are the ones of a worker.
    """
    out = after_a_long_first_integration
    one, two = out[f"{key}native1"], out[f"{key}native2"]
    exact = out["first"] == "sbmlsim" or key != ""
    for variable in EXPECTED[key]:
        g1, g2 = out[f"{key}grid1"][variable], out[f"{key}grid2"][variable]
        if exact:
            np.testing.assert_array_equal(one["time"], two["time"])
            np.testing.assert_array_equal(one[variable], two[variable])
            np.testing.assert_array_equal(g1, g2)
        else:
            np.testing.assert_allclose(g1, g2, rtol=1e-12, atol=1e-15)
    if not exact:
        for k in range(2):
            t1, t2 = np.asarray(one["time"][k]), np.asarray(two["time"][k])
            x1, x2 = np.asarray(one["X"][k]), np.asarray(two["X"][k])
            np.testing.assert_allclose(
                t1[_event(t1, x1)], t2[_event(t2, x2)], rtol=0.0, atol=1e-15
            )


@pytest.mark.parametrize("key", list(EXPECTED))
def test_the_result_does_not_depend_on_the_first_integration_of_the_process(
    tmp_path_factory: pytest.TempPathFactory, key: str
) -> None:
    """A process in which roadrunner integrated first has the results of one
    in which sbmlsim did, but for the step of CVODE after the reset of the
    sawtooth, see `test_the_result_does_not_depend_on_the_number_of_workers`."""
    rr = _results("roadrunner", tmp_path_factory)
    sb = _results("sbmlsim", tmp_path_factory)
    for n in (1, 2):
        for variable in EXPECTED[key]:
            if key == "":
                np.testing.assert_allclose(
                    rr[f"grid{n}"][variable],
                    sb[f"grid{n}"][variable],
                    rtol=1e-12,
                    atol=1e-15,
                )
            else:
                np.testing.assert_array_equal(
                    rr[f"{key}native{n}"][variable], sb[f"{key}native{n}"][variable]
                )
                np.testing.assert_array_equal(
                    rr[f"{key}grid{n}"][variable], sb[f"{key}grid{n}"][variable]
                )
