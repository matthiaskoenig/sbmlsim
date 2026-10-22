"""The rows of an event of the model do not depend on the history of the process.

libroadrunner reports an event of the model in the steps of the integrator
as two rows, the state before it a little before its time and the state after
it a little after it; up to libroadrunner 2.10 the first integration of the
process fixed that distance. sbmlsim tells the rows by the markers of the
events, see `sbmlsim.model.event_rows`. The script runs in two fresh
processes at once: the first integration of the process is one over
`0..10000` of an unrelated model, either with roadrunner before sbmlsim
loaded a model or with sbmlsim. Every check is a subtest of one test, so the
processes run once per session, also with the workers of pytest-xdist.
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

#: the time of the reset of `LATE`, closer to the end 2.5 than 7.1e-9, the
#: distance of the rows of an event after a first integration over `0..10000`
#: with libroadrunner 2.10, farther than the resolution of the time, 2.5e-9
LATE_TIME = 2.5 - 3e-9

#: the times of the resets of `TWO`, both closer to the end 2.5 than 7.1e-9
TWO_TIMES = (2.5 - 6.5e-9, 2.5 - 3e-9)

#: the times of the doses of `SHORT` and of the changes, in simulations of one
#: time unit after them, whose resolution of the time is less than the
#: distance of the rows of an event; CVODE locates the dose a few ulps after
#: the end of the segment before the change
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
# two resets less than 7.1e-9 before the end
TWO = (
    "model two\\n  var X = 0\\n  var Y = 0\\n  k = 1\\n  X' = k\\n  Y' = k\\n"
    "  E1: at (time >= TWO_A): X = 0\\n  E2: at (time >= TWO_B): Y = 0\\nend"
)
# a dose of the model at the time of a change of the plan
DOSE = (
    "model dose\\n  var A = 0\\n  k = 0.1\\n  amount = 10\\n  A' = -k * A\\n"
    "  E: at (time >= 2.0): A = A + amount\\nend"
)
# a dose at 0 of a simulation from a negative start
ZERO = (
    "model zero\\n  var A = 1\\n  var Y = 0\\n  k = 0.5\\n  A' = -k * A\\n"
    "  E: at (time >= 0): A = A + 10, Y = Y + 1\\nend"
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
    zero = directory / "zero.xml"
    zero.write_text(sbml(ZERO))
    for name, simulation in (
        ("end", Simulation(start=-4, end=0)),
        ("end_times", Simulation(start=-4, end=0, times=[-4, -2, 0])),
        ("change", Simulation(start=-4, end=4, changes=[Change(0, {"A": 2.0})])),
        ("change_end", Simulation(start=-4, end=0, changes=[Change(0, {"A": 2.0})])),
    ):
        out[f"zero_{name}"] = rows(zero, simulation, ["A", "Y"])
    # the second reset of the sawtooth and the dose at the end
    for name, variable in (("sawtooth", "X"), ("dose", "A")):
        for output, times in (("steps", None), ("times", [0.0, 1.0, 2.0])):
            end = Simulation(end=2.0, times=times)
            out[f"end_{name}_{output}"] = rows(paths[name], end, [variable])
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

#: the first integrations of the processes of the script
FIRSTS = ("roadrunner", "sbmlsim")


def _run(directory: Path) -> dict[str, dict[str, Any]]:
    """Run the script in a fresh process per first integration, at once."""
    processes = {}
    for first in FIRSTS:
        folder = directory / first
        folder.mkdir()
        processes[first] = subprocess.Popen(
            [sys.executable, "-c", SCRIPT, first, str(folder)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
    results: dict[str, dict[str, Any]] = {}
    try:
        for first, process in processes.items():
            stdout, stderr = process.communicate(timeout=TIMEOUT)
            assert process.returncode == 0, stderr
            results[first] = json.loads(stdout.strip().splitlines()[-1]) | {
                "first": first
            }
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.communicate()
    return results


def _event(time: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Get the indices of the rows of the first reset of `x`, before and after it."""
    after = int(np.flatnonzero((time > 0.5) & (x == 0.0))[0])
    return np.array([after - 1, after])


#: the rows of a reset: the result, the variable and the time of the reset
ROWS = [
    ("rows", "X", 1.0),
    ("late_rows", "X", LATE_TIME),
    ("two_rows", "X", TWO_TIMES[0]),
    ("two_rows", "Y", TWO_TIMES[1]),
]


def check_rows_of_an_event(
    out: dict[str, Any], key: str, variable: str, at: float
) -> None:
    """The states before and after a reset are two rows of one time.

    CVODE locates a reset a little after its time; with libroadrunner 2.10
    the rows were 7.1e-9 from it after a first integration over `0..10000`,
    and the first step of CVODE after the reset was dropped. The late resets
    are closer to the end than that: roadrunner folded the row after a reset
    into its last row and left the row before it alone.
    """
    time, x = np.asarray(out[key]["time"]), np.asarray(out[key][variable])
    rows = _event(time, x)
    assert time[rows[0]] == time[rows[1]]
    assert abs(time[rows[0]] - at) < 1e-12
    np.testing.assert_allclose(x[rows], [at, 0.0], rtol=1e-9)
    assert np.all(np.diff(time) >= 0.0)


#: the states before and after the event at the end 2 of a simulation
END = {"sawtooth": ("X", 1.0, 0.0), "dose": ("A", 0.0, 10.0)}


def check_an_event_at_the_end(out: dict[str, Any], name: str) -> None:
    """The steps have the second reset and the dose at the end as two rows.

    An output of times has the state after it at the end.
    """
    variable, before, after = END[name]
    steps = out[f"end_{name}_steps"]
    assert steps["time"][-2:] == [2.0, 2.0]
    assert steps["time"][-3] < 2.0
    np.testing.assert_allclose(steps[variable][-2:], [before, after], atol=1e-9)
    times = out[f"end_{name}_times"]
    assert times["time"] == [0.0, 1.0, 2.0]
    np.testing.assert_allclose(times[variable][-1], after, atol=1e-9)


#: `A` at 0 of the model `ZERO`, before the dose
A_AT_0 = float(np.exp(-2.0))

#: the states at 0 of the model `ZERO`: before and after the dose at the end 0,
#: an output time at the end 0, a change at 0 and a change at the end 0
ZERO_ROWS = {
    "end": [A_AT_0, A_AT_0 + 10.0],
    "end_times": [A_AT_0 + 10.0],
    "change": [A_AT_0, 12.0],
    "change_end": [A_AT_0, 12.0],
}


def check_a_dose_at_0(out: dict[str, Any], name: str) -> None:
    """A dose at 0 from a negative start fires once, after a change at 0.

    The integration stopped one unit in the last place before 0, a subnormal
    time, at which CVODE failed or gave `NaN`.
    """
    rows = out[f"zero_{name}"]
    time, a = np.asarray(rows["time"]), np.asarray(rows["A"])
    np.testing.assert_allclose(a[time == 0.0], ZERO_ROWS[name], rtol=1e-4)
    assert rows["Y"][-1] == 1.0


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


def check_a_time_at_an_event_in_a_run(out: dict[str, Any], key: str) -> None:
    """A time at an event sees the state after it, and a dose at a change fires once.

    roadrunner applied the dose at 2 a little after its row before it, after
    the end of the segment before the change at 2; the integration back to
    the end did not see it, and the event fired a second time after the
    change.
    """
    for n in (1, 2):
        for variable, expected in EXPECTED[key].items():
            np.testing.assert_allclose(
                out[f"{key}grid{n}"][variable],
                [expected] * 2,
                rtol=1e-3,
                atol=1e-9,
            )


def check_a_time_at_an_event_in_a_fit(out: dict[str, Any]) -> None:
    """A time of the data at an event sees the state after it in a fit."""
    for task in ("task_steps", "task_past"):
        np.testing.assert_allclose(out[task], [0.5, 0.0, 0.5], atol=1e-6)


def check_the_number_of_workers(out: dict[str, Any], key: str) -> None:
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


def check_the_first_integration(
    rr: dict[str, Any], sb: dict[str, Any], key: str
) -> None:
    """A process in which roadrunner integrated first has the results of one
    in which sbmlsim did, but for the step of CVODE after the reset of the
    sawtooth, see `check_the_number_of_workers`."""
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


def test_the_rows_of_events_do_not_depend_on_the_history_of_the_process(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    """The checks of the rows of events after a long first integration."""
    results = _run(tmp_path)
    for first, out in results.items():
        for key, variable, at in ROWS:
            with subtests.test("the rows of an event", first=first, key=key, at=at):
                check_rows_of_an_event(out, key, variable, at)
        for name in END:
            with subtests.test("an event at the end", first=first, name=name):
                check_an_event_at_the_end(out, name)
        for name in ZERO_ROWS:
            with subtests.test("a dose at 0", first=first, name=name):
                check_a_dose_at_0(out, name)
        for key in EXPECTED:
            with subtests.test("a time at an event in a run", first=first, key=key):
                check_a_time_at_an_event_in_a_run(out, key)
            with subtests.test("the number of workers", first=first, key=key):
                check_the_number_of_workers(out, key)
        with subtests.test("a time at an event in a fit", first=first):
            check_a_time_at_an_event_in_a_fit(out)
    for key in EXPECTED:
        with subtests.test("the first integration", key=key):
            check_the_first_integration(results["roadrunner"], results["sbmlsim"], key)
