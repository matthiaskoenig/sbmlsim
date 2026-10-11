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

SCRIPT = """
import json
import sys
from pathlib import Path

import numpy as np

from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml
from tests.simulator.test_executor import SAWTOOTH

DECAY = "model decay\\n  S = 10\\n  S' = -0.1 * S\\nend"
LATE = "model late\\n  var X = 0\\n  k = 1\\n  X' = k\\n  E: at (time >= {late!r}): X = 0\\nend"


def rows(path: Path) -> dict:
    result = Simulator(n_workers=1).simulate(
        Simulator(n_workers=1).load(path), Simulation(end=2.5)
    )
    return {"time": result.time.tolist(), "X": np.asarray(result["X"]).tolist()}


def scans(path: Path, grid: list[float], out: dict, key: str) -> None:
    scan = Scan(Simulation(end=2.5), [Dimension("rep", values={"k": [1.0, 1.0]})])
    for n in (1, 2):
        simulator = Simulator(n_workers=n)
        model = simulator.load(path)
        native = simulator.run(model, scan, keep=["X"])
        gridded = simulator.run(model, scan, time=np.array(grid), keep=["X"])
        out[f"{key}native{n}"] = {
            "time": native.ds["time"].values.tolist(),
            "X": native.ds["X"].values.tolist(),
        }
        out[f"{key}grid{n}"] = gridded.ds["X"].values.tolist()


def main(first: str, directory: Path) -> dict:
    if first == "roadrunner":
        import roadrunner

        r = roadrunner.RoadRunner(sbml(DECAY))
        r.getIntegrator().setValue("variable_step_size", True)
        r.simulate(0, 10000)
    else:
        simulator = Simulator(n_workers=1)
        simulator.simulate(simulator.load(sbml(DECAY)), Simulation(end=10000))
    sawtooth, late = directory / "sawtooth.xml", directory / "late.xml"
    sawtooth.write_text(sbml(SAWTOOTH))
    late.write_text(sbml(LATE))
    out = {"rows": rows(sawtooth), "late_rows": rows(late)}
    scans(sawtooth, [0.5, 1.0, 1.5], out, "")
    scans(late, [1.0, {late!r}, 2.5], out, "late_")
    from tests.fit.test_data_at_change import sawtooth_prediction

    for task in ("task_steps", "task_past"):
        folder = directory / task
        folder.mkdir()
        out[task] = sawtooth_prediction(folder, task, "at (X >= 1)").tolist()
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], Path(sys.argv[2]))))
""".replace("{late!r}", repr(LATE_TIME))


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


@pytest.fixture(scope="module", params=["roadrunner", "sbmlsim"])
def after_a_long_first_integration(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """The results of the sawtooth after a first integration over `0..10000`."""
    directory = tmp_path_factory.mktemp(request.param)
    return _run(request.param, directory) | {"first": request.param}


def _event(time: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Get the indices of the rows of the first reset, before and after it."""
    after = int(np.flatnonzero((time > 0.5) & (x == 0.0))[0])
    return np.array([after - 1, after])


@pytest.mark.parametrize(("key", "at"), [("rows", 1.0), ("late_rows", LATE_TIME)])
def test_the_rows_of_an_event_have_the_time_of_the_event(
    after_a_long_first_integration: dict[str, Any], key: str, at: float
) -> None:
    """The states before and after a reset are two rows of one time.

    CVODE locates a reset a little after its time; before, the rows were at
    `root -/+ roottol`, 7.1e-9 from it after a first integration over
    `0..10000`, and the first step of CVODE after the reset was dropped. The
    late reset is closer to the end than `roottol`, roadrunner folded its row
    after it into the last row and left the row before it alone.
    """
    out = after_a_long_first_integration[key]
    time, x = np.asarray(out["time"]), np.asarray(out["X"])
    rows = _event(time, x)
    assert time[rows[0]] == time[rows[1]]
    assert abs(time[rows[0]] - at) < 1e-12
    np.testing.assert_allclose(x[rows], [at, 0.0], rtol=1e-9)
    assert np.all(np.diff(time) >= 0.0)


def test_a_time_at_an_event_sees_the_state_after_it_in_a_run_and_a_fit(
    after_a_long_first_integration: dict[str, Any],
) -> None:
    out = after_a_long_first_integration
    for n in (1, 2):
        np.testing.assert_allclose(out[f"grid{n}"], [[0.5, 0.0, 0.5]] * 2, atol=1e-9)
        np.testing.assert_allclose(
            out[f"late_grid{n}"], [[1.0, 0.0, 3e-9]] * 2, rtol=1e-3, atol=1e-15
        )
    for task in ("task_steps", "task_past"):
        np.testing.assert_allclose(out[task], [0.5, 0.0, 0.5], atol=1e-6)


@pytest.mark.parametrize("key", ["", "late_"])
def test_the_result_does_not_depend_on_the_number_of_workers(
    after_a_long_first_integration: dict[str, Any], key: str
) -> None:
    """A scan with two workers has the results of a scan in one process.

    A worker's first integration is its own. After a first integration with
    sbmlsim every process has the root tolerance of the probe and the results
    are equal. After one with roadrunner before sbmlsim loaded a model, the
    process of the run lacks a first step of CVODE after an event which is
    shorter than its root tolerance, which a worker has: the times of the
    events agree to rounding, and the values on a grid within the tolerances
    of the integrator, here to rounding.
    """
    out = after_a_long_first_integration
    one, two = out[f"{key}native1"], out[f"{key}native2"]
    if out["first"] == "sbmlsim":
        np.testing.assert_array_equal(one["time"], two["time"])
        np.testing.assert_array_equal(one["X"], two["X"])
        np.testing.assert_array_equal(out[f"{key}grid1"], out[f"{key}grid2"])
        return
    for k in range(2):
        t1, t2 = np.asarray(one["time"][k]), np.asarray(two["time"][k])
        x1, x2 = np.asarray(one["X"][k]), np.asarray(two["X"][k])
        np.testing.assert_allclose(
            t1[_event(t1, x1)], t2[_event(t2, x2)], rtol=0.0, atol=1e-15
        )
    np.testing.assert_allclose(
        out[f"{key}grid1"], out[f"{key}grid2"], rtol=1e-12, atol=1e-15
    )
