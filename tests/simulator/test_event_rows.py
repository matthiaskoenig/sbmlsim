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
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

#: the root of the repository, which the script imports the tests from
ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from sbmlsim.simulation import Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml
from tests.simulator.test_executor import SAWTOOTH

DECAY = "model decay\\n  S = 10\\n  S' = -0.1 * S\\nend"


def main(first: str) -> dict:
    if first == "roadrunner":
        import roadrunner

        r = roadrunner.RoadRunner(sbml(DECAY))
        r.getIntegrator().setValue("variable_step_size", True)
        r.simulate(0, 10000)
    else:
        simulator = Simulator(n_workers=1)
        simulator.simulate(simulator.load(sbml(DECAY)), Simulation(end=10000))
    directory = Path(tempfile.mkdtemp())
    path = directory / "sawtooth.xml"
    path.write_text(sbml(SAWTOOTH))
    out = {}
    simulator = Simulator(n_workers=1)
    model = simulator.load(path)
    rows = simulator.simulate(model, Simulation(end=2.5))
    out["time"] = rows.time.tolist()
    out["X"] = np.asarray(rows["X"]).tolist()
    grid = np.array([0.5, 1.0, 1.5])
    scan = Scan(Simulation(end=2.5), [Dimension("rep", values={"k": [1.0, 1.0]})])
    for n in (1, 2):
        simulator = Simulator(n_workers=n)
        model = simulator.load(path)
        native = simulator.run(model, scan, keep=["X"])
        gridded = simulator.run(model, scan, time=grid, keep=["X"])
        out[f"native{n}"] = {
            "time": native.ds["time"].values.tolist(),
            "X": native.ds["X"].values.tolist(),
        }
        out[f"grid{n}"] = gridded.ds["X"].values.tolist()
    from tests.fit.test_data_at_change import sawtooth_prediction

    for task in ("task_steps", "task_past"):
        prediction = sawtooth_prediction(Path(tempfile.mkdtemp()), task, "at (X >= 1)")
        out[task] = prediction.tolist()
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])))
"""


def _run(first: str) -> dict[str, Any]:
    """Run the script in a fresh process, see the module."""
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, first],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module", params=["roadrunner", "sbmlsim"])
def after_a_long_first_integration(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The results of the sawtooth after a first integration over `0..10000`."""
    return _run(request.param) | {"first": request.param}


def _event(time: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Get the indices of the rows of the reset at 1, before and after it."""
    after = int(np.flatnonzero((time > 0.5) & (x == 0.0))[0])
    return np.array([after - 1, after])


def test_the_rows_of_an_event_have_the_time_of_the_event(
    after_a_long_first_integration: dict[str, Any],
) -> None:
    """The states before and after the reset at 1 are two rows of one time.

    CVODE locates the reset a little after 1; before, the rows were at
    `root -/+ roottol`, 7.1e-9 from it after a first integration over
    `0..10000`, and the first step of CVODE after the reset was dropped.
    """
    out = after_a_long_first_integration
    time, x = np.asarray(out["time"]), np.asarray(out["X"])
    rows = _event(time, x)
    assert time[rows[0]] == time[rows[1]]
    assert abs(time[rows[0]] - 1.0) < 1e-12
    np.testing.assert_allclose(x[rows], [1.0, 0.0], atol=1e-12)
    assert np.all(np.diff(time) >= 0.0)


def test_a_time_at_an_event_sees_the_state_after_it_in_a_run_and_a_fit(
    after_a_long_first_integration: dict[str, Any],
) -> None:
    out = after_a_long_first_integration
    for n in (1, 2):
        np.testing.assert_allclose(out[f"grid{n}"], [[0.5, 0.0, 0.5]] * 2, atol=1e-9)
    for task in ("task_steps", "task_past"):
        np.testing.assert_allclose(out[task], [0.5, 0.0, 0.5], atol=1e-6)


def test_the_result_does_not_depend_on_the_number_of_workers(
    after_a_long_first_integration: dict[str, Any],
) -> None:
    """A scan with two workers has the results of a scan in one process.

    A worker's first integration is its own. After a first integration with
    sbmlsim every process has the root tolerance of the probe and the results
    are equal; after one with roadrunner before sbmlsim loaded a model, the
    process of the run lacks the first step of CVODE after an event which is
    shorter than its root tolerance, which a worker has, and the times of the
    events and the values on a grid agree to rounding.
    """
    out = after_a_long_first_integration
    one, two = out["native1"], out["native2"]
    if out["first"] == "sbmlsim":
        np.testing.assert_array_equal(one["time"], two["time"])
        np.testing.assert_array_equal(one["X"], two["X"])
        np.testing.assert_array_equal(out["grid1"], out["grid2"])
    else:
        for k in range(2):
            t1, t2 = np.asarray(one["time"][k]), np.asarray(two["time"][k])
            x1, x2 = np.asarray(one["X"][k]), np.asarray(two["X"][k])
            np.testing.assert_allclose(
                t1[_event(t1, x1)], t2[_event(t2, x2)], rtol=0.0, atol=1e-15
            )
        np.testing.assert_allclose(out["grid1"], out["grid2"], rtol=1e-12, atol=1e-15)
