"""The changes of a model are pre-initialization changes of its simulations."""

from pathlib import Path

import numpy as np

from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.simulator.models import sbml


def _experiment(path: Path) -> type[SimulationExperiment]:
    """Get an experiment of the probe model with changes of the model."""

    class Exp(SimulationExperiment):
        def models(self) -> dict[str, AbstractModel | Path]:
            return {"m": AbstractModel(source=path, changes={"b0": 0.0, "a0": 2.0})}

        def simulations(self) -> dict[str, Simulation | Scan]:
            return {
                "s": Simulation(end=1, steps=2, preinit_changes={"a0": 3.0}),
                "dosing": Simulation(
                    end=2, times=[0, 1, 2], changes=[Change([0, 1], {"X": "X + 1"})]
                ),
                "later": Simulation(
                    end=1, times=[0, 1], changes=[Change(0.5, {"b0": 3.0})]
                ),
                "scan": Scan(
                    Simulation(end=1, steps=2),
                    [Dimension("d", values={"b0": np.array([1.0, 4.0])})],
                ),
            }

        def tasks(self) -> dict[str, Task]:
            return {
                "t": Task(model="m", simulation="s"),
                "t_dosing": Task(model="m", simulation="dosing"),
                "t_scan": Task(model="m", simulation="scan"),
                "t_later": Task(model="m", simulation="later"),
            }

    return Exp


def test_model_changes_merge_into_preinit(tmp_path: Path) -> None:
    """A change of the model applies unless the simulation or the scan sets it."""
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    exp_class = _experiment(path)
    runner = ExperimentRunner(
        [exp_class], base_path=tmp_path, data_path=tmp_path, on_error="raise"
    )
    exp = runner.experiments["Exp"]
    exp.run(Simulator(), reduced_selections=False)

    res = exp.results["t"]
    assert res["[B]"].values[0] == 0.0
    assert res["[A]"].values[0] == 3.0

    res = exp.results["t_dosing"]
    np.testing.assert_allclose(res["X"].values, [13.0, 14.0, 14.0])

    # the change of the model is a pre-initialization change, the change of
    # the simulation at a later time does not replace it
    res = exp.results["t_later"]
    assert res["b0"].values[0] == 0.0
    assert res["[B]"].values[0] == 0.0

    # the scan has the time last
    res = exp.results["t_scan"]
    np.testing.assert_allclose(res["[B]"].values[:, 0], [1.0, 4.0])
    np.testing.assert_allclose(res["[A]"].values[:, 0], [2.0, 2.0])
