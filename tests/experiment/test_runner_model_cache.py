"""Experiment model caching respects resolved source identities."""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.experiment.runner import model_key
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.model.test_model_roadrunner_init import EXTERNAL_SBML, HIERARCHICAL_SBML
from tests.simulator.models import PROBE, sbml


def test_relative_model_names_in_different_directories_are_distinct(
    tmp_path: Path,
) -> None:
    class First(SimulationExperiment):
        model_directory: ClassVar[Path]

        def models(self) -> dict:
            return {"m": AbstractModel("model.xml", base_path=self.model_directory)}

        def simulations(self) -> dict:
            return {"sim": Simulation(end=1, steps=1)}

        def tasks(self) -> dict:
            return {"task": Task(model="m", simulation="sim")}

        def data(self) -> dict:
            return {"rate": Data("k1", task="task")}

    class Second(First):
        pass

    for cls, rate in [(First, 0.8), (Second, 0.1)]:
        cls.model_directory = tmp_path / cls.__name__
        cls.model_directory.mkdir()
        (cls.model_directory / "model.xml").write_text(
            sbml(PROBE.replace("k1 = 0.8", f"k1 = {rate}"))
        )
    runner = ExperimentRunner(
        [First, Second],
        base_path=tmp_path,
        data_path=tmp_path,
        simulator=Simulator(n_workers=1),
        on_error="raise",
    )
    assert len(runner.models) == 2
    for name, rate in [("First", 0.8), ("Second", 0.1)]:
        experiment = runner.experiments[name]
        experiment.run(runner.simulator)
        np.testing.assert_array_equal(experiment.results["task"]["k1"], [rate, rate])


def test_absolute_and_relative_names_share_the_same_model_key(tmp_path: Path) -> None:
    path = tmp_path / "model.xml"
    path.write_text(sbml())
    absolute = AbstractModel(path)
    relative = AbstractModel("model.xml", base_path=tmp_path)
    assert relative.source.path == path.resolve()
    assert model_key(absolute) == model_key(relative)


def test_inline_external_models_use_their_own_base_directories(tmp_path: Path) -> None:
    """Identical inline XML can resolve to distinct external models."""

    class First(SimulationExperiment):
        model_directory: ClassVar[Path]

        def models(self) -> dict:
            return {"m": AbstractModel(EXTERNAL_SBML, base_path=self.model_directory)}

        def simulations(self) -> dict:
            return {"sim": Simulation(end=1, steps=1)}

        def tasks(self) -> dict:
            return {"task": Task(model="m", simulation="sim")}

        def data(self) -> dict:
            return {"amount": Data("sub__A1", task="task")}

    class Second(First):
        pass

    inner = HIERARCHICAL_SBML.split("<comp:listOfModelDefinitions>")[1]
    inner = inner.split("</comp:listOfModelDefinitions>")[0]
    inner = inner.replace("comp:modelDefinition", "model")
    for cls, rate in [(First, 3), (Second, 5)]:
        cls.model_directory = tmp_path / f"{cls.__name__} with spaces"
        cls.model_directory.mkdir()
        (cls.model_directory / "inner.xml").write_text(
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" '
            'level="3" version="1">'
            + inner.replace('value="3"', f'value="{rate}"')
            + "</sbml>",
            encoding="utf-8",
        )
    runner = ExperimentRunner(
        [First, Second],
        base_path=tmp_path,
        data_path=tmp_path,
        simulator=Simulator(n_workers=1),
        on_error="raise",
    )
    assert len(runner.models) == 2
    for name, amount in [("First", 6), ("Second", 10)]:
        experiment = runner.experiments[name]
        experiment.run(runner.simulator)
        np.testing.assert_array_equal(
            experiment.results["task"]["sub__A1"], [amount, amount]
        )


def test_inline_base_directories_are_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    implicit = AbstractModel(EXTERNAL_SBML)
    relative = AbstractModel(EXTERNAL_SBML, base_path=Path("."))
    absolute = AbstractModel(EXTERNAL_SBML, base_path=tmp_path)
    assert model_key(implicit) == model_key(relative) == model_key(absolute)
    other = AbstractModel(EXTERNAL_SBML, base_path=tmp_path / "elsewhere")
    assert model_key(other) != model_key(absolute)


def test_the_base_directory_of_inline_sbml_without_external_models_is_no_key(
    tmp_path: Path,
) -> None:
    content = sbml()
    here = AbstractModel(content, base_path=tmp_path)
    other = AbstractModel(content, base_path=tmp_path / "elsewhere")
    assert model_key(here) == model_key(other)


def test_the_selections_of_a_model_are_part_of_its_key() -> None:
    content = sbml()
    default = AbstractModel(content)
    selected = AbstractModel(content, selections=["time", "A"])
    other = AbstractModel(content, selections=["time", "B"])
    assert model_key(selected) == model_key(
        AbstractModel(content, selections=["time", "A"])
    )
    assert len({model_key(default), model_key(selected), model_key(other)}) == 3
