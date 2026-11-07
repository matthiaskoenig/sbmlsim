"""The optimization level the code of a model is compiled with (#331)."""

import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import libsbml
import numpy as np
import pytest
import roadrunner

from examples.hctz_fitting import MODEL_PATH
from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.model import AbstractModel, JitOptimization, RoadrunnerSBMLModel
from sbmlsim.model.model_roadrunner import (
    DEFAULT_JIT_OPTIMIZATION,
    JIT_OPTIMIZATION_VARIABLE,
    _with_level,
    resolve_jit_optimization,
)
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.worker import ModelSpec
from sbmlsim.task import Task
from tests.simulator.models import sbml


def test_the_level_without_the_variable_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(JIT_OPTIMIZATION_VARIABLE, raising=False)
    assert resolve_jit_optimization() is DEFAULT_JIT_OPTIMIZATION
    assert RoadrunnerSBMLModel(source=sbml()).jit_optimization is (
        DEFAULT_JIT_OPTIMIZATION
    )


def test_the_variable_sets_the_level_of_a_model_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(JIT_OPTIMIZATION_VARIABLE, "O0")
    assert RoadrunnerSBMLModel(source=sbml()).jit_optimization is JitOptimization.O0
    assert (
        RoadrunnerSBMLModel(source=sbml(), jit_optimization="O3").jit_optimization
        is JitOptimization.O3
    )


def test_an_unknown_level_names_where_it_was_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"jit_optimization 'fast'.*'O0', 'O1'"):
        RoadrunnerSBMLModel(source=sbml(), jit_optimization="fast")
    monkeypatch.setenv(JIT_OPTIMIZATION_VARIABLE, "o0")
    with pytest.raises(ValueError, match=f"{JIT_OPTIMIZATION_VARIABLE} 'o0'"):
        RoadrunnerSBMLModel(source=sbml())


def test_a_model_is_compiled_with_lljit_at_its_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    levels: list[int] = []
    set_level = roadrunner.LoadSBMLOptions.setLLJitOptimizationLevel

    def spy(options: roadrunner.LoadSBMLOptions, level: int) -> None:
        levels.append(level)
        set_level(options, level)

    monkeypatch.setattr(roadrunner.LoadSBMLOptions, "setLLJitOptimizationLevel", spy)
    for level in JitOptimization:
        RoadrunnerSBMLModel(source=sbml(), jit_optimization=level)
    assert levels == [
        roadrunner.LoadSBMLOptions.NONE,
        roadrunner.LoadSBMLOptions.LESS,
        roadrunner.LoadSBMLOptions.DEFAULT,
        roadrunner.LoadSBMLOptions.AGGRESSIVE,
    ]


def test_the_levels_simulate_the_same() -> None:
    simulator = Simulator(n_workers=1)
    simulation = Simulation(end=10, steps=20)
    results = [
        simulator.simulate(
            RoadrunnerSBMLModel(source=sbml(), jit_optimization=level), simulation
        )
        for level in JitOptimization
    ]
    for result in results[1:]:
        np.testing.assert_allclose(result.values, results[0].values, rtol=1e-12)


def test_a_worker_compiles_a_model_at_the_level_of_the_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RoadrunnerSBMLModel(source=sbml(), jit_optimization="O0")
    spec = ModelSpec.of(model)
    assert spec.jit_optimization is JitOptimization.O0
    other = RoadrunnerSBMLModel(source=sbml(), jit_optimization="O2")
    assert ModelSpec.of(other).key != spec.key
    # the environment of a worker is not the one of the parent
    monkeypatch.setenv(JIT_OPTIMIZATION_VARIABLE, "O2")
    loaded = pickle.loads(pickle.dumps(spec)).load()
    assert loaded.jit_optimization is JitOptimization.O0
    assert ModelSpec.of(loaded) == spec


class _Experiment(SimulationExperiment):
    model_path: Path

    def models(self) -> dict:
        return {"m": AbstractModel(self.model_path)}

    def simulations(self) -> dict:
        return {"sim": Simulation(end=1, steps=1)}

    def tasks(self) -> dict:
        return {"task": Task(model="m", simulation="sim")}

    def data(self) -> dict:
        return {"A": Data("[A]", task="task")}


def test_a_runner_compiles_its_models_at_its_level(tmp_path: Path) -> None:
    _Experiment.model_path = tmp_path / "model.xml"
    _Experiment.model_path.write_text(sbml(), encoding="utf-8")
    runner = ExperimentRunner(
        _Experiment,
        base_path=tmp_path,
        data_path=tmp_path,
        on_error="raise",
        jit_optimization="O1",
    )
    assert runner.jit_optimization is JitOptimization.O1
    models = list(runner.models.values())
    assert all(isinstance(model, RoadrunnerSBMLModel) for model in models)
    assert [
        model.jit_optimization
        for model in models
        if isinstance(model, RoadrunnerSBMLModel)
    ] == [JitOptimization.O1]


#: compiles a model at the levels of the arguments in one process, its dose a
#: parameter ending in `PODOSE_hctz`, and prints the bytes of the timecourse of
#: the last one
_LEVELS_SCRIPT = """
import sys
import numpy as np
from sbmlsim.model.model_roadrunner import load_roadrunner
path, levels = sys.argv[1], sys.argv[2:]
for level in levels:
    r = load_roadrunner(path, level)
    (dose,) = [p for p in r.model.getGlobalParameterIds() if p.endswith("PODOSE_hctz")]
    r.setValue(dose, 25)
    values = np.asarray(r.simulate(0, 1440, 20))
    del r
sys.stdout.write(values.tobytes().hex())
"""

#: a hierarchical model of the HCTZ model of another file
_TOP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core"
      xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1"
      level="3" version="2" comp:required="true">
  <model id="top">
    <comp:listOfSubmodels>
      <comp:submodel comp:id="sub" comp:modelRef="inner"/>
    </comp:listOfSubmodels>
  </model>
  <comp:listOfExternalModelDefinitions>
    <comp:externalModelDefinition comp:id="inner" comp:source="inner.xml"
        comp:modelRef="{model}"/>
  </comp:listOfExternalModelDefinitions>
</sbml>
"""


def _hierarchical(directory: Path) -> Path:
    """Write a hierarchical model of the HCTZ model into a directory."""
    model = libsbml.readSBMLFromFile(str(MODEL_PATH)).getModel().getId()
    shutil.copy(MODEL_PATH, directory / "inner.xml")
    top = directory / "top.xml"
    top.write_text(_TOP.format(model=model), encoding="utf-8")
    return top


@pytest.mark.parametrize("hierarchical", [False, True], ids=["flat", "hierarchical"])
def test_a_level_compiled_before_in_the_process_does_not_change_another(
    tmp_path: Path, hierarchical: bool
) -> None:
    """LLJIT reuses the code of an SBML it compiled before, also at another level.

    The SBML carries its level, so an instance at `O2` after one at `O0` is an
    `O2` one, bitwise; HCTZ integrates differently at the two levels. A
    hierarchical model is flattened before, so it carries its level as well.
    """
    path = _hierarchical(tmp_path) if hierarchical else MODEL_PATH

    def timecourse(*levels: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", _LEVELS_SCRIPT, str(path), *levels],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parents[2],
            timeout=300,
        )
        return result.stdout

    o2 = timecourse("O2")
    assert timecourse("O0", "O2") == o2
    assert timecourse("O0") != o2


def test_the_level_follows_the_root_and_the_declaration_stays_first(
    tmp_path: Path,
) -> None:
    text = sbml()
    tagged = _with_level(text, JitOptimization.O0)
    assert tagged == text.rstrip() + "\n<!-- sbmlsim LLJIT O0 -->\n"
    assert tagged.startswith(text[:5])
    # a string with a byte order mark and whitespace is SBML, not a path
    assert _with_level("\ufeff \n" + text, JitOptimization.O2).endswith(
        "<!-- sbmlsim LLJIT O2 -->\n"
    )
    # a path, given as a path or a string, is read; a leading comment which
    # names the root changes nothing
    path = tmp_path / "model.xml"
    commented = text.replace("<sbml", "<!-- <sbml> --><sbml", 1)
    path.write_bytes(b"\xef\xbb\xbf" + commented.encode("utf-8"))
    for source in (path, str(path)):
        tagged = _with_level(source, JitOptimization.O1)
        assert tagged == commented.rstrip() + "\n<!-- sbmlsim LLJIT O1 -->\n"


def test_a_file_is_read_in_the_encoding_of_its_declaration(tmp_path: Path) -> None:
    text = sbml().replace('encoding="UTF-8"', 'encoding="ISO-8859-1"', 1)
    assert 'encoding="ISO-8859-1"' in text
    text = text.replace("<model ", '<model name="Ödem" ', 1)
    path = tmp_path / "latin.xml"
    path.write_bytes(text.encode("latin-1"))
    assert 'name="Ödem"' in _with_level(path, JitOptimization.O2)


def test_a_hierarchical_model_is_flattened(tmp_path: Path) -> None:
    tagged = _with_level(_hierarchical(tmp_path), JitOptimization.O2)
    assert "externalModelDefinition" not in tagged
    assert "sub__PODOSE_hctz" in tagged
    assert tagged.endswith("<!-- sbmlsim LLJIT O2 -->\n")
