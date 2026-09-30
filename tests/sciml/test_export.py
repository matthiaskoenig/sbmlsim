"""Tests of the export of a hybrid problem as PEtab SciML and its round trip.

The problems are the python defined problems of `tests.sciml.test_fit`. The
predictions of the problem which is read back are compared with a tolerance:
the first model roadrunner loads in a process differs by about `1e-9` from
every later one; the round trip of the cases of the test suite compares
against a second read and is exact, see `tests/sciml/test_testsuite.py`.
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import petab.v2 as petab_v2
import pytest
import yaml

from sbmlsim.fit import FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2 import to_petab
from sbmlsim.fit.petab_v2.export import PetabExporter
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.petab_v2.sciml_export import petab_index, petab_math
from sbmlsim.sciml import (
    Hybridization,
    NetworkInput,
    compile_network,
    compiled_path,
    network_fit_parameters,
)
from tests.fit.hooks import Scaling, factor_parameter
from tests.sciml.experiment import LotkaVolterra
from tests.sciml.hybrid import MODEL_PATH, feed_forward, two_inputs
from tests.sciml.test_fit import MECHANISTIC, PRE, RHS, SETTINGS, _before, _problem


def _read(
    yaml_file: Path, derived_dir: Path
) -> tuple[PetabReader, OptimizationProblem]:
    reader = PetabReader.from_yaml(yaml_file)
    reader.derived_dir = derived_dir
    problem = reader.to_optimization_problem(opid="restored")
    problem.initialize(SETTINGS)
    return reader, problem


def _tables(petab_dir: Path) -> dict[str, Any]:
    """Read the tables of an exported problem without `petab`, which needs torch."""
    tables = {
        name: pd.read_csv(
            petab_dir / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False
        )
        for name in ("parameters", "observables", "experiments", "mapping")
    }
    tables["config"] = yaml.safe_load((petab_dir / "problem.yaml").read_text())
    return tables


def _tuple(p: FitParameter) -> tuple:
    return (
        p.pid,
        p.start_value,
        p.lower_bound,
        p.upper_bound,
        p.unit,
        p.scale,
        p.target,
        p.is_versioned,
    )


def assert_round_trip(
    problem: OptimizationProblem, tmp_path: Path
) -> OptimizationProblem:
    """Write the problem, read it back and compare the two."""
    problem.initialize(SETTINGS)
    yaml_file = to_petab(problem, tmp_path / "petab")
    reader, restored = _read(yaml_file, tmp_path / "derived")

    assert [_tuple(p) for p in restored.parameters] == [
        _tuple(p) for p in problem.parameters
    ]
    assert restored.hybridizations == problem.hybridizations
    assert len(restored.mapping_keys) == len(problem.mapping_keys)

    keys = {
        (problem.experiment_keys[k], problem.mapping_keys[k]): k
        for k in range(len(problem.mapping_keys))
    }
    x = np.asarray(problem.x0, dtype=float)
    expected = problem.predictions(x)
    observed = restored.predictions(
        np.asarray(
            [dict(zip(problem.pids, x, strict=True))[pid] for pid in restored.pids]
        )
    )
    for i, key in enumerate(restored.mapping_keys):
        info = reader.observable_info(key)
        k = keys[(info["experiment"], info["mapping"])]
        assert np.array_equal(restored.x_references[i], problem.x_references[k])
        np.testing.assert_allclose(restored.y_references[i], problem.y_references[k])
        assert restored.mapping_kinds[i] == problem.mapping_kinds[k]
        assert restored.weights_curves[i] == problem.weights_curves[k]
        np.testing.assert_allclose(observed[i], expected[k], rtol=1e-7)
    assert log_likelihood(restored) == pytest.approx(log_likelihood(problem), rel=1e-8)
    return restored


# --- ROUND TRIPS ---


def test_a_network_before_the_simulation(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    restored = assert_round_trip(_problem([_before(network)], elements), tmp_path)
    (hybridization,) = restored.hybridizations
    assert hybridization.constants == {"k": 0.5}
    tables = _tables(tmp_path / "petab")
    rows = tables["parameters"].set_index("parameterId")
    assert rows.loc["net1__parameters", "estimate"] == "true"
    assert rows.loc["net1__parameters", "nominalValue"] == "array"
    assert rows.loc["k", "estimate"] == "false"
    assert float(rows.loc["k", "nominalValue"]) == 0.5
    sciml = tables["config"]["extensions"]["sciml"]
    assert sciml["neural_networks"]["net1"]["pre_initialization"] is True
    # the elements are not in the `sbmlsim` block
    assert set(tables["config"]["extensions"]["sbmlsim"]["parameters"]) == {
        "alpha",
        "beta",
    }


def test_a_frozen_layer_and_bounds(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network,
        estimate={"net1": True, "net1.layer1": False},
        bounds={"net1": (-5.0, 5.0)},
        external=True,
    )
    frozen = set(network.parameter_ids()) - {p.pid for p in elements}
    assert_round_trip(_problem([_before(network, frozen=frozen)], elements), tmp_path)
    rows = _tables(tmp_path / "petab")["parameters"].set_index("parameterId")
    # the most common row is the row of the network, the layer which differs
    # has a row of its own
    network_row = rows.loc["net1__parameters"]
    layer_row = rows.loc["net1__layer2__parameters"]
    assert {network_row["estimate"], layer_row["estimate"]} == {"true", "false"}
    estimated = layer_row if layer_row["estimate"] == "true" else network_row
    assert (float(estimated["lowerBound"]), float(estimated["upperBound"])) == (
        -5.0,
        5.0,
    )


def test_the_arrays_of_the_simulations(tmp_path: Path) -> None:
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=PRE,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="alpha"),
            "net6__input1": NetworkInput(
                arrays={"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0]}
            ),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    elements = network_fit_parameters(
        network, estimate={"net6": True}, bounds={}, external=True
    )
    assert_round_trip(_problem([hybridization], elements), tmp_path)
    # the arrays are keyed by the condition of the first period of the experiment
    experiments = _tables(tmp_path / "petab")["experiments"]
    assert sorted(experiments["conditionId"]) == ["e1__tc0", "e2__tc0"]


def _compiled(tmp_path: Path, hybridization: Hybridization) -> type[LotkaVolterra]:
    compiled = compile_network(
        MODEL_PATH, [hybridization], compiled_path(MODEL_PATH, tmp_path / "model")
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    return Compiled


def test_a_network_in_the_right_hand_side(tmp_path: Path) -> None:
    network = feed_forward()
    hybridization = Hybridization(
        network=network,
        pattern=RHS,
        model="lv",
        inputs={
            "net1__input0__0": NetworkInput(formula="prey"),
            "net1__input0__1": NetworkInput(formula="predator"),
        },
        outputs={"net1__output0__0": "gamma"},
    )
    elements = network_fit_parameters(network, estimate={"net1": True}, bounds={})
    problem = _problem(
        [hybridization], elements, experiment=_compiled(tmp_path, hybridization)
    )
    assert_round_trip(problem, tmp_path)
    config = _tables(tmp_path / "petab")["config"]
    # the model of the problem is the model without the network
    assert config["model_files"]["lv"]["location"] == "lotka_volterra.xml"
    assert (
        "net1__output0__0"
        not in (tmp_path / "petab" / "lotka_volterra.xml").read_text()
    )
    assert (tmp_path / "petab" / "net1.yaml").is_file()
    assert (tmp_path / "petab" / "net1_arrays.hdf5").is_file()
    assert (tmp_path / "petab" / "hybridization.tsv").is_file()


def test_the_arrays_of_a_compiled_network(tmp_path: Path) -> None:
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=RHS,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="prey"),
            "net6__input1": NetworkInput(
                arrays={"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0]}
            ),
        },
        outputs={"net6__output0__0": "gamma"},
        frozen=set(network.parameter_ids()),
    )
    problem = _problem(
        [hybridization], [], experiment=_compiled(tmp_path, hybridization)
    )
    assert_round_trip(problem, tmp_path)


def test_an_observable_which_shadows_an_entity(tmp_path: Path) -> None:
    """The mapping keys `prey_e1` and `prey_e2` become one observable, not `prey`."""
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    assert_round_trip(_problem([_before(network)], elements), tmp_path)
    observables = _tables(tmp_path / "petab")["observables"]
    assert set(observables["observableId"]) == {
        "observable__prey",
        "observable__predator",
    }


def test_the_exported_problem_is_valid_petab(tmp_path: Path) -> None:
    """`petab` reads the networks of a problem through torch, the dev extra has it."""
    pytest.importorskip("torch")
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    yaml_file = to_petab(problem, tmp_path / "petab")
    issues = petab_v2.Problem.from_yaml(yaml_file).validate()
    assert not issues.has_errors(), str(issues)


# --- WHAT IS REFUSED ---


def test_a_partial_array_is_refused(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    # one element of the bias of the first layer is frozen
    elements = [p for p in elements if p.pid != "net1__layer1__bias__0"]
    problem = _problem([_before(network, frozen={"net1__layer1__bias__0"})], elements)
    problem.initialize(SETTINGS)
    with pytest.raises(ValueError, match="sciml-partial-array"):
        to_petab(problem, tmp_path / "petab")


def test_an_element_which_differs_from_the_network_is_refused(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    elements[0] = FitParameter(
        elements[0].pid, 99.0, unit="dimensionless", target=elements[0].target
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    with pytest.raises(ValueError, match=r"starts from 99\.0, but the network 'net1'"):
        to_petab(problem, tmp_path / "petab")


def test_an_element_on_another_scale_is_refused(tmp_path: Path) -> None:
    network = feed_forward()
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    # a positive element, which a logarithmic scale can search
    k = next(i for i, p in enumerate(elements) if float(p.start_value or 0.0) > 0.0)
    first = elements[k]
    elements[k] = FitParameter(
        first.pid,
        first.start_value,
        1e-6,
        10.0,
        unit="dimensionless",
        target=first.target,
        scale=ParameterScaleType.LOG10,
    )
    problem = _problem([_before(network)], elements)
    problem.initialize(SETTINGS)
    with pytest.raises(ValueError, match="has the scale 'LOG10'"):
        to_petab(problem, tmp_path / "petab")


def test_a_hook_which_is_no_network_is_refused() -> None:
    problem = OptimizationProblem(
        opid="hook",
        mapping_collections=_problem([], []).mapping_collections,
        fit_parameters=[*MECHANISTIC, factor_parameter()],
        base_path=MODEL_PATH.parent,
        data_path=MODEL_PATH.parent,
        hybridizations=[Scaling(model="lv", target="gamma")],
    )
    problem.initialize(SETTINGS)
    with pytest.raises(ValueError, match="is not a `Hybridization`"):
        PetabExporter(problem)


# --- THE HELPERS ---


def test_petab_math() -> None:
    assert petab_math("alpha + (prey - 1.3)") == "alpha + prey - 1.3"
    assert petab_math("ln(x)") == "log(x)"
    assert petab_math("10.0") == "10"
    assert petab_math("x / 3") == "x/3"


def test_petab_index() -> None:
    assert petab_index((0, 1), (2, 3)) == "[0][1]"
    assert petab_index((0, 0), (1, 1)) == "[0]"
    assert petab_index((0, 2), (1, 5)) == "[2]"
    assert petab_index((0,), (1,)) == "[0]"
