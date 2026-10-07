"""Tests of the export of a hybrid problem as PEtab SciML and its round trip.

The problems are the python defined problems of `tests.sciml.test_fit`. The
predictions of the problem which is read back are compared with a tolerance:
the first model roadrunner loads in a process differs by about `1e-9` from
every later one; the round trip of the cases of the test suite compares
against a second read and is exact, see `tests/sciml/test_testsuite.py`.
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import petab.v2 as petab_v2
import pytest
import sympy
import yaml

from sbmlsim import Q
from sbmlsim.fit import FitMappingCollection, FitParameter
from sbmlsim.fit.objects import EXTERNAL_PREFIX
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.petab_v2 import to_petab
from sbmlsim.fit.petab_v2.export import PetabExporter
from sbmlsim.fit.petab_v2.likelihood import log_likelihood, nominal_parameters
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.petab_v2.sciml_export import parameters_id, petab_index, petab_math
from sbmlsim.mathml import formula_expression
from sbmlsim.sciml import (
    Hybridization,
    NetworkInput,
    compile_network,
    compiled_path,
    network_fit_parameters,
)
from sbmlsim.sciml.hybridization import ALL_CONDITIONS
from sbmlsim.sciml.testsuite import SciMLSuite
from sbmlsim.simulation import Simulation
from sbmlsim.units import Quantity
from tests.fit.hooks import Scaling, factor_parameter
from tests.sciml.experiment import SIMULATIONS, LotkaVolterra
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
    """Read the tables of an exported problem with pandas.

    `petab.v2.Problem.from_yaml` reads the networks of the problem through
    torch, which the tests of the tables do not need.
    """
    tables = {
        name: pd.read_csv(
            petab_dir / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False
        )
        for name in ("parameters", "observables", "experiments", "mapping")
    }
    tables["config"] = yaml.safe_load((petab_dir / "problem.yaml").read_text())
    return tables


def _validate(yaml_file: Path) -> None:
    """Validate an exported problem with `petab`, which reads the networks with torch."""
    pytest.importorskip("torch")
    issues = petab_v2.Problem.from_yaml(yaml_file).validate()
    assert not issues.has_errors(), str(issues)


def _condition_targets(petab_dir: Path) -> set[str]:
    path = petab_dir / "conditions.tsv"
    if not path.is_file():
        return set()
    return set(pd.read_csv(path, sep="\t", dtype=str)["targetId"])


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


def _same_formulas(a: NetworkInput, b: NetworkInput) -> bool:
    """Compare the formulas of two inputs as math, e.g. `kin * 2` and `2 * kin`."""

    def formulas(network_input: NetworkInput) -> dict[str, str]:
        # a formula for every condition is the formula of the input, which
        # is how the reader gives it back
        if network_input.formula is not None:
            return {ALL_CONDITIONS: network_input.formula}
        return dict(network_input.formulas or {})

    fa, fb = formulas(a), formulas(b)
    return fa.keys() == fb.keys() and all(
        sympy.simplify(formula_expression(fa[key]))
        == sympy.simplify(formula_expression(fb[key]))
        for key in fa
    )


def assert_same_hybridizations(restored: list[Any], expected: list[Any]) -> None:
    """Compare hybridizations, the formulas of their inputs as math."""
    assert len(restored) == len(expected)
    for a, b in zip(restored, expected, strict=True):
        assert isinstance(a, Hybridization)
        assert isinstance(b, Hybridization)
        for f in dataclasses.fields(Hybridization):
            if f.compare and f.name != "inputs":
                assert getattr(a, f.name) == getattr(b, f.name), f.name
        assert a.inputs.keys() == b.inputs.keys()
        for key, network_input in a.inputs.items():
            other = b.inputs[key]
            if network_input.arrays is not None or other.arrays is not None:
                assert network_input == other, key
            else:
                assert _same_formulas(network_input, other), (network_input, other)


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
    assert_same_hybridizations(restored.hybridizations, problem.hybridizations)
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

    # no condition sets an element or a parameter which is no entity of the model
    targets = _condition_targets(tmp_path / "petab")
    assert not any(target.startswith(EXTERNAL_PREFIX) for target in targets)
    assert not targets & {p.pid for p in problem.parameters if p.is_external}
    _validate(yaml_file)
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
    # the mapping keys `prey_e1` and `prey_e2` become one observable, which
    # would shadow the species `prey`
    assert set(tables["observables"]["observableId"]) == {
        "observable__prey",
        "observable__predator",
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


def test_a_simulation_of_two_experiments(tmp_path: Path) -> None:
    """The arrays of a simulation are keyed by the condition of each experiment."""
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
    # the simulation `e1` is in both collections, i.e. in two experiments
    problem = OptimizationProblem(
        opid="hybrid",
        mapping_collections=[
            FitMappingCollection(
                experiment=LotkaVolterra, sid="first", mappings=["prey_e1"]
            ),
            FitMappingCollection(
                experiment=LotkaVolterra,
                sid="second",
                mappings=["predator_e1", "prey_e2", "predator_e2"],
            ),
        ],
        fit_parameters=[*MECHANISTIC, *elements],
        base_path=MODEL_PATH.parent,
        data_path=MODEL_PATH.parent,
        hybridizations=[hybridization],
    )
    problem.initialize(SETTINGS)
    yaml_file = to_petab(problem, tmp_path / "petab")
    experiments = _tables(tmp_path / "petab")["experiments"]
    assert sorted(experiments["conditionId"]) == [
        "first__tc0",
        "second__sim0__tc0",
        "second__sim1__tc0",
    ]
    _, restored = _read(yaml_file, tmp_path / "derived")
    (restored_hybridization,) = restored.hybridizations
    assert isinstance(restored_hybridization, Hybridization)
    arrays = restored_hybridization.inputs["net6__input1"].arrays
    assert arrays is not None
    assert {key: np.asarray(value).tolist() for key, value in arrays.items()} == {
        "first": [1.0, 2.0, 3.0],
        "second__sim0": [1.0, 2.0, 3.0],
        "second__sim1": [3.0, 2.0, 1.0],
    }
    assert log_likelihood(restored) == pytest.approx(log_likelihood(problem), rel=1e-8)


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


def test_a_fitted_parameter_set_is_written(tmp_path: Path) -> None:
    """The exported problem starts from the set, the problem is not changed."""
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
    problem.initialize(SETTINGS)
    starts = [p.start_value for p in problem.parameters]
    start = log_likelihood(problem)

    # a set which differs from the start in a parameter of the model and in
    # the elements of the network
    values = {
        pid: value * 1.05 + 0.01
        for pid, value in nominal_parameters(problem).values.items()
    }
    fitted = ParameterSet(sid="fitted", values=values)
    expected = log_likelihood(problem, fitted)
    assert expected != pytest.approx(start)

    yaml_file = to_petab(problem, tmp_path / "petab", parameter_set=fitted)
    assert [p.start_value for p in problem.parameters] == starts

    # the tables and the arrays carry the set
    parameters = _tables(tmp_path / "petab")["parameters"].set_index("parameterId")
    assert float(parameters.loc["alpha", "nominalValue"]) == pytest.approx(
        values["alpha"]
    )
    assert parameters.loc[parameters_id("net1"), "nominalValue"] == "array"
    written = network.read_arrays(tmp_path / "petab" / "net1_arrays.hdf5")
    for element, (layer, name, index) in network.parameter_ids().items():
        assert written[layer][name][index] == pytest.approx(values[element])

    # the problem which is read starts from the set
    _, restored = _read(yaml_file, tmp_path / "derived")
    assert {p.pid: p.start_value for p in restored.parameters} == pytest.approx(values)
    assert log_likelihood(restored) == pytest.approx(expected, rel=1e-8)


def test_a_parameter_set_without_a_parameter_is_refused(tmp_path: Path) -> None:
    """A set which lacks a parameter of the problem has nothing to write for it."""
    problem = _problem([], [])
    problem.initialize(SETTINGS)
    with pytest.raises(KeyError, match="'beta'"):
        to_petab(
            problem,
            tmp_path / "petab",
            parameter_set=ParameterSet(sid="partial", values={"alpha": 1.0}),
        )


def test_a_parameter_set_with_a_frozen_element_is_refused(tmp_path: Path) -> None:
    """A set of another fit must not move the elements the problem freezes."""
    network = feed_forward()
    elements, hybridization = _before(network).fit_parameters(
        estimate={"net1": False, "net1.layer1": True}
    )
    problem = _problem([hybridization], elements)
    problem.initialize(SETTINGS)
    values = dict(nominal_parameters(problem).values)
    # the set of a fit of the whole network
    values["net1__layer2__bias__0"] = 0.5
    with pytest.raises(ValueError, match=r"'net1__layer2__bias__0'.*'net1'"):
        to_petab(
            problem,
            tmp_path / "petab",
            parameter_set=ParameterSet(sid="other", values=values),
        )


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


# --- THE INPUTS ---


def _inputs(
    network_sid: str = "net1", **inputs: NetworkInput
) -> dict[str, NetworkInput]:
    return {f"{network_sid}__{key}": value for key, value in inputs.items()}


def _external(pid: str) -> FitParameter:
    """Get an estimated parameter which is no entity of the model."""
    return FitParameter(
        pid,
        1.0,
        0.0,
        10.0,
        unit="dimensionless",
        target=f"{EXTERNAL_PREFIX}{pid}",
        scale=ParameterScaleType.LINEAR,
    )


def _elements(network: Any) -> list[FitParameter]:
    return network_fit_parameters(
        network, estimate={network.sid: True}, bounds={}, external=True
    )


def test_an_estimated_parameter_which_is_an_input(tmp_path: Path) -> None:
    """The parameter is the `petabEntityId` of the input, as in case 006."""
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(formula="alpha"),
            input0__1=NetworkInput(formula="kin"),
        ),
        constants={},
    )
    assert_round_trip(
        _problem([hybridization], [_external("kin"), *_elements(network)]), tmp_path
    )
    mapping = _tables(tmp_path / "petab")["mapping"].set_index("modelEntityId")
    assert mapping.loc["net1.inputs[0][1]", "petabEntityId"] == "kin"


def test_a_constant_in_a_formula(tmp_path: Path) -> None:
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(formula="alpha"),
            input0__1=NetworkInput(formula="2 * k"),
        ),
    )
    assert_round_trip(_problem([hybridization], _elements(network)), tmp_path)


def test_a_constant_of_two_inputs(tmp_path: Path) -> None:
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(formula="k"),
            input0__1=NetworkInput(formula="k"),
        ),
    )
    assert_round_trip(_problem([hybridization], _elements(network)), tmp_path)


def test_an_input_per_condition_which_is_an_estimated_parameter(
    tmp_path: Path,
) -> None:
    """A change `input = beta` is the input, not a version of `beta`."""
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(formulas={"e1": "beta", "e2": "alpha"}),
            input0__1=NetworkInput(formula="k"),
        ),
    )
    assert_round_trip(_problem([hybridization], _elements(network)), tmp_path)


def test_an_input_with_a_formula_for_the_other_conditions(tmp_path: Path) -> None:
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(
                formulas={ALL_CONDITIONS: "alpha + 1", "e1": "2 * beta"}
            ),
            input0__1=NetworkInput(formulas={ALL_CONDITIONS: "k", "e2": "2 * k"}),
        ),
    )
    assert_round_trip(_problem([hybridization], _elements(network)), tmp_path)


def test_an_input_with_one_formula_for_all_conditions(tmp_path: Path) -> None:
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(formulas={ALL_CONDITIONS: "alpha + 1"}),
            input0__1=NetworkInput(formula="k"),
        ),
    )
    assert_round_trip(_problem([hybridization], _elements(network)), tmp_path)


def test_a_formula_which_is_not_in_the_canonical_form(tmp_path: Path) -> None:
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(formula="kin * 2"),
            input0__1=NetworkInput(formula="k"),
        ),
    )
    assert_round_trip(
        _problem([hybridization], [_external("kin"), *_elements(network)]), tmp_path
    )


def test_a_constant_and_a_species_in_the_right_hand_side(tmp_path: Path) -> None:
    """The formula is evaluated along the trajectory, a row of the hybridization table."""
    network = feed_forward()
    hybridization = Hybridization(
        network=network,
        pattern=RHS,
        model="lv",
        inputs=_inputs(
            input0__0=NetworkInput(formula="k * prey"),
            input0__1=NetworkInput(formula="predator"),
        ),
        outputs={"net1__output0__0": "gamma"},
        constants={"k": 0.5},
    )
    problem = _problem(
        [hybridization],
        network_fit_parameters(network, estimate={"net1": True}, bounds={}),
        experiment=_compiled(tmp_path, hybridization),
    )
    assert_round_trip(problem, tmp_path)
    assert not _condition_targets(tmp_path / "petab")
    rows = pd.read_csv(tmp_path / "petab" / "hybridization.tsv", sep="\t", dtype=str)
    assert "net1__input0__0" in set(rows["targetId"])


def test_a_parameter_of_the_model_before_the_simulation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`delta` is no parameter of the fit, the table fixes it to its value."""
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(formula="delta"),
            input0__1=NetworkInput(formula="delta * k"),
        ),
    )
    with caplog.at_level(logging.WARNING):
        assert_round_trip(_problem([hybridization], _elements(network)), tmp_path)
    # the fixed row is a change of the model in the unit of the model
    assert not [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    rows = _tables(tmp_path / "petab")["parameters"].set_index("parameterId")
    assert rows.loc["delta", "estimate"] == "false"
    assert float(rows.loc["delta", "nominalValue"]) == 1.8

    # a model whose units cannot be read gets the dimensionless quantity
    reader = PetabReader.from_yaml(tmp_path / "petab" / "problem.yaml")
    reader.uinfo = None
    change = reader._nominal_changes()["delta"]
    assert isinstance(change, Quantity)
    assert (change.magnitude, str(change.units)) == (1.8, "dimensionless")


def test_the_arrays_of_a_simulation_the_problem_does_not_have(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A filter of the data can leave out a simulation which an input names."""
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=PRE,
        model="lv",
        inputs=_inputs(
            "net6",
            input0__0=NetworkInput(formula="alpha"),
            input1=NetworkInput(
                arrays={"e1": [1.0, 2.0, 3.0], "e2": [3.0, 2.0, 1.0], "e9": [0, 0, 0]}
            ),
        ),
        outputs={"net6__output0__0": "gamma"},
    )
    problem = _problem([hybridization], _elements(network))
    problem.initialize(SETTINGS)
    with caplog.at_level(logging.INFO, logger="sbmlsim.fit.petab_v2.sciml_export"):
        yaml_file = to_petab(problem, tmp_path / "petab")
    assert "'net6__input1' of the network 'net6'" in caplog.text
    assert "['e9']" in caplog.text
    _validate(yaml_file)
    _, restored = _read(yaml_file, tmp_path / "derived")
    (restored_hybridization,) = restored.hybridizations
    assert isinstance(restored_hybridization, Hybridization)
    arrays = restored_hybridization.inputs["net6__input1"].arrays
    assert arrays is not None
    assert set(arrays) == {"e1", "e2"}
    assert log_likelihood(restored) == pytest.approx(log_likelihood(problem), rel=1e-8)


@pytest.mark.sciml_testsuite
def test_the_case_006_is_valid_petab(tmp_path: Path) -> None:
    """The estimated input of case 006 is a parameter of the parameter table."""
    suite = SciMLSuite.cached()
    if suite is None:
        pytest.skip("the PEtab SciML test suite is not downloaded")
    case = next(c for c in suite.problem_import_cases() if c.cid == "006")
    reader = PetabReader.from_yaml(case.problem_path)
    reader.derived_dir = tmp_path / "derived"
    problem = reader.to_optimization_problem(opid="case_006")
    problem.initialize(case.settings())
    _validate(to_petab(problem, tmp_path / "petab"))


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


def test_an_input_of_an_entity_outside_the_parameter_table_is_refused(
    tmp_path: Path,
) -> None:
    """The compartment is a constant of the simulation, but no parameter of PEtab."""
    network = feed_forward()
    hybridization = _before(
        network,
        inputs=_inputs(
            input0__0=NetworkInput(formulas={"e1": "default", "e2": "2 * default"}),
            input0__1=NetworkInput(formula="k"),
        ),
    )
    problem = _problem([hybridization], _elements(network))
    problem.initialize(SETTINGS)
    with pytest.raises(
        ValueError,
        match=r"'net1__input0__0' of the network 'net1'.*'default'.*sciml-input-formula",
    ):
        to_petab(problem, tmp_path / "petab")


class ChangedAlpha(LotkaVolterra):
    """The simulations change `alpha`, which the fit estimates."""

    def simulations(self) -> dict[str, Simulation]:
        return {
            sid: Simulation(
                start=0.0,
                end=10.0,
                steps=100,
                preinit_changes={"alpha": Q(1.0, "dimensionless")},
            )
            for sid in SIMULATIONS
        }


def test_a_change_of_an_estimated_parameter_is_refused(tmp_path: Path) -> None:
    problem = _problem([], [], experiment=ChangedAlpha)
    problem.initialize(SETTINGS)
    with pytest.raises(
        ValueError,
        match=r"'alpha'.*'e1'.*a parameter of the fit or a parameter an input",
    ):
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
