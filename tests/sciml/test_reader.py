"""Tests of the reader of PEtab SciML problems."""

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from petab_sciml import Layer, Node
from petab_sciml.constants import ALL_CONDITION_IDS

from sbmlsim.fit import FitSettings
from sbmlsim.fit.objects import EXTERNAL_PREFIX
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2 import GapKind, from_petab, gaps_of_problem
from sbmlsim.fit.petab_v2.gaps import GAPS_BY_ID
from sbmlsim.fit.petab_v2.likelihood import gradient, log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.petab_v2.sciml import (
    NetworkEntity,
    SciMLProblemError,
    parse_entity,
)
from sbmlsim.sciml import (
    Network,
    NetworkHybridizationError,
    NetworkInput,
    NetworkPattern,
)
from sbmlsim.sciml.hybridization import ALL_CONDITIONS
from tests.sciml.hybrid import convolution, feed_forward, two_inputs
from tests.sciml.petab import write_problem, write_table


def _changes(problem: OptimizationProblem, k: int) -> dict[str, float]:
    """Get the values the simulation of a fit mapping starts with at `x0`."""
    group = next(g for g, ks in enumerate(problem.mapping_groups) if k in ks)
    plan = problem.evaluated_plan(group, np.asarray(problem.x0, dtype=float))
    return {a.target: float(a.value) for a in plan.preinit if a.value is not None}


PRE = NetworkPattern.PRE_INITIALIZATION
RHS = NetworkPattern.RHS
OBSERVABLE = NetworkPattern.OBSERVABLE

SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-12,
)

#: the mapping and the hybridization of a network with two inputs and one
#: output which are species
INPUTS = [("net1_input1", "net1.inputs[0][0]"), ("net1_input2", "net1.inputs[0][1]")]
OUTPUT = [("net1_output1", "net1.outputs[0][0]")]
SPECIES = [("net1_input1", "prey"), ("net1_input2", "predator")]


def _read(path: Path) -> PetabReader:
    reader = PetabReader.from_yaml(path)
    reader.derived_dir = path.parent / "derived"
    return reader


# --- THE MAPPING TABLE ---


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("net1.inputs[0][1]", NetworkEntity("x", "net1", "inputs", 0, (1,))),
        ("net1.inputs[0][1][2]", NetworkEntity("x", "net1", "inputs", 0, (1, 2))),
        ("net1.inputs[2]", NetworkEntity("x", "net1", "inputs", 2, None)),
        ("net1.outputs[0][0]", NetworkEntity("x", "net1", "outputs", 0, (0,))),
        ("net1.parameters", NetworkEntity("x", "net1", "parameters", key="net1")),
        (
            "net1.parameters[layer1]",
            NetworkEntity("x", "net1", "parameters", key="net1.layer1"),
        ),
        (
            "net1.parameters[block.0].weight",
            NetworkEntity("x", "net1", "parameters", key="net1.block.0.weight"),
        ),
        ("prey", None),
        ("compartment.default", None),
    ],
)
def test_the_parts_of_a_network_of_the_mapping_table(
    model_id: str, expected: NetworkEntity | None
) -> None:
    """A `modelEntityId` names an input, an output or the parameters."""
    assert parse_entity("x", model_id) == expected


@pytest.mark.parametrize(
    "model_id",
    [
        "net1.inputs",
        "net1.inputs[a]",
        "net1.inputs[0]x",
        "net1.outputs[0]",
        "net1.parameters.weight",
        "net1.parameters[layer1]weight",
        "net1.parameters[]",
    ],
)
def test_a_part_which_is_not_one(model_id: str) -> None:
    """A row which names a network and no part of it is an error."""
    with pytest.raises(
        SciMLProblemError,
        match=f"'x' to '{model_id}'".replace("[", r"\[").replace("]", r"\]"),
    ):
        parse_entity("x", model_id)


# --- THE PATTERNS ---


def test_a_network_in_the_right_hand_side(tmp_path: Path) -> None:
    """The network is compiled into the model the fit simulates."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=[*SPECIES, ("gamma", "net1_output1")],
    )
    reader = _read(path)
    assert reader.sciml is not None
    assert reader.sciml.networks["net1"] == network
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.pattern is RHS
    assert hybridization.model == "lv"
    assert hybridization.inputs == {
        "net1__input0__0": NetworkInput(formula="prey"),
        "net1__input0__1": NetworkInput(formula="predator"),
    }
    assert hybridization.outputs == {"net1__output0__0": "gamma"}
    assert hybridization.frozen == frozenset()
    assert hybridization.constants == {}

    problem = reader.to_optimization_problem()
    assert problem.hybridizations == [hybridization]
    elements = [p for p in problem.parameters if p.pid.startswith("net1__")]
    assert len(elements) == len(network.parameter_ids())
    assert all(not p.is_external for p in elements)
    assert all(p.scale is ParameterScaleType.LINEAR for p in elements)
    assert all(p.unit == "dimensionless" for p in elements)
    assert [p.pid for p in problem.parameters[:3]] == ["alpha", "beta", "delta"]
    assert all(p.scale is ParameterScaleType.LINEAR for p in problem.parameters[:3])

    source = reader.model_source()
    assert source.name == "lv_sciml.xml"
    assert source.parent == tmp_path / "derived"
    problem.initialize(SETTINGS)
    assert np.isfinite(log_likelihood(problem))
    grad = gradient(problem)
    assert np.all(np.isfinite(grad.to_numpy()))
    assert np.all(grad[[p.pid for p in elements]].abs() > 0.0)


def test_a_network_before_the_simulation(tmp_path: Path) -> None:
    """The inputs are parameters of the table, the elements are external."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": True},
        mapping=[("k1", "net1.inputs[0][0]"), ("k2", "net1.inputs[0][1]"), *OUTPUT],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "k1", "nominalValue": 1.0, "estimate": False},
            {
                "parameterId": "k2",
                "nominalValue": 2.0,
                "estimate": True,
                "lowerBound": 0.1,
                "upperBound": 10.0,
                "parameterScale": "log10",
            },
        ],
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.pattern is PRE
    assert hybridization.inputs == {
        "net1__input0__0": NetworkInput(formula="k1"),
        "net1__input0__1": NetworkInput(formula="k2"),
    }
    # the parameter which is not estimated is a constant, the estimated one
    # is a parameter of the fit which is not an entity of the model
    assert hybridization.constants == {"k1": 1.0}
    problem = reader.to_optimization_problem()
    by_id = {p.pid: p for p in problem.parameters}
    assert by_id["k2"].target == f"{EXTERNAL_PREFIX}k2"
    assert by_id["k2"].scale is ParameterScaleType.LOG10
    assert by_id["k2"].start_value == 2.0
    assert by_id["k2"].unit == "dimensionless"
    elements = [p for p in problem.parameters if p.pid.startswith("net1__")]
    assert all(p.target == f"{EXTERNAL_PREFIX}{p.pid}" for p in elements)
    assert reader.model_source() == tmp_path / "lv.xml"

    problem.initialize(SETTINGS)
    x = np.asarray(problem.x0, dtype=float)
    problem.predictions(x)
    changes = _changes(problem, 0)
    (expected,) = network.forward(np.array([1.0, 2.0]))
    assert changes["gamma"] == pytest.approx(expected[0])


def test_a_network_in_an_observable(tmp_path: Path) -> None:
    """The output is the symbol of the observable formula."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=SPECIES,
        observables={"prey_o": "net1_output1 - 0.9 + prey", "predator_o": "predator"},
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.pattern is OBSERVABLE
    assert hybridization.outputs == {"net1__output0__0": "net1_output1"}
    assert reader.model_source().name == "lv_sciml_observables.xml"
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    k = problem.mapping_keys.index("prey_o")
    assert problem.yid_observable[k] == "observable_prey_o"
    predictions = problem.predictions(np.asarray(problem.x0, dtype=float))
    assert np.all(np.isfinite(predictions[k]))
    assert not np.allclose(
        predictions[k], predictions[problem.mapping_keys.index("predator_o")]
    )


def test_a_network_with_outputs_of_two_patterns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A network in the right hand side and an observable is two hybridizations.

    The parameters of the network are resolved once.
    """
    network = feed_forward(n_outputs=2)
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[
            *INPUTS,
            ("net1_output1", "net1.outputs[0][0]"),
            ("net1_output2", "net1.outputs[0][1]"),
        ],
        hybridization=[*SPECIES, ("gamma", "net1_output2")],
        observables={"prey_o": "net1_output1", "predator_o": "predator"},
    )
    reader = _read(path)
    assert reader.sciml is not None
    with caplog.at_level(logging.INFO, logger="sbmlsim.sciml.parameters"):
        observable, rhs = sorted(reader.sciml.hybridizations(), key=lambda h: h.pattern)
        problem = reader.to_optimization_problem()
    assert caplog.text.count("elements are estimated") == 1
    assert rhs.pattern is RHS
    assert rhs.outputs == {"net1__output0__1": "gamma"}
    assert observable.pattern is OBSERVABLE
    assert observable.outputs == {"net1__output0__0": "net1_output1"}
    problem.initialize(SETTINGS)
    assert np.isfinite(log_likelihood(problem))


def test_a_frozen_layer(tmp_path: Path) -> None:
    """A row of a layer which is not estimated freezes its elements."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[*INPUTS, *OUTPUT, ("net1_layer1", "net1.parameters[layer1]")],
        hybridization=[*SPECIES, ("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "net1_layer1", "nominalValue": "array", "estimate": False}
        ],
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.frozen == {
        sid for sid in network.parameter_ids() if "layer1" in sid
    }
    problem = reader.to_optimization_problem()
    assert not [p for p in problem.parameters if "layer1" in p.pid]
    assert [p for p in problem.parameters if "layer2" in p.pid]


def test_the_nominal_values_of_the_parameter_table(tmp_path: Path) -> None:
    """A number in the parameter table sets the elements it covers."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": True},
        mapping=[
            ("k1", "net1.inputs[0][0]"),
            ("k2", "net1.inputs[0][1]"),
            *OUTPUT,
            ("net1_layer2_bias", "net1.parameters[layer2].bias"),
        ],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "k1", "nominalValue": 1.0, "estimate": False},
            {"parameterId": "k2", "nominalValue": 2.0, "estimate": False},
            {
                "parameterId": "net1_layer2_bias",
                "nominalValue": 0.25,
                "estimate": True,
                "lowerBound": -1.0,
                "upperBound": 1.0,
            },
        ],
    )
    reader = _read(path)
    assert reader.sciml is not None
    read = reader.sciml.networks["net1"]
    np.testing.assert_array_equal(read.parameters["layer2"]["bias"], [0.25])
    np.testing.assert_array_equal(
        read.parameters["layer1"]["weight"], network.parameters["layer1"]["weight"]
    )
    problem = reader.to_optimization_problem()
    by_id = {p.pid: p for p in problem.parameters}
    assert by_id["net1__layer2__bias__0"].start_value == 0.25


# --- THE CONDITIONS ---


def test_the_inputs_of_the_conditions(tmp_path: Path) -> None:
    """A condition sets an input, a number or a parameter of the table."""
    network = feed_forward()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": True},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "k1", "nominalValue": 1.0, "estimate": False},
            {"parameterId": "k2", "nominalValue": 2.0, "estimate": False},
        ],
        conditions=[
            ("cond1", "net1_input1", "10.0"),
            ("cond1", "net1_input2", "20.0"),
            ("cond2", "net1_input1", "k1"),
            ("cond2", "net1_input2", "k2"),
        ],
        experiments={"e1": "cond1", "e2": "cond2"},
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.inputs == {
        "net1__input0__0": NetworkInput(formulas={"e1": "10", "e2": "k1"}),
        "net1__input0__1": NetworkInput(formulas={"e1": "20", "e2": "k2"}),
    }
    assert hybridization.constants == {"k1": 1.0, "k2": 2.0}
    problem = reader.to_optimization_problem()
    # one fit mapping per observable and experiment
    assert sorted(
        problem.mapping_collections[0].mappings
        + problem.mapping_collections[1].mappings
    ) == [
        "predator_o_e1",
        "predator_o_e2",
        "prey_o_e1",
        "prey_o_e2",
    ]
    problem.initialize(SETTINGS)
    assert problem.simulation_keys == ["e1", "e1", "e2", "e2"]
    problem.predictions(np.asarray(problem.x0, dtype=float))
    for sid, inputs in (("e1", [10.0, 20.0]), ("e2", [1.0, 2.0])):
        k = problem.simulation_keys.index(sid)
        changes = _changes(problem, k)
        (expected,) = network.forward(np.array(inputs))
        assert changes["gamma"] == pytest.approx(expected[0])
        # the conditions set no change of the model
        assert "net1_input1" not in changes


def test_the_arrays_of_the_conditions(tmp_path: Path) -> None:
    """An array file holds the array of every condition, or of all of them."""
    network = convolution()
    arrays = {"cond1": np.ones((1, 4, 4)), "cond2": 2.0 * np.ones((1, 4, 4))}
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net3": True},
        mapping=[("input0", "net3.inputs[0]"), ("net3_output1", "net3.outputs[0][0]")],
        hybridization=[("input0", "array"), ("gamma", "net3_output1")],
        inputs={"input0": arrays},
        experiments={"e1": "cond1", "e2": "cond2"},
    )
    reader = _read(path)
    assert reader.sciml is not None
    assert reader.sciml.condition_ids == {"cond1", "cond2"}
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.inputs == {
        "net3__input0": NetworkInput(
            arrays={"e1": arrays["cond1"], "e2": arrays["cond2"]}
        )
    }
    # the output of the convolution has the axis of the batch
    assert hybridization.outputs == {"net3__output0__0": "gamma"}
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    problem.predictions(np.asarray(problem.x0, dtype=float))
    k = problem.simulation_keys.index("e2")
    (expected,) = network.forward(arrays["cond2"])
    assert _changes(problem, k)["gamma"] == (pytest.approx(expected[0]))

    # one array for every condition
    path = write_problem(
        tmp_path / "all",
        networks=[network],
        pre_initialization={"net3": True},
        mapping=[("input0", "net3.inputs[0]"), ("net3_output1", "net3.outputs[0][0]")],
        hybridization=[("input0", "array"), ("gamma", "net3_output1")],
        inputs={"input0": {ALL_CONDITION_IDS: arrays["cond1"]}},
    )
    reader = _read(path)
    assert reader.sciml is not None
    (hybridization,) = reader.sciml.hybridizations()
    assert hybridization.inputs == {
        "net3__input0": NetworkInput(arrays={ALL_CONDITIONS: arrays["cond1"]})
    }


def test_an_array_of_a_network_in_the_right_hand_side(tmp_path: Path) -> None:
    """The arrays of the conditions are changes of the compiled model."""
    network = two_inputs()
    arrays = {"cond1": np.array([1.0, 2.0, 3.0]), "cond2": np.array([3.0, 2.0, 1.0])}
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net6": False},
        mapping=[
            ("net6_input1", "net6.inputs[0][0]"),
            ("net6_input2", "net6.inputs[1]"),
            ("net6_output1", "net6.outputs[0][0]"),
        ],
        hybridization=[
            ("net6_input1", "prey"),
            ("net6_input2", "array"),
            ("gamma", "net6_output1"),
        ],
        inputs={"net6_input2": arrays},
        experiments={"e1": "cond1", "e2": "cond2"},
    )
    reader = _read(path)
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    problem.predictions(np.asarray(problem.x0, dtype=float))
    k = problem.simulation_keys.index("e2")
    changes = _changes(problem, k)
    assert [changes[f"net6__input1__{i}"] for i in range(3)] == [
        3.0,
        2.0,
        1.0,
    ]


def _periods(path: Path, rows: list[tuple[str, float, str]]) -> None:
    """Replace the experiments of a problem by periods, id, time and condition."""
    write_table(
        path.parent / "experiments.tsv",
        [
            {"experimentId": sid, "time": time, "conditionId": condition}
            for sid, time, condition in rows
        ],
    )


@pytest.mark.parametrize(
    ("conditions", "message"),
    [
        (
            # the input changes in the second period
            [
                ("cond1", "net1_input1", "10.0"),
                ("cond1", "net1_input2", "20.0"),
                ("cond2", "net1_input1", "3.0"),
            ],
            r"experiment 'e1'.*condition 'cond2'.*period at the time 5\.0 sets "
            r"\['net1_input1'\]",
        ),
        (
            # only the second period sets the input
            [("cond1", "delta", "1.8"), ("cond2", "net1_input1", "3.0")],
            r"experiment 'e1'.*condition 'cond2'.*sets \['net1_input1'\]",
        ),
    ],
)
def test_an_input_of_a_later_period(
    tmp_path: Path, conditions: list[tuple[str, str, str]], message: str
) -> None:
    """A network before the simulation is evaluated once per simulation."""
    path = write_problem(
        tmp_path,
        networks=[feed_forward()],
        pre_initialization={"net1": True},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "net1_input2", "nominalValue": 2.0, "estimate": False}
        ],
        conditions=conditions,
        experiments={"e1": "cond1"},
    )
    _periods(path, [("e1", 0.0, "cond1"), ("e1", 5.0, "cond2")])
    with pytest.raises(SciMLProblemError, match=message):
        _read(path)


def test_an_input_of_the_main_period_after_a_pre_equilibration(
    tmp_path: Path,
) -> None:
    """The pre-equilibration is the first period, it gives the inputs."""
    path = write_problem(
        tmp_path,
        networks=[feed_forward()],
        pre_initialization={"net1": True},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "net1_input2", "nominalValue": 2.0, "estimate": False}
        ],
        conditions=[
            ("cond1", "net1_input1", "10.0"),
            ("cond2", "net1_input1", "3.0"),
        ],
        experiments={"e1": "cond1"},
    )
    _periods(path, [("e1", -np.inf, "cond1"), ("e1", 0.0, "cond2")])
    with pytest.raises(
        SciMLProblemError,
        match=r"period at the time 0\.0 sets \['net1_input1'\].*main period "
        r"after the pre-equilibration",
    ):
        _read(path)


def test_an_array_of_a_later_period(tmp_path: Path) -> None:
    """The array of an input is selected by the first period."""
    network = convolution()
    arrays = {"cond1": np.ones((1, 4, 4)), "cond2": 2.0 * np.ones((1, 4, 4))}
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net3": True},
        mapping=[("input0", "net3.inputs[0]"), ("net3_output1", "net3.outputs[0][0]")],
        hybridization=[("input0", "array"), ("gamma", "net3_output1")],
        inputs={"input0": arrays},
        experiments={"e1": "cond1"},
    )
    _periods(path, [("e1", 0.0, "cond1"), ("e1", 5.0, "cond2")])
    with pytest.raises(
        SciMLProblemError,
        match=r"experiment 'e1'.*condition 'cond2'.*arrays of \['input0'\]",
    ):
        _read(path)


def test_an_input_of_a_condition_which_starts_no_experiment(tmp_path: Path) -> None:
    """An input which only a condition sets that no experiment uses has no value."""
    path = write_problem(
        tmp_path,
        networks=[feed_forward()],
        pre_initialization={"net1": True},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=[("gamma", "net1_output1")],
        parameters=[
            {"parameterId": "net1_input2", "nominalValue": 2.0, "estimate": False}
        ],
        conditions=[("cond9", "net1_input1", "3.0")],
    )
    with pytest.raises(
        SciMLProblemError,
        match=r"input 'net1_input1'.*\['cond9'\].*no experiment starts with",
    ):
        _read(path)


def test_the_hybridizations_are_read_once(tmp_path: Path) -> None:
    """The compiled model and the problem are built from the same objects."""
    reader = _read(_problem(tmp_path))
    assert reader.sciml is not None
    (first,) = reader.sciml.hybridizations()
    (second,) = reader.sciml.hybridizations()
    assert first is second
    problem = reader.to_optimization_problem()
    assert problem.hybridizations[0] is first


# --- WHAT IS NOT READ ---


def _problem(tmp_path: Path, **kwargs: Any) -> Path:
    """Write the problem of a network in the right hand side, changed by kwargs."""
    arguments: dict[str, Any] = {
        "networks": [feed_forward()],
        "pre_initialization": {"net1": False},
        "mapping": [*INPUTS, *OUTPUT],
        "hybridization": [*SPECIES, ("gamma", "net1_output1")],
    }
    arguments.update(kwargs)
    return write_problem(tmp_path, **arguments)


def test_a_format_which_is_not_read(tmp_path: Path) -> None:
    """Only the format `YAML` is read, the others are a gap."""
    path = _problem(tmp_path, formats={"net1": "pytorch"})
    with pytest.raises(
        SciMLProblemError, match=r"'pytorch'.*sciml-model-format"
    ) as excinfo:
        _read(path)
    assert excinfo.value.gap == "sciml-model-format"


def test_two_array_files_of_a_network(tmp_path: Path) -> None:
    """The values of a network are in one array file, two would be a guess."""
    path = _problem(tmp_path)
    shutil.copy(tmp_path / "arrays.hdf5", tmp_path / "arrays2.hdf5")
    config = yaml.safe_load(path.read_text())
    config["extensions"]["sciml"]["array_files"].append("arrays2.hdf5")
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(
        SciMLProblemError,
        match=r"Network 'net1': the array files \['arrays\.hdf5', "
        r"'arrays2\.hdf5'\] each hold its arrays",
    ):
        _read(path).to_optimization_problem()


def test_a_prior_of_a_network(tmp_path: Path) -> None:
    """Priors on the parameters of a network are not read (#190)."""
    path = _problem(
        tmp_path,
        parameters=[
            {
                "parameterId": "net1_layer1",
                "nominalValue": "array",
                "estimate": True,
                "lowerBound": "-inf",
                "upperBound": "inf",
                "priorDistribution": "normal",
                "priorParameters": "0.0;1.0",
            }
        ],
        mapping=[*INPUTS, *OUTPUT, ("net1_layer1", "net1.parameters[layer1]")],
    )
    with pytest.raises(
        SciMLProblemError, match=r"prior 'normal'.*sciml-priors"
    ) as excinfo:
        _read(path)
    assert excinfo.value.gap == "sciml-priors"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"mapping": [*INPUTS, *OUTPUT, ("x", "net9.inputs[0][0]")]},
            r"names the network 'net9'",
        ),
        (
            {"hybridization": [*SPECIES, ("gamma", "2 * net1_output1")]},
            r"assigns 'gamma' the value '2\.0\*net1_output1'",
        ),
        (
            {"hybridization": [("net1_input1", "prey"), ("gamma", "net1_output1")]},
            "the input 'net1_input2' has no value",
        ),
        (
            {
                "hybridization": [
                    *SPECIES,
                    ("gamma", "net1_output1"),
                    ("alpha", "net1_output1"),
                ]
            },
            "assigns the output 'net1_output1' to 'gamma' and to 'alpha'",
        ),
        (
            {"hybridization": SPECIES},
            "no output of the network is used",
        ),
        (
            {"hybridization": [*SPECIES, ("gamma", "net1_output1"), ("gamma", "prey")]},
            "assigns 'gamma' twice",
        ),
        (
            {
                "pre_initialization": {"net1": True},
                "observables": {"prey_o": "net1_output1", "predator_o": "predator"},
                "hybridization": [],
                "mapping": [
                    ("k1", "net1.inputs[0][0]"),
                    ("k2", "net1.inputs[0][1]"),
                    *OUTPUT,
                ],
                "parameters": [
                    {"parameterId": "k1", "nominalValue": 1.0, "estimate": False},
                    {"parameterId": "k2", "nominalValue": 1.0, "estimate": False},
                ],
            },
            "used by an observable, but the network runs before the simulation",
        ),
        (
            {
                "mapping": [("net1_input1", "net1.inputs[0][0]"), *OUTPUT],
                "hybridization": [("net1_input1", "prey"), ("gamma", "net1_output1")],
            },
            r"inputs of the shapes \[\(1,\)\] do not fit",
        ),
        (
            {"hybridization": [*SPECIES, ("gamma", "net1_output1"), ("net1_ps", "1")]},
            r"assigns 'net1_ps' the value '1\.0', but 'net1_ps' is the parameters "
            r"of the network 'net1'",
        ),
        (
            {
                "hybridization": [
                    *SPECIES,
                    ("gamma", "net1_output1"),
                    ("net1_output1", "1"),
                ]
            },
            r"assigns 'net1_output1' the value '1\.0', but 'net1_output1' is the "
            r"outputs of the network 'net1'",
        ),
        (
            {
                "hybridization": [
                    ("net1_input1", "net1_output1"),
                    ("net1_input2", "predator"),
                ]
            },
            "assigns the output 'net1_output1' to the input 'net1_input1'",
        ),
        (
            {"mapping": [*INPUTS, *OUTPUT, ("net1_output9", "net1.outputs[0][7]")]},
            r"'net1_output9' to 'net1\.outputs\[0\]\[7\]' is not an element of "
            r"the outputs of the shapes \[\(1,\)\]",
        ),
        (
            {"mapping": [*INPUTS, *OUTPUT, ("net1_output9", "net1.outputs[2][0]")]},
            r"'net1_output9' to 'net1\.outputs\[2\]\[0\]' is not an element",
        ),
    ],
)
def test_a_problem_which_cannot_be_read(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    """The error names the network, the input, the output or the row."""
    path = _problem(tmp_path, **kwargs)
    with pytest.raises((SciMLProblemError, NetworkHybridizationError), match=message):
        _read(path).to_optimization_problem()


def test_a_convolution_in_the_right_hand_side(tmp_path: Path) -> None:
    """A network which is not evaluated on expressions is not compiled."""
    network = convolution()
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net3": False},
        mapping=[("input0", "net3.inputs[0]"), ("net3_output1", "net3.outputs[0][0]")],
        hybridization=[("input0", "array"), ("gamma", "net3_output1")],
        inputs={"input0": {ALL_CONDITION_IDS: np.ones((1, 4, 4))}},
    )
    with pytest.raises(
        SciMLProblemError,
        match=r"Network 'net3'.*node 'layer1' \(Conv2d\).*sciml-layer-sbml",
    ) as excinfo:
        _read(path).to_optimization_problem()
    assert excinfo.value.gap == "sciml-layer-sbml"


def test_a_gelu_in_the_right_hand_side(tmp_path: Path) -> None:
    """The error function has no MathML, the compiler refuses it with the gap."""
    network = feed_forward(activation="gelu", kwargs={"approximate": "none"})
    path = write_problem(
        tmp_path,
        networks=[network],
        pre_initialization={"net1": False},
        mapping=[*INPUTS, *OUTPUT],
        hybridization=[*SPECIES, ("gamma", "net1_output1")],
    )
    reader = _read(path)
    with pytest.raises(
        SciMLProblemError, match=r"Network 'net1', node 'act'.*\['erf'\]"
    ) as excinfo:
        reader.model_source()
    assert excinfo.value.gap == "sciml-layer-sbml"


def test_from_petab(tmp_path: Path) -> None:
    """`from_petab` reads a problem with networks like any other."""
    path = _problem(tmp_path)
    problem, settings = from_petab(path)
    assert problem.hybridizations
    assert settings == FitSettings()
    problem.initialize(SETTINGS)
    assert np.isfinite(log_likelihood(problem))


# --- THE GAPS ---


def test_the_gaps_of_a_hybrid_problem(tmp_path: Path) -> None:
    """A problem with networks runs into the scale of a parameter."""
    problem, _ = from_petab(_problem(tmp_path))
    problem.initialize(SETTINGS)
    ids = {gap.id for gap in gaps_of_problem(problem)}
    assert "sciml-parameter-scale" in ids
    assert "sciml-training-mode" not in ids
    assert GAPS_BY_ID["sciml-parameter-scale"].kind is GapKind.EXTENSION
    for gap_id in ("sciml-model-format", "sciml-layer-sbml", "sciml-priors"):
        assert GAPS_BY_ID[gap_id].kind is GapKind.UNSUPPORTED
    assert GAPS_BY_ID["sciml-training-mode"].kind is GapKind.LOSSY


def test_the_gap_of_the_training_mode(tmp_path: Path) -> None:
    """A network with dropout is evaluated in evaluation mode."""
    network = feed_forward()
    model = network.model.model_copy(deep=True)
    model.layers.append(Layer(layer_id="drop", layer_type="Dropout", args={"p": 0.5}))
    model.forward.insert(
        2,
        Node(name="drop", op="call_module", target="drop", args=["layer1"], kwargs={}),
    )
    model.forward[3].args = ["drop"]
    with_dropout = Network(sid="net1", model=model, parameters=network.parameters)
    problem, _ = from_petab(_problem(tmp_path, networks=[with_dropout]))
    problem.initialize(SETTINGS)
    assert "sciml-training-mode" in {gap.id for gap in gaps_of_problem(problem)}


# --- THE INPUTS WHICH BITE ---


def test_a_problem_with_two_models(tmp_path: Path) -> None:
    """A problem with networks has one model, the error names the models."""
    path = _problem(
        tmp_path,
        extra_yaml={
            "model_files": {
                "lv": {"location": "lv.xml", "language": "sbml"},
                "lv2": {"location": "lv.xml", "language": "sbml"},
            }
        },
    )
    with pytest.raises(SciMLProblemError, match=r"one model.*\['lv', 'lv2'\]"):
        _read(path)


def test_a_condition_which_sets_the_input_of_a_compiled_network(tmp_path: Path) -> None:
    """A network in the right hand side has one formula per input."""
    path = _problem(
        tmp_path,
        hybridization=[("net1_input2", "predator"), ("gamma", "net1_output1")],
        conditions=[("cond1", "net1_input1", "prey"), ("cond2", "net1_input1", "1.0")],
        experiments={"e1": "cond1", "e2": "cond2"},
    )
    with pytest.raises(NetworkHybridizationError, match="one formula per input"):
        _read(path).to_optimization_problem()


def test_an_output_in_the_right_hand_side_and_an_observable(tmp_path: Path) -> None:
    """An output sets an entity or is a symbol of an observable, not both."""
    path = _problem(
        tmp_path, observables={"prey_o": "net1_output1", "predator_o": "predator"}
    )
    with pytest.raises(
        SciMLProblemError, match="'net1_output1' is used by an observable and assigned"
    ):
        _read(path).to_optimization_problem()


def test_a_problem_which_petab_read_with_torch(tmp_path: Path) -> None:
    """A problem `petab` read with its networks is read from its configuration."""
    pytest.importorskip("torch")
    from petab.v2 import Problem as PetabProblem

    petab_problem = PetabProblem.from_yaml(_problem(tmp_path))
    assert petab_problem.extensions.sciml is not None
    reader = PetabReader(petab_problem)
    reader.derived_dir = tmp_path / "derived"
    assert reader.sciml is not None
    assert reader.sciml.networks["net1"] == feed_forward()
    problem = reader.to_optimization_problem()
    assert len(problem.hybridizations) == 1
