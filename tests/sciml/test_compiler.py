"""Tests of the compilation of networks into an SBML model.

The claim of the compilation is that the model evaluates the network: the
values of the assignment rules which roadrunner calculates are the forward
pass of the network at the same inputs.
"""

from pathlib import Path
from time import perf_counter
from typing import Any

import libsbml
import numpy as np
import pytest
import roadrunner

from sbmlsim.sciml import (
    Hybridization,
    Network,
    NetworkCompilationError,
    NetworkHybridizationError,
    NetworkInput,
    NetworkPattern,
    UnsupportedLayerError,
    compile_network,
    compiled_path,
)
from sbmlsim.sciml.backend import BackendKind
from sbmlsim.sciml.compiler import _compile, _Model
from sbmlsim.sciml.hybridization import ALL_CONDITIONS
from sbmlsim.sciml.layers import FUNCTIONS
from tests.sciml.hybrid import convolution, feed_forward, two_inputs, write_model

PRE = NetworkPattern.PRE_INITIALIZATION
RHS = NetworkPattern.RHS
OBSERVABLE = NetworkPattern.OBSERVABLE

#: relative and absolute tolerance of the rules against the forward pass
TOLERANCE = 1e-12

#: the activations which take the input alone, with their keyword arguments
ACTIVATIONS: dict[str, dict[str, Any]] = {
    name: {}
    for name, function in FUNCTIONS.items()
    if BackendKind.SYMPY in function.backends
    and name not in {"cat", "concat", "concatenate", "flatten"}
}
ACTIVATIONS["gelu"] = {"approximate": "tanh"}
ACTIVATIONS["softmax"] = {"dim": 0}
ACTIVATIONS["log_softmax"] = {"dim": 0}


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Write the model of Lotka and Volterra."""
    return write_model(tmp_path / "lv.xml")


def _hybridization(
    network: Network | None = None,
    pattern: NetworkPattern = RHS,
    target: str = "gamma",
    **kwargs: Any,
) -> Hybridization:
    network = feed_forward() if network is None else network
    arguments: dict[str, Any] = {
        "network": network,
        "pattern": pattern,
        "model": "lv",
        "inputs": {
            f"{network.sid}__input0__0": NetworkInput(formula="prey"),
            f"{network.sid}__input0__1": NetworkInput(formula="predator"),
        },
        "outputs": {f"{network.sid}__output0__0": target},
    }
    arguments.update(kwargs)
    return Hybridization(**arguments)


def _load(path: Path) -> roadrunner.RoadRunner:
    r = roadrunner.RoadRunner(str(path))
    r.integrator.absolute_tolerance = 1e-12
    r.integrator.relative_tolerance = 1e-12
    return r


def _edit(path: Path, tmp_path: Path, edit: Any) -> Path:
    """Write a copy of the model which was changed."""
    document = libsbml.readSBMLFromFile(str(path))
    edit(document.getModel())
    edited = tmp_path / "edited.xml"
    libsbml.writeSBMLToFile(document, str(edited))
    return edited


# --- THE COMPILED MODEL ---


def test_the_path_of_the_compiled_model(tmp_path: Path) -> None:
    """The model with the networks is `<stem>_sciml.xml`."""
    assert compiled_path(Path("models/lv.xml")) == Path("models/lv_sciml.xml")
    assert compiled_path(Path("models/lv.xml"), tmp_path) == tmp_path / "lv_sciml.xml"


@pytest.mark.parametrize(("level", "version"), [(3, 1), (3, 2), (2, 4)])
def test_the_rules_are_the_forward_pass(
    tmp_path: Path, level: int, version: int
) -> None:
    """The network in the right hand side is evaluated at every time point."""
    path = write_model(tmp_path / "lv.xml", level=level, version=version)
    network = feed_forward()
    compiled = compile_network(
        path, [_hybridization(network)], tmp_path / "out" / "lv_sciml.xml"
    )
    assert compiled == tmp_path / "out" / "lv_sciml.xml"
    assert compiled.is_file()

    r = _load(compiled)
    r.timeCourseSelections = ["time", "prey", "predator", "gamma", "net1__output0__0"]
    result = r.simulate(0, 5, 26)
    assert np.ptp(result[:, 3]) > 0.1
    for row in result:
        (expected,) = network.forward(np.array([row[1], row[2]]))
        assert row[3] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
        assert row[4] == row[3]


def test_the_model_which_was_compiled_is_not_changed(model_path: Path) -> None:
    """The network is written into a copy."""
    before = model_path.read_text()
    compile_network(model_path, [_hybridization()], compiled_path(model_path))
    assert model_path.read_text() == before


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_a_function_in_the_model(model_path: Path, activation: str) -> None:
    """Every function which is evaluated on expressions is math of SBML."""
    network = feed_forward(activation=activation, kwargs=ACTIVATIONS[activation])
    compiled = compile_network(
        model_path, [_hybridization(network)], compiled_path(model_path)
    )
    r = _load(compiled)
    for prey, predator in [(0.4, 4.6), (2.0, 0.1), (0.0, 0.0), (7.5, 3.0)]:
        r["init(prey)"] = prey
        r["init(predator)"] = predator
        r.reset()
        (expected,) = network.forward(np.array([prey, predator]))
        assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)


def test_the_parameters_of_the_compiled_model(model_path: Path) -> None:
    """The elements, the inputs, the units and the outputs are parameters."""
    network = feed_forward()
    compiled = compile_network(
        model_path, [_hybridization(network)], compiled_path(model_path)
    )
    document = libsbml.readSBMLFromFile(str(compiled))
    model = document.getModel()
    ids = {model.getParameter(k).getId() for k in range(model.getNumParameters())} - {
        "alpha",
        "beta",
        "gamma",
        "delta",
    }
    elements = set(network.parameter_ids())
    inputs = {"net1__input0__0", "net1__input0__1"}
    units = {f"net1__layer1__{k}" for k in range(3)}
    units |= {f"net1__act__{k}" for k in range(3)}
    units |= {"net1__layer2__0"}
    assert ids == elements | inputs | units | {"net1__output0__0"}

    for sid, (layer, name, index) in network.parameter_ids().items():
        parameter = model.getParameter(sid)
        assert parameter.getConstant()
        # libsbml writes a number with 15 digits
        assert parameter.getValue() == pytest.approx(
            network.parameters[layer][name][index], rel=1e-14
        )
        assert parameter.getUnits() == "dimensionless"
    for sid in inputs | units | {"net1__output0__0", "gamma"}:
        assert not model.getParameter(sid).getConstant()
        assert model.getRuleByVariable(sid) is not None
    # one layer deep: the rule of a unit names the units of the node before
    rule = libsbml.formulaToL3String(model.getRuleByVariable("net1__act__1").getMath())
    assert rule == "tanh(net1__layer1__1)"
    assert libsbml.formulaToL3String(model.getRuleByVariable("gamma").getMath()) == (
        "net1__output0__0"
    )


def test_an_element_of_the_model_is_a_parameter_of_a_fit(model_path: Path) -> None:
    """A change of an element changes the output, as a fit does it."""
    network = feed_forward()
    compiled = compile_network(
        model_path, [_hybridization(network)], compiled_path(model_path)
    )
    r = _load(compiled)
    sid = "net1__layer1__weight__2_1"
    r[sid] = 0.25
    r.reset()
    parameters = network.with_values({sid: 0.25})
    x = np.array([r["prey"], r["predator"]])
    (expected,) = network.forward(x, parameters=parameters)
    assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
    assert r["gamma"] != pytest.approx(network.forward(x)[0][0])


def test_a_network_in_an_observable(model_path: Path) -> None:
    """The target of an observable is a parameter which is added."""
    network = feed_forward()
    hybridization = _hybridization(network, pattern=OBSERVABLE, target="net1_output1")
    compiled = compile_network(model_path, [hybridization], compiled_path(model_path))
    r = _load(compiled)
    r.timeCourseSelections = ["time", "prey", "predator", "net1_output1", "gamma"]
    result = r.simulate(0, 5, 11)
    for row in result:
        (expected,) = network.forward(np.array([row[1], row[2]]))
        assert row[3] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
        # the right hand side is the one of the model
        assert row[4] == 0.8


def test_a_network_in_the_right_hand_side_and_an_observable(model_path: Path) -> None:
    """The two hybridizations of one network are one network in the model."""
    network = feed_forward(n_outputs=2)
    inputs = {
        "net1__input0__0": NetworkInput(formula="prey + (alpha - 1.3)"),
        "net1__input0__1": NetworkInput(formula="2 * k"),
    }
    hybridizations = [
        _hybridization(
            network,
            pattern=RHS,
            inputs=inputs,
            outputs={"net1__output0__1": "gamma"},
            constants={"k": 0.5},
        ),
        _hybridization(
            network,
            pattern=OBSERVABLE,
            inputs=inputs,
            outputs={"net1__output0__0": "y"},
            constants={"k": 0.5},
        ),
    ]
    compiled = compile_network(model_path, hybridizations, compiled_path(model_path))
    r = _load(compiled)
    assert r["k"] == 0.5
    r.timeCourseSelections = ["time", "prey", "y", "gamma"]
    for row in r.simulate(0, 5, 11):
        (expected,) = network.forward(np.array([row[1], 1.0]))
        assert row[2] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
        assert row[3] == pytest.approx(expected[1], rel=TOLERANCE, abs=TOLERANCE)


def test_two_networks_in_one_model(model_path: Path) -> None:
    """All networks of a model are compiled in one pass."""
    net1 = feed_forward("net1", seed=1)
    net2 = feed_forward("net2", seed=2, activation="relu")
    compiled = compile_network(
        model_path,
        [
            _hybridization(net1, target="gamma"),
            _hybridization(net2, target="alpha"),
            _hybridization(feed_forward("net3"), pattern=PRE, target="beta"),
        ],
        compiled_path(model_path),
    )
    r = _load(compiled)
    x = np.array([r["prey"], r["predator"]])
    assert r["gamma"] == pytest.approx(net1.forward(x)[0][0], rel=TOLERANCE)
    assert r["alpha"] == pytest.approx(net2.forward(x)[0][0], rel=TOLERANCE)
    # a network before the simulation is not a part of the model
    assert r["beta"] == 0.9
    assert "net3__output0__0" not in r.model.getGlobalParameterIds()


@pytest.mark.parametrize(("pattern", "target"), [(RHS, "gamma"), (OBSERVABLE, "y")])
def test_two_networks_with_one_target(
    model_path: Path, pattern: NetworkPattern, target: str
) -> None:
    """A target has one rule, the error names both outputs."""
    hybridizations = [
        _hybridization(feed_forward(sid, seed=seed), pattern=pattern, target=target)
        for sid, seed in (("net1", 1), ("net2", 2))
    ]
    with pytest.raises(
        NetworkCompilationError,
        match=rf"Network 'net2': the target '{target}' of 'net2__output0__0' is "
        r"set by the output 'net1__output0__0' of the network 'net1'",
    ):
        compile_network(model_path, hybridizations, compiled_path(model_path))
    assert not compiled_path(model_path).exists()


#: the largest size of the model with a softmax over 8 units: 33 KB, with a
#: maximum in every exponential 530 KB at L3V1
SOFTMAX_SIZE = 60 * 1024

#: the longest time roadrunner takes to load it, in multiples of the time it
#: takes to load a network of the same size with a tanh in the same test: 4
#: times (0.2 s against 0.05 s on an idle machine), with a maximum in every
#: exponential 500 times (28 s) at L3V1. A bound in seconds does not hold on a
#: loaded machine, the load takes 0.5 to 0.8 s with every core busy and three
#: times the idle time fails, while the load of the tanh network is slowed
#: alike. The factor is 10 times above the one and 12 times below the other.
SOFTMAX_LOAD_FACTOR = 40.0

LEVELS = [(3, 1), (3, 2), (2, 4)]


def _large_values(
    tmp_path: Path, level: int, version: int, activation: str, n_hidden: int
) -> tuple[Network, Path]:
    """Compile a network whose activation gets values which overflow `exp`."""
    path = write_model(tmp_path / "lv.xml", level=level, version=version)
    network = feed_forward(n_hidden=n_hidden, activation=activation, kwargs={"dim": 0})
    parameters = {layer: dict(arrays) for layer, arrays in network.parameters.items()}
    parameters["layer1"]["weight"] = 400.0 * parameters["layer1"]["weight"]
    network = Network(sid="net1", model=network.model, parameters=parameters)
    return network, compile_network(
        path, [_hybridization(network)], compiled_path(path)
    )


def _compare_large_values(network: Network, r: roadrunner.RoadRunner) -> None:
    """Compare the model with the forward pass where `exp` overflows."""
    weight, bias = (
        network.parameters["layer1"]["weight"],
        network.parameters["layer1"]["bias"],
    )
    logits = []
    for prey, predator in [(0.4, 4.6), (2.0, 0.1), (7.5, 3.0)]:
        r["init(prey)"] = prey
        r["init(predator)"] = predator
        r.reset()
        x = np.array([prey, predator])
        logits.append(weight @ x + bias)
        (expected,) = network.forward(x)
        assert np.isfinite(expected[0])
        assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)
    # the exponentials of the values overflow without care
    assert np.ptp(logits) > 710.0


def _load_time(compiled: Path) -> float:
    """Get the time roadrunner takes to load a compiled model."""
    start = perf_counter()
    _load(compiled)
    return perf_counter() - start


def _tanh_load_time(tmp_path: Path, level: int, version: int, n_hidden: int) -> float:
    """Get the time to load a network with a tanh, the cost of a model of its size."""
    directory = tmp_path / "tanh"
    directory.mkdir()
    path = write_model(directory / "lv.xml", level=level, version=version)
    network = feed_forward(n_hidden=n_hidden, activation="tanh")
    compiled = compile_network(path, [_hybridization(network)], compiled_path(path))
    return _load_time(compiled)


def _rules(compiled: Path, ids: list[str]) -> list[str]:
    model = libsbml.readSBMLFromFile(str(compiled)).getModel()
    return [
        libsbml.formulaToL3String(model.getRuleByVariable(sid).getMath()) for sid in ids
    ]


@pytest.mark.parametrize(("level", "version"), LEVELS)
def test_a_softmax_of_large_values(tmp_path: Path, level: int, version: int) -> None:
    """A softmax is finite where the exponentials overflow, without a maximum.

    The rule of a unit is `1 / sum_j exp(x_j - x_i)`, the model grows with the
    square of the number of units: roadrunner inlines the assignment rules,
    a maximum would be a part of every exponential.
    """
    network, compiled = _large_values(tmp_path, level, version, "softmax", 8)
    assert compiled.stat().st_size < SOFTMAX_SIZE
    start = perf_counter()
    r = _load(compiled)
    load_time = perf_counter() - start
    reference = _tanh_load_time(tmp_path, level, version, 8)
    assert load_time < SOFTMAX_LOAD_FACTOR * reference
    _compare_large_values(network, r)
    for rule in _rules(compiled, [f"net1__act__{k}" for k in range(8)]):
        assert "max(" not in rule and "piecewise(" not in rule


@pytest.mark.parametrize(("level", "version"), LEVELS)
def test_a_log_softmax_of_large_values(
    tmp_path: Path, level: int, version: int
) -> None:
    """A log_softmax is shifted by the maximum, which is written once.

    The maximum is `max` from L3V2 on and a piecewise before, a parameter of
    its own which the rules of the units name.
    """
    network, compiled = _large_values(tmp_path, level, version, "log_softmax", 4)
    _compare_large_values(network, _load(compiled))
    (maximum,) = _rules(compiled, ["net1__act__max__0"])
    assert maximum.startswith("max(" if (level, version) >= (3, 2) else "piecewise(")
    model = libsbml.readSBMLFromFile(str(compiled)).getModel()
    assert model.getParameter("net1__act__max__1") is None
    for rule in _rules(compiled, [f"net1__act__{k}" for k in range(4)]):
        assert "net1__act__max__0" in rule
        assert "max(" not in rule and "piecewise(" not in rule


def test_a_layer_without_expressions(model_path: Path) -> None:
    """The compilation of a layer without expressions names the network and node.

    `Hybridization` refuses such a network for the patterns which are
    compiled, so only a direct call of the compilation reaches the error.
    """
    hybridization = Hybridization(
        network=convolution(),
        pattern=PRE,
        model="lv",
        inputs={
            "net3__input0": NetworkInput(arrays={ALL_CONDITIONS: np.ones((1, 4, 4))})
        },
        outputs={"net3__output0__0": "gamma"},
    )
    with pytest.raises(
        UnsupportedLayerError,
        match=r"Network 'net3', node 'layer1': 'Conv2d' is not supported.*'sympy'",
    ):
        _compile(_Model(model_path, "net3"), hybridization)


def test_an_input_which_is_an_array(model_path: Path) -> None:
    """The elements of an array are constant parameters of the model."""
    network = two_inputs()

    def hybridization(arrays: dict[str, Any]) -> Hybridization:
        return Hybridization(
            network=network,
            pattern=RHS,
            model="lv",
            inputs={
                "net6__input0__0": NetworkInput(formula="prey"),
                "net6__input1": NetworkInput(arrays=arrays),
            },
            outputs={"net6__output0__0": "gamma"},
        )

    array = np.array([1.0, 2.0, 3.0])
    compiled = compile_network(
        model_path, [hybridization({ALL_CONDITIONS: array})], compiled_path(model_path)
    )
    r = _load(compiled)
    (expected,) = network.forward(np.array([r["prey"]]), array)
    assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE)

    # the arrays of conditions: the model has the values of the first one and
    # the derived changes of a fit set the ones of the simulation
    conditional = hybridization({"e2": array[::-1], "e1": array})
    compiled = compile_network(model_path, [conditional], compiled_path(model_path))
    r = _load(compiled)
    assert [r[f"net6__input1__{k}"] for k in range(3)] == [1.0, 2.0, 3.0]
    for sid, value in conditional.derived_changes({}, "e2").items():
        r[sid] = value
    r.reset()
    (expected,) = network.forward(np.array([r["prey"]]), array[::-1])
    assert r["gamma"] == pytest.approx(expected[0], rel=TOLERANCE)


def test_the_time_is_an_input(model_path: Path) -> None:
    """A formula of an input uses the time of the model."""
    network = feed_forward()
    hybridization = _hybridization(
        network,
        inputs={
            "net1__input0__0": NetworkInput(formula="time"),
            "net1__input0__1": NetworkInput(formula="0.5"),
        },
    )
    compiled = compile_network(model_path, [hybridization], compiled_path(model_path))
    r = _load(compiled)
    r.timeCourseSelections = ["time", "gamma"]
    for time, gamma in r.simulate(0, 2, 5):
        (expected,) = network.forward(np.array([time, 0.5]))
        assert gamma == pytest.approx(expected[0], rel=TOLERANCE, abs=TOLERANCE)


# --- WHAT IS NOT COMPILED ---


def test_the_error_function_is_not_math_of_sbml(model_path: Path) -> None:
    """`gelu` with the error function names its node."""
    network = feed_forward(activation="gelu", kwargs={"approximate": "none"})
    with pytest.raises(
        NetworkCompilationError, match=r"Network 'net1', node 'act'.*\['erf'\]"
    ):
        compile_network(
            model_path, [_hybridization(network)], compiled_path(model_path)
        )
    assert not compiled_path(model_path).exists()


@pytest.mark.parametrize(
    "sid",
    [
        "net1__layer1__weight__0_0",
        "net1__input0__1",
        "net1__act__2",
        "net1__output0__0",
        "k",
    ],
)
def test_an_id_of_the_model_which_is_an_id_of_the_network(
    model_path: Path, tmp_path: Path, sid: str
) -> None:
    """An entity of the model named like a part of a network is an error."""

    def add(model: libsbml.Model) -> None:
        parameter = model.createParameter()
        parameter.setId(sid)
        parameter.setValue(1.0)
        parameter.setConstant(True)

    path = _edit(model_path, tmp_path, add)
    hybridization = _hybridization(constants={"q": 1.0} if sid == "k" else {"k": 1.0})
    if sid == "k":
        compile_network(path, [hybridization], compiled_path(path))
        hybridization = _hybridization(constants={"k": 1.0})
        with pytest.raises(NetworkHybridizationError, match="'k' is an entity"):
            compile_network(path, [hybridization], compiled_path(path))
        return
    with pytest.raises(
        NetworkCompilationError, match=rf"Network 'net1'.*'{sid}'.*entity of the model"
    ):
        compile_network(path, [hybridization], compiled_path(path))


def test_two_parts_of_a_network_with_one_id(model_path: Path) -> None:
    """The ids of the units of a node and of the elements of an array differ."""
    network = feed_forward()
    model = network.model.model_copy(deep=True)
    # the units of the node are `net1__layer1__bias__0`, like the bias
    model.forward[2].name = "layer1__bias"
    model.forward[3].args = ["layer1__bias"]
    renamed = Network(sid="net1", model=model, parameters=network.parameters)
    with pytest.raises(
        NetworkCompilationError,
        match=r"unit \(0,\) of the node 'layer1__bias' and the element "
        r"'net1__layer1__bias__0' both have the id 'net1__layer1__bias__0'",
    ):
        compile_network(
            model_path, [_hybridization(renamed)], compiled_path(model_path)
        )


def _rule(model: libsbml.Model) -> None:
    model.getParameter("gamma").setConstant(False)
    rule = model.createAssignmentRule()
    rule.setVariable("gamma")
    rule.setMath(libsbml.parseL3Formula("2 * alpha"))


def _initial_assignment(model: libsbml.Model) -> None:
    assignment = model.createInitialAssignment()
    assignment.setSymbol("gamma")
    assignment.setMath(libsbml.parseL3Formula("2 * alpha"))


def _event(model: libsbml.Model) -> None:
    model.getParameter("gamma").setConstant(False)
    event = model.createEvent()
    event.setId("e1")
    event.setUseValuesFromTriggerTime(True)
    trigger = event.createTrigger()
    trigger.setMath(libsbml.parseL3Formula("time > 1"))
    trigger.setInitialValue(True)
    trigger.setPersistent(True)
    assignment = event.createEventAssignment()
    assignment.setVariable("gamma")
    assignment.setMath(libsbml.parseL3Formula("1"))


@pytest.mark.parametrize(
    ("edit", "error", "message"),
    [
        (_rule, NetworkHybridizationError, "is set by a rule of the model"),
        (
            _initial_assignment,
            NetworkCompilationError,
            "has an initial assignment in the model 'edited.xml'",
        ),
        (_event, NetworkHybridizationError, "is changed by an event of the model"),
    ],
)
def test_a_target_which_the_model_sets(
    model_path: Path,
    tmp_path: Path,
    edit: Any,
    error: type[Exception],
    message: str,
) -> None:
    """A target has one rule, the error names the target and what sets it.

    The validation of the hybridization refuses a rule and an event, the
    compiler an initial assignment.
    """
    path = _edit(model_path, tmp_path, edit)
    with pytest.raises(error, match=message) as excinfo:
        compile_network(path, [_hybridization()], compiled_path(path))
    assert "Network 'net1': the target 'gamma' of 'net1__output0__0'" in str(
        excinfo.value
    )
    assert not compiled_path(path).exists()


def _food(model: libsbml.Model) -> None:
    species = model.createSpecies()
    species.setId("food")
    species.setCompartment("default")
    species.setInitialAmount(1.0)
    species.setHasOnlySubstanceUnits(True)
    species.setBoundaryCondition(True)
    species.setConstant(True)


@pytest.mark.parametrize(
    ("target", "kind"),
    [
        ("prey", "a species"),
        ("food", "a species"),
        ("default", "a compartment"),
        ("v1", "a reaction"),
    ],
)
def test_a_target_which_is_not_a_parameter(
    model_path: Path, tmp_path: Path, target: str, kind: str
) -> None:
    """A network in the right hand side sets a parameter of the rate equations.

    A species is refused also when no reaction changes it (`food`).
    """
    path = _edit(model_path, tmp_path, _food)
    inputs = {
        "net1__input0__0": NetworkInput(formula="alpha"),
        "net1__input0__1": NetworkInput(formula="predator"),
    }
    with pytest.raises(
        NetworkHybridizationError,
        match=rf"'{target}' of 'net1__output0__0' is {kind}, but a network in the "
        r"right hand side sets a parameter",
    ):
        compile_network(
            path, [_hybridization(target=target, inputs=inputs)], compiled_path(path)
        )


def test_a_formula_with_a_symbol_the_model_does_not_have(model_path: Path) -> None:
    """A symbol of a formula is an entity of the model or a constant."""
    hybridization = _hybridization(
        inputs={
            "net1__input0__0": NetworkInput(formula="prey * kappa"),
            "net1__input0__1": NetworkInput(formula="predator"),
        }
    )
    with pytest.raises(
        NetworkCompilationError,
        match=r"Network 'net1', input 'net1__input0__0'.*'prey \* kappa' uses "
        r"\['kappa'\]",
    ):
        compile_network(model_path, [hybridization], compiled_path(model_path))


def _local_parameter(model: libsbml.Model) -> None:
    local = model.getReaction("v1").getKineticLaw().createLocalParameter()
    local.setId("k1")
    local.setValue(2.0)


def _event_id(model: libsbml.Model) -> None:
    event = model.createEvent()
    event.setId("k1")
    event.setUseValuesFromTriggerTime(True)
    trigger = event.createTrigger()
    trigger.setMath(libsbml.parseL3Formula("time > 100"))
    trigger.setInitialValue(True)
    trigger.setPersistent(True)
    assignment = event.createEventAssignment()
    assignment.setVariable("delta")
    assignment.setMath(libsbml.parseL3Formula("1"))
    model.getParameter("delta").setConstant(False)


def _function_definition(model: libsbml.Model) -> None:
    definition = model.createFunctionDefinition()
    definition.setId("k1")
    definition.setMath(libsbml.parseL3Formula("lambda(x, 2 * x)"))


@pytest.mark.parametrize("edit", [_local_parameter, _event_id, _function_definition])
def test_a_formula_with_an_id_which_has_no_value(
    model_path: Path, tmp_path: Path, edit: Any
) -> None:
    """A symbol of an input is a species, a compartment, a parameter or a reaction.

    A local parameter of a reaction, an event or a function definition has an
    id but no value in a rule.
    """
    path = _edit(model_path, tmp_path, edit)
    hybridization = _hybridization(
        inputs={
            "net1__input0__0": NetworkInput(formula="prey * k1"),
            "net1__input0__1": NetworkInput(formula="predator"),
        }
    )
    with pytest.raises(
        NetworkCompilationError,
        match=r"Network 'net1', input 'net1__input0__0'.*'prey \* k1' uses \['k1'\]",
    ):
        compile_network(path, [hybridization], compiled_path(path))


def _species_reference(model: libsbml.Model) -> None:
    model.getReaction("v1").getProduct(0).setId("sr1")


@pytest.mark.parametrize(
    ("level", "version", "symbol", "is_value"),
    [
        (3, 1, "sr1", True),
        (2, 4, "sr1", False),
        (2, 4, "v1", True),
        (2, 1, "v1", False),
    ],
)
def test_the_values_of_a_level(
    tmp_path: Path, level: int, version: int, symbol: str, is_value: bool
) -> None:
    """A species reference is a value from L3 on, a reaction from L2V2 on."""
    path = write_model(tmp_path / "lv.xml", level=level, version=version)
    if (level, version) >= (2, 2):
        path = _edit(path, tmp_path, _species_reference)
    formula = f"prey + {symbol}"
    hybridization = _hybridization(
        inputs={
            "net1__input0__0": NetworkInput(formula=formula),
            "net1__input0__1": NetworkInput(formula="predator"),
        }
    )
    if is_value:
        compile_network(path, [hybridization], compiled_path(path))
        return
    with pytest.raises(
        NetworkCompilationError,
        match=rf"Network 'net1', input 'net1__input0__0': the formula "
        rf"'prey \+ {symbol}' uses \['{symbol}'\]",
    ):
        compile_network(path, [hybridization], compiled_path(path))


def test_what_is_compiled_together(model_path: Path, tmp_path: Path) -> None:
    """The hybridizations of a call belong to one model and are compiled."""
    with pytest.raises(NetworkCompilationError, match="No network is compiled"):
        compile_network(model_path, [], compiled_path(model_path))
    with pytest.raises(
        NetworkCompilationError, match=r"No network.*\['pre_initialization'\]"
    ):
        compile_network(
            model_path, [_hybridization(pattern=PRE)], compiled_path(model_path)
        )
    with pytest.raises(NetworkCompilationError, match=r"name the models \['lv', 'x'\]"):
        compile_network(
            model_path,
            [_hybridization(), _hybridization(feed_forward("net2"), model="x")],
            compiled_path(model_path),
        )
    with pytest.raises(NetworkCompilationError, match="replaces the model"):
        compile_network(model_path, [_hybridization()], model_path)
    with pytest.raises(NetworkHybridizationError, match="does not exist"):
        compile_network(
            tmp_path / "missing.xml", [_hybridization()], compiled_path(model_path)
        )


def test_two_hybridizations_of_a_network_which_differ(model_path: Path) -> None:
    """A network is compiled once, its hybridizations share what they share."""
    network = feed_forward(n_outputs=2)
    first = _hybridization(network, outputs={"net1__output0__0": "gamma"})
    with pytest.raises(NetworkCompilationError, match="differ in 'inputs'"):
        compile_network(
            model_path,
            [
                first,
                _hybridization(
                    network,
                    outputs={"net1__output0__1": "alpha"},
                    inputs={
                        "net1__input0__0": NetworkInput(formula="prey"),
                        "net1__input0__1": NetworkInput(formula="prey"),
                    },
                ),
            ],
            compiled_path(model_path),
        )
    with pytest.raises(NetworkCompilationError, match="differ in 'network'"):
        compile_network(
            model_path,
            [
                first,
                _hybridization(
                    feed_forward(n_outputs=2, seed=5),
                    outputs={"net1__output0__1": "alpha"},
                ),
            ],
            compiled_path(model_path),
        )
    with pytest.raises(
        NetworkCompilationError, match=r"use the outputs \['net1__output0__0'\]"
    ):
        compile_network(
            model_path,
            [first, _hybridization(network, pattern=OBSERVABLE, target="y")],
            compiled_path(model_path),
        )
    with pytest.raises(
        NetworkCompilationError,
        match=r"the constant 'k' has the values '1\.0' and '2\.0'",
    ):
        compile_network(
            model_path,
            [
                _hybridization(
                    network,
                    outputs={"net1__output0__0": "gamma"},
                    constants={"k": 1.0},
                ),
                _hybridization(
                    network,
                    outputs={"net1__output0__1": "alpha"},
                    constants={"k": 2.0},
                ),
            ],
            compiled_path(model_path),
        )


def test_a_constant_of_two_networks(model_path: Path) -> None:
    """The networks of a model share a constant, which has one value."""
    net1 = feed_forward("net1", seed=1)
    net2 = feed_forward("net2", seed=2)

    def hybridizations(k2: float) -> list[Hybridization]:
        return [
            _hybridization(
                net,
                target=target,
                inputs={
                    f"{net.sid}__input0__0": NetworkInput(formula="prey"),
                    f"{net.sid}__input0__1": NetworkInput(formula="k"),
                },
                constants={"k": k},
            )
            for net, target, k in ((net1, "gamma", 0.5), (net2, "alpha", k2))
        ]

    compiled = compile_network(
        model_path, hybridizations(0.5), compiled_path(model_path)
    )
    r = _load(compiled)
    assert r["k"] == 0.5
    x = np.array([r["prey"], 0.5])
    assert r["alpha"] == pytest.approx(net2.forward(x)[0][0], rel=TOLERANCE)

    with pytest.raises(
        NetworkCompilationError,
        match=r"Network 'net2': the constant 'k' has the value '2\.0', but another "
        r"network of the model gives it the value '0\.5'",
    ):
        compile_network(model_path, hybridizations(2.0), compiled_path(model_path))
