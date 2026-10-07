"""Tests of a fit of a hybrid problem which is defined in python.

The problem is `tests.sciml.experiment.LotkaVolterra` with networks of the
three patterns, without PEtab.
"""

import pickle
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2.likelihood import gradient, log_likelihood
from sbmlsim.fit.runner import run_optimization
from sbmlsim.sciml import (
    Hybridization,
    Network,
    NetworkInput,
    NetworkPattern,
    compile_network,
    compiled_path,
    network_fit_parameters,
    nominal_parameters,
)
from sbmlsim.sciml.hybridization import ALL_CONDITIONS
from tests.sciml.experiment import LotkaVolterra, collections
from tests.sciml.hybrid import MODEL_PATH, feed_forward, two_inputs


def _changes(problem: OptimizationProblem, k: int) -> dict[str, float]:
    """Get the values the simulation of a fit mapping starts with at `x0`."""
    group = next(g for g, ks in enumerate(problem.mapping_groups) if k in ks)
    plan = problem.evaluated_plan(group, np.asarray(problem.x0, dtype=float))
    return {a.target: float(a.value) for a in plan.preinit if a.value is not None}


PRE = NetworkPattern.PRE_INITIALIZATION
RHS = NetworkPattern.RHS

#: settings with which two simulations of the problem agree
SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-12,
)

#: the parameters of the model which are fitted
MECHANISTIC = [
    FitParameter("alpha", 1.3, 0.0, 15.0, unit="dimensionless"),
    FitParameter("beta", 0.9, 0.0, 15.0, unit="dimensionless"),
]


def _problem(
    hybridizations: Sequence[Hybridization],
    parameters: Sequence[FitParameter],
    experiment: type[LotkaVolterra] = LotkaVolterra,
) -> OptimizationProblem:
    return OptimizationProblem(
        opid="hybrid",
        mapping_collections=collections(experiment),
        fit_parameters=[*MECHANISTIC, *parameters],
        base_path=MODEL_PATH.parent,
        data_path=MODEL_PATH.parent,
        hybridizations=list(hybridizations),
    )


def _before(network: Network, **kwargs: object) -> Hybridization:
    """Get the network before the simulation, `gamma = net(alpha, k)`."""
    arguments: dict[str, object] = {
        "network": network,
        "pattern": PRE,
        "model": "lv",
        "inputs": {
            f"{network.sid}__input0__0": NetworkInput(formula="alpha"),
            f"{network.sid}__input0__1": NetworkInput(formula="k"),
        },
        "outputs": {f"{network.sid}__output0__0": "gamma"},
        "constants": {"k": 0.5},
    }
    arguments.update(kwargs)
    return Hybridization(**arguments)  # ty: ignore[invalid-argument-type]


# --- BEFORE THE SIMULATION ---


def test_a_network_before_the_simulation() -> None:
    """The elements are parameters of the fit which are not in the model."""
    network = feed_forward()
    hybridization = _before(network)
    elements = network_fit_parameters(
        network, estimate={"net1": True}, bounds={}, external=True
    )
    problem = _problem([hybridization], elements)
    problem.initialize(SETTINGS)
    assert problem.pids == ["alpha", "beta", *network.parameter_ids()]
    assert problem.scales_initialized == [ParameterScaleType.LINEAR] * len(problem.pids)
    # the elements start from their nominal values
    x = np.asarray(problem.x0, dtype=float)
    np.testing.assert_allclose(problem.xmodel, x)

    problem.predictions(x)
    changes = _changes(problem, 0)
    (expected,) = network.forward(np.array([1.3, 0.5]))
    assert changes["gamma"] == pytest.approx(expected[0])

    # a plain problem with the value of gamma has the same predictions
    plain = OptimizationProblem(
        opid="plain",
        mapping_collections=collections(),
        fit_parameters=[
            *MECHANISTIC,
            FitParameter("gamma", float(expected[0]), unit="dimensionless"),
        ],
        base_path=MODEL_PATH.parent,
        data_path=MODEL_PATH.parent,
    )
    plain.initialize(SETTINGS)
    for k, values in problem.predictions(x).items():
        np.testing.assert_allclose(
            values, plain.predictions(np.array([1.3, 0.9, expected[0]]))[k], rtol=1e-9
        )

    # the objective depends on every element
    grad = gradient(problem, order=4)
    assert np.all(np.isfinite(grad.to_numpy()))
    assert np.all(grad[list(network.parameter_ids())].abs() > 0.0)
    assert np.isfinite(log_likelihood(problem))


def test_the_arrays_of_the_simulations() -> None:
    """Every simulation is evaluated with the arrays of its condition."""
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
    problem = _problem([hybridization], elements)
    problem.initialize(SETTINGS)
    x = np.asarray(problem.x0, dtype=float)
    predictions = problem.predictions(x, indices=problem.indices())
    e1 = predictions[problem.mapping_keys.index("prey_e1")]
    e2 = predictions[problem.mapping_keys.index("prey_e2")]
    assert not np.allclose(e1, e2)
    for sid, array in (("e1", [1.0, 2.0, 3.0]), ("e2", [3.0, 2.0, 1.0])):
        k = problem.simulation_keys.index(sid)
        changes = _changes(problem, k)
        (expected,) = network.forward(np.array([1.3]), np.array(array))
        assert changes["gamma"] == pytest.approx(expected[0])


def test_a_frozen_layer() -> None:
    """The elements of a frozen layer are not parameters and keep their values."""
    network = feed_forward()
    elements, hybridization = _before(network).fit_parameters(
        estimate={"net1": True, "net1.layer1": False}, bounds={"net1": (-5.0, 5.0)}
    )
    assert hybridization.frozen == {
        sid for sid in network.parameter_ids() if "layer1" in sid
    }
    problem = _problem([hybridization], elements)
    problem.initialize(SETTINGS)
    assert all(np.isfinite(problem.bounds[0]))
    grad = gradient(problem)
    assert set(grad.index) == {"alpha", "beta", *(p.pid for p in elements)}

    with pytest.raises(ValueError, match=r"\['net1__layer1__bias__0'\] are frozen"):
        _problem(
            [hybridization],
            [
                *elements,
                FitParameter(
                    "net1__layer1__bias__0",
                    0.0,
                    unit="dimensionless",
                    target="sciml:net1__layer1__bias__0",
                ),
            ],
        ).initialize(SETTINGS)


def test_a_parallel_fit_pickles_the_networks(tmp_path: Path) -> None:
    """The workers of a parallel fit get the hybridizations with the problem."""
    network = feed_forward()
    # an element which is not estimated is frozen
    elements, hybridization = _before(network).fit_parameters(
        estimate={"net1.layer2": True}
    )
    problem = _problem([hybridization], elements)
    restored = pickle.loads(pickle.dumps(problem))
    assert restored.hybridizations == problem.hybridizations
    assert restored.hybridizations[0].network == network

    result = run_optimization(
        problem,
        settings=SETTINGS,
        size=2,
        n_cores=2,
        seed=1,
        show_progress=False,
        max_nfev=2,
    )
    assert len(result.fits) == 2
    assert all(np.all(np.isfinite(fit.x)) for fit in result.fits)
    assert all("Error" not in fit.message for fit in result.fits)


# --- IN THE RIGHT HAND SIDE ---


def test_a_network_in_the_right_hand_side(tmp_path: Path) -> None:
    """The elements are parameters of the model with the network."""
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
    compiled = compile_network(
        MODEL_PATH, [hybridization], compiled_path(MODEL_PATH, tmp_path)
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    elements = network_fit_parameters(network, estimate={"net1": True}, bounds={})
    assert all(not p.is_external for p in elements)
    problem = _problem([hybridization], elements, experiment=Compiled)
    problem.initialize(SETTINGS)
    # the network is a part of the model, nothing is derived
    assert [
        [(hook, set(values)) for hook, values in derived]
        for derived in problem.group_derived
    ] == [[(hybridization, {"net1__output0__0"})]] * 2
    x = np.asarray(problem.x0, dtype=float)
    np.testing.assert_allclose(problem.xmodel, x, rtol=1e-14)
    cost = problem.cost_least_square(x)
    shifted = x.copy()
    shifted[problem.pids.index("net1__layer2__bias__0")] += 0.1
    assert problem.cost_least_square(shifted) != pytest.approx(cost)
    grad = gradient(problem)
    assert np.all(grad[list(network.parameter_ids())].abs() > 0.0)


def test_the_arrays_of_the_simulations_of_a_compiled_network(tmp_path: Path) -> None:
    """The elements of an array of a condition are set before its simulation."""
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
    )
    compiled = compile_network(
        MODEL_PATH, [hybridization], compiled_path(MODEL_PATH, tmp_path)
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    problem = _problem([hybridization], [], experiment=Compiled)
    problem.initialize(SETTINGS)
    x = np.asarray(problem.x0, dtype=float)
    predictions = problem.predictions(x, indices=problem.indices())
    e1 = predictions[problem.mapping_keys.index("prey_e1")]
    e2 = predictions[problem.mapping_keys.index("prey_e2")]
    assert not np.allclose(e1, e2)
    k = problem.simulation_keys.index("e2")
    changes = _changes(problem, k)
    assert [changes[f"net6__input1__{i}"] for i in range(3)] == [
        3.0,
        2.0,
        1.0,
    ]


def test_a_network_which_the_model_does_not_have() -> None:
    """The elements of a compiled network are entities of the compiled model."""
    network = feed_forward()
    elements = network_fit_parameters(network, estimate={"net1": True}, bounds={})
    with pytest.raises(
        ValueError,
        match=r"FitParameter 'net1__layer1__weight__0_0' writes "
        r"'net1__layer1__weight__0_0', which is not an entity of the model",
    ):
        _problem([], elements).initialize(SETTINGS)


def _rhs(network: Network, **kwargs: object) -> Hybridization:
    """Get the network in the right hand side, `gamma = net(prey, predator)`."""
    arguments: dict[str, object] = {
        "network": network,
        "pattern": RHS,
        "model": "lv",
        "inputs": {
            f"{network.sid}__input0__0": NetworkInput(formula="prey"),
            f"{network.sid}__input0__1": NetworkInput(formula="predator"),
        },
        "outputs": {f"{network.sid}__output0__0": "gamma"},
    }
    arguments.update(kwargs)
    return Hybridization(**arguments)  # ty: ignore[invalid-argument-type]


def test_a_compiled_model_with_other_values_than_the_network(
    tmp_path: Path,
) -> None:
    """A compiled model whose network was changed afterwards is refused.

    The frozen elements are evaluated by the model, so the model must carry
    the values of the network the hybridization describes.
    """
    network = feed_forward()
    compiled = compile_network(
        MODEL_PATH, [_rhs(network)], compiled_path(MODEL_PATH, tmp_path)
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    changed = replace(
        network, parameters=nominal_parameters(network, {"net1.layer1": 0.0})
    )
    elements = network_fit_parameters(
        changed, estimate={"net1.layer2": True}, bounds={}
    )
    frozen = set(network.parameter_ids()) - {p.pid for p in elements}
    problem = _problem([_rhs(changed, frozen=frozen)], elements, experiment=Compiled)
    problem.initialize(SETTINGS)
    with pytest.raises(
        ValueError,
        match=r"Network 'net1': the model carries other values of the frozen "
        r"elements \['net1__layer1__bias__0', .*compile the network again",
    ):
        problem.residuals(np.asarray(problem.x0, dtype=float))

    # the network which was compiled runs
    same = _problem([_rhs(network, frozen=frozen)], elements, experiment=Compiled)
    same.initialize(SETTINGS)
    residuals = np.asarray(
        same.residuals(np.asarray(same.x0, dtype=float)), dtype=float
    )
    assert np.all(np.isfinite(residuals))


def test_a_compiled_network_with_a_model_without_it() -> None:
    """A model which does not carry the compiled network is refused."""
    network = feed_forward()
    frozen = set(network.parameter_ids())
    problem = _problem([_rhs(network, frozen=frozen)], [])
    with pytest.raises(
        ValueError,
        match=r"reads \['net1__layer1__bias__0', .*which are neither entities of "
        r"the model",
    ):
        problem.initialize(SETTINGS)


def test_an_array_which_all_conditions_share(tmp_path: Path) -> None:
    """An array of all conditions is a part of the compiled model."""
    network = two_inputs()
    hybridization = Hybridization(
        network=network,
        pattern=RHS,
        model="lv",
        inputs={
            "net6__input0__0": NetworkInput(formula="prey"),
            "net6__input1": NetworkInput(arrays={ALL_CONDITIONS: [1.0, 2.0, 3.0]}),
        },
        outputs={"net6__output0__0": "gamma"},
    )
    compiled = compile_network(
        MODEL_PATH, [hybridization], compiled_path(MODEL_PATH, tmp_path)
    )

    class Compiled(LotkaVolterra):
        model_path: ClassVar[Path] = compiled

    problem = _problem([hybridization], [], experiment=Compiled)
    problem.initialize(SETTINGS)
    assert [
        [(hook, set(values)) for hook, values in derived]
        for derived in problem.group_derived
    ] == [[(hybridization, {"net6__output0__0"})]] * 2
    problem.predictions(np.asarray(problem.x0, dtype=float))
    assert "net6__input1__0" not in _changes(problem, 0)
