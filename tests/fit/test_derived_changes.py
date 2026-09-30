"""Tests of the derived changes of a problem, with a hook of the tests.

The hook scales an entity of the model by a parameter of the fit which is
not an entity of the model, i.e. what a network before the simulation does
without a network.
"""

import pickle
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest

from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.derived import DerivedChanges
from sbmlsim.fit.objects import EXTERNAL_PREFIX
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType

#: the entity of the hctz model the hook writes
TARGET = "KI__HCTZEX_k"

#: the parameter of the fit the hook reads, which is not an entity
FACTOR = "factor_k"


@dataclass(frozen=True)
class Scaling:
    """The change `target = factor * value of the model`.

    Attributes:
        model: id of the model in the experiment.
        target: the entity which is set.
        factor: the parameter of the fit which is read.
        frozen: ids the fit must not write.
        calls: the conditions the changes were calculated for.
    """

    model: str = "model"
    target: str = TARGET
    factor: str = FACTOR
    frozen: frozenset[str] = frozenset()
    calls: list[str] = field(default_factory=list, compare=False)

    def symbols(self) -> Collection[str]:
        return {self.factor, self.target}

    def targets(self) -> Collection[str]:
        return {self.target}

    def check_parameters(self, targets: Collection[str]) -> None:
        frozen = sorted(set(targets) & self.frozen)
        if frozen:
            raise ValueError(f"the parameters write {frozen}, which are frozen")

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        self.calls.append(condition)
        return {self.target: values[self.factor] * values.get(self.target, 1.0)}


def _factor(start: float = 1.0) -> FitParameter:
    return FitParameter(
        FACTOR,
        start,
        lower_bound=0.0,
        upper_bound=np.inf,
        unit="dimensionless",
        target=f"{EXTERNAL_PREFIX}{FACTOR}",
        scale=ParameterScaleType.LINEAR,
    )


def _problem(
    definition: FitDefinition, parameters: list[FitParameter], **kwargs: object
) -> OptimizationProblem:
    """Build the iv problem with the parameters and the hybridizations."""
    return OptimizationProblem(
        opid="derived",
        mapping_collections=definition.collections(),
        fit_parameters=parameters,
        base_path=definition.base_path,
        data_path=definition.data_path,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def test_the_hook_is_the_protocol() -> None:
    """A hybridization is what provides the derived changes."""
    assert isinstance(Scaling(), DerivedChanges)
    assert not isinstance(object(), DerivedChanges)
    with pytest.raises(TypeError, match="does not provide"):
        OptimizationProblem(
            "x",
            [],
            [_factor()],
            hybridizations=[object()],  # ty: ignore[invalid-argument-type]
        )


def test_an_external_parameter_writes_no_change_and_the_hook_reads_it(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """The prediction with the factor is the prediction with the scaled entity."""
    scaling = Scaling()
    problem = _problem(definition_hctz_iv, [_factor(2.0)], hybridizations=[scaling])
    problem.initialize(fit_settings)
    assert problem.pids == [FACTOR]
    assert problem.parameter_mapping_initialized.changes_for(0, [1.0]) == {}
    nominal = float(problem.models[0].r[TARGET])
    assert problem.xmodel[0] == 2.0

    predictions = problem.predictions(np.array([2.0]))
    # once per simulation, with the id of the simulation
    assert scaling.calls == ["hctz_iv1", "hctz_iv35"]
    # the target of the hook was written, its value follows from the factor
    changes = problem.simulations[0].timecourses[0].changes
    assert changes[TARGET].magnitude == pytest.approx(2.0 * nominal)

    plain = _problem(
        definition_hctz_iv,
        [FitParameter(TARGET, 2.0 * nominal, 1e-10, 1.0, unit="1/ml")],
    )
    plain.initialize(fit_settings)
    expected = plain.predictions(np.array([2.0 * nominal]))
    for k, values in predictions.items():
        np.testing.assert_allclose(values, expected[k], rtol=1e-10)
    # the cost depends on the factor
    assert problem.cost_least_square(np.array([2.0])) != pytest.approx(
        problem.cost_least_square(np.array([1.0]))
    )


def test_the_hook_reads_the_changes_of_the_fit(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A value of the fit has precedence over the value of the model."""
    problem = _problem(
        definition_hctz_iv,
        [FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml")],
        hybridizations=[Scaling(target="Ka_dis_hctz", factor=TARGET)],
    )
    problem.initialize(fit_settings)
    model = problem.models[0]
    # the value of the model as it was loaded: a simulation changes the state
    nominal = float(model.r["Ka_dis_hctz"])
    problem.predictions(np.array([1e-4]))
    assert float(model.r["Ka_dis_hctz"]) != nominal
    changes = problem.simulations[0].timecourses[0].changes
    # the values of the fit reach the hook in the units of the model
    factor = problem.runner_initialized.Q_(1e-4, "1/ml").to(model.uinfo[TARGET])
    assert changes["Ka_dis_hctz"].magnitude == pytest.approx(factor.magnitude * nominal)
    # and the value of the model does not change between the evaluations
    problem.predictions(np.array([1e-4]))
    assert changes["Ka_dis_hctz"].magnitude == pytest.approx(factor.magnitude * nominal)


def test_a_problem_with_hooks_is_pickled(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """The workers of a parallel fit get the definition with its hooks."""
    problem = _problem(definition_hctz_iv, [_factor(2.0)], hybridizations=[Scaling()])
    problem.initialize(fit_settings)
    restored = pickle.loads(pickle.dumps(problem))
    assert not restored.is_initialized
    assert restored.hybridizations == [Scaling()]
    restored.initialize(fit_settings)
    np.testing.assert_allclose(
        np.asarray(restored.residuals(restored.to_scale([2.0])), dtype=float),
        np.asarray(problem.residuals(problem.to_scale([2.0])), dtype=float),
    )


def test_a_definition_carries_its_hooks(definition_hctz_iv: FitDefinition) -> None:
    """A `FitDefinition` builds the problem with the hybridizations."""
    from dataclasses import replace

    definition = replace(
        definition_hctz_iv, parameters=[_factor()], hybridizations=[Scaling()]
    )
    assert definition.problem("hooked").hybridizations == [Scaling()]
    assert definition_hctz_iv.problem("plain").hybridizations == []


@pytest.mark.parametrize(
    ("parameters", "hybridizations", "message"),
    [
        ([_factor()], [], r"'factor_k' writes 'sciml:factor_k'.*no hybridization"),
        ([_factor()], [Scaling(model="other")], r"name the models \['other'\]"),
        (
            [_factor(), FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml")],
            [Scaling(frozen=frozenset({TARGET}))],
            "which are frozen",
        ),
        (
            [_factor()],
            [Scaling(), Scaling(target="Ka_dis_hctz", factor=TARGET)],
            r"reads \['KI__HCTZEX_k'\], which a hybridization sets",
        ),
        ([_factor()], [Scaling(), Scaling()], "two hybridizations.*set 'KI__HCTZEX_k'"),
        (
            [
                FitParameter(
                    FACTOR,
                    None,
                    0.0,
                    1.0,
                    unit="dimensionless",
                    target="sciml:x",
                    scale=ParameterScaleType.LINEAR,
                )
            ],
            [Scaling(factor="x")],
            "requires a 'start_value'",
        ),
    ],
)
def test_hooks_and_parameters_which_do_not_fit(
    definition_hctz_iv: FitDefinition,
    fit_settings: FitSettings,
    parameters: list[FitParameter],
    hybridizations: list[Scaling],
    message: str,
) -> None:
    """The problem refuses what would make the objective flat or wrong."""
    problem = _problem(definition_hctz_iv, parameters, hybridizations=hybridizations)
    with pytest.raises(ValueError, match=message):
        problem.initialize(fit_settings)


def test_a_hook_which_sets_no_entity(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A change of something which is not in the model is an error."""
    problem = _problem(
        definition_hctz_iv, [_factor()], hybridizations=[Scaling(target="nothing")]
    )
    problem.initialize(fit_settings)
    with pytest.raises(ValueError, match="sets 'nothing', which is not an entity"):
        problem.predictions(np.array([1.0]))


def test_an_external_target_names_something() -> None:
    """The prefix alone is not a target."""
    with pytest.raises(ValueError, match="names nothing"):
        FitParameter("x", 1.0, unit="dimensionless", target=EXTERNAL_PREFIX)
    parameter = FitParameter("x", 1.0, unit="dimensionless", target="sciml:x")
    assert parameter.is_external
    assert parameter.entity_id == "x"
    assert not FitParameter("x", 1.0, unit="dimensionless").is_external
