"""Tests of the derived changes of a problem, with a hook of the tests.

The hook of `hooks.py` scales an entity of the model by a parameter of the fit
which is not an entity of the model, i.e. what a network before the
simulation does without a network.
"""

import json
import logging
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pytest

from sbmlsim.fit import FitParameter, FitSettings
from sbmlsim.fit.cli import FitDefinition, run_fit
from sbmlsim.fit.derived import DerivedChanges
from sbmlsim.fit.objects import EXTERNAL_PREFIX
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from tests.fit.hooks import FACTOR, TARGET, Scaling
from tests.fit.hooks import factor_parameter as _factor


def _preinit(problem: OptimizationProblem, k_group: int, x: np.ndarray) -> dict:
    """Get the pre-initialization values of a group for the parameters."""
    plan = problem.evaluated_plan(k_group, x)
    return {a.target: a.value for a in plan.preinit}


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
    changes = _preinit(problem, 0, np.array([2.0]))
    assert changes[TARGET] == pytest.approx(2.0 * nominal)

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
    changes = _preinit(problem, 0, np.array([1e-4]))
    # the values of the fit reach the hook in the units of the model
    factor = problem.runner_initialized.Q_(1e-4, "1/ml").to(model.uinfo[TARGET])
    assert changes["Ka_dis_hctz"] == pytest.approx(factor.magnitude * nominal)
    # and the value of the model does not change between the evaluations
    problem.predictions(np.array([1e-4]))
    changes = _preinit(problem, 0, np.array([1e-4]))
    assert changes["Ka_dis_hctz"] == pytest.approx(factor.magnitude * nominal)


def test_a_condition_the_problem_does_not_simulate_is_logged(
    definition_hctz_iv: FitDefinition,
    fit_settings: FitSettings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A typo in the key of a condition would fall back to the other values."""
    scaling = Scaling(per_condition={"input0": ["hctz_iv1", "hctz_iv_35"]})
    problem = _problem(definition_hctz_iv, [_factor()], hybridizations=[scaling])
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.derived"):
        problem.initialize(fit_settings)
    (record,) = caplog.records
    message = record.getMessage()
    assert "'input0' of 'scaling'" in message
    assert "['hctz_iv_35']" in message
    assert "['hctz_iv1', 'hctz_iv35']" in message

    caplog.clear()
    known = Scaling(per_condition={"input0": ["hctz_iv1", "hctz_iv35"]})
    problem = _problem(definition_hctz_iv, [_factor()], hybridizations=[known])
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.derived"):
        problem.initialize(fit_settings)
    assert not caplog.records


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
        # a constant the fit estimates: the value of the fit would win silently
        (
            [_factor()],
            [Scaling(constants={FACTOR: 3.0})],
            r"reads 'factor_k' \(FitParameter 'factor_k'\) as a constant of "
            r"the hybridization and as a parameter of the fit",
        ),
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
        (
            [_factor()],
            [Scaling(target="nothing")],
            r"Beermann1976\|fm_hctz_iv1_5_feces.*sets 'nothing', which is not an entity",
        ),
        # a misspelled symbol is neither an entity nor a parameter
        (
            [_factor(), FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml")],
            [Scaling(target="Ka_dis_hctz", factor="factr_k")],
            r"Beermann1976\|fm_hctz_iv1_5_feces.*reads \['factr_k'\], which",
        ),
        # the check of the problem, the hook does not freeze the target
        (
            [_factor(), FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml")],
            [Scaling()],
            r"sets 'KI__HCTZEX_k', which the parameters \['KI__HCTZEX_k'\] of the fit",
        ),
        # the first timecourse changes the dose, the hook would overwrite it
        (
            [_factor()],
            [Scaling(target="IVDOSE_hctz")],
            r"Beermann1976\|fm_hctz_iv1_5_feces.*sets 'IVDOSE_hctz', which the "
            r"first timecourse",
        ),
        # an external parameter and an entity reach the hook with one id
        (
            [
                FitParameter(
                    "ext",
                    1.0,
                    0.0,
                    np.inf,
                    unit="dimensionless",
                    target=f"{EXTERNAL_PREFIX}{TARGET}",
                    scale=ParameterScaleType.LINEAR,
                ),
                FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml"),
            ],
            [Scaling(target="Ka_dis_hctz", factor=TARGET)],
            r"'ext' \(target 'sciml:KI__HCTZEX_k'\) and 'KI__HCTZEX_k' "
            r"\(target 'KI__HCTZEX_k'\)",
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


def _is_iv35(key: str, mapping: object) -> bool:
    """Select the fit mappings of the dose of 35 mg."""
    return "iv35" in key


def test_a_symbol_a_simulation_does_not_have(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """An external parameter versioned away from a simulation is not read there."""
    factor = FitParameter(
        FACTOR,
        1.0,
        0.0,
        np.inf,
        unit="dimensionless",
        target=f"{EXTERNAL_PREFIX}{FACTOR}",
        scale=ParameterScaleType.LINEAR,
        mappings=_is_iv35,
    )
    problem = _problem(definition_hctz_iv, [factor], hybridizations=[Scaling()])
    with pytest.raises(
        ValueError, match=r"'Beermann1976\|fm_hctz_iv1_5_feces'.*\['factor_k'\]"
    ):
        problem.initialize(fit_settings)


def test_a_hook_reads_its_constants(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A symbol which is neither an entity nor a parameter is a constant."""
    problem = _problem(
        definition_hctz_iv,
        [FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml")],
        hybridizations=[
            Scaling(target="Ka_dis_hctz", factor="c", constants={"c": 3.0})
        ],
    )
    problem.initialize(fit_settings)
    nominal = float(problem.models[0].r["Ka_dis_hctz"])
    problem.predictions(np.array([1e-4]))
    changes = _preinit(problem, 0, np.array([1e-4]))
    assert changes["Ka_dis_hctz"] == pytest.approx(3.0 * nominal)


def test_a_change_of_the_simulation_has_precedence_over_the_model(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """The hook reads the dose of the simulation, not the dose of the model."""
    problem = _problem(
        definition_hctz_iv,
        [FitParameter(TARGET, 1e-4, 1e-10, 1.0, unit="1/ml")],
        hybridizations=[Scaling(target="Ka_dis_hctz", factor="IVDOSE_hctz")],
    )
    problem.initialize(fit_settings)
    model = problem.models[0]
    nominal = float(model.r["Ka_dis_hctz"])
    assert float(model.r["IVDOSE_hctz"]) == 0.0
    problem.predictions(np.array([1e-4]))
    Q_ = problem.runner_initialized.Q_
    for k, dose in ((0, 1.0), (1, 35.0)):
        value = Q_(dose, "mg").to(model.uinfo["IVDOSE_hctz"]).magnitude
        changes = _preinit(problem, k, np.array([1e-4]))
        assert changes["Ka_dis_hctz"] == pytest.approx(value * nominal)


@dataclass(frozen=True)
class Extra(Scaling):
    """A hook which answers with a change it does not name as a target."""

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        return {**super().derived_changes(values, condition), "Ka_dis_hctz": 1.0}


def test_a_hook_which_sets_more_than_its_targets(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """The changes of a hook are its targets."""
    problem = _problem(definition_hctz_iv, [_factor()], hybridizations=[Extra()])
    problem.initialize(fit_settings)
    with pytest.raises(
        ValueError,
        match=r"'Beermann1976\|fm_hctz_iv1_5_feces'.*\['Ka_dis_hctz'\].*not its "
        r"targets\.$",
    ):
        problem.predictions(np.array([1.0]))


@dataclass(frozen=True)
class Missing(Scaling):
    """A hook which answers without its target."""

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        return {}


def test_a_hook_which_sets_less_than_its_targets(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """The message names what is missing and nothing else."""
    problem = _problem(definition_hctz_iv, [_factor()], hybridizations=[Missing()])
    problem.initialize(fit_settings)
    with pytest.raises(
        ValueError, match=r"answers without its targets \['KI__HCTZEX_k'\]\.$"
    ):
        problem.predictions(np.array([1.0]))


def test_a_problem_with_hooks_is_a_dict(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """The JSON of a problem records its hooks."""
    problem = _problem(definition_hctz_iv, [_factor()], hybridizations=[Scaling()])
    assert problem.to_dict()["hybridizations"] == [
        {
            "type": "Scaling",
            "model": "model",
            "targets": [TARGET],
            "summary": {
                "name": "scaling",
                "kind": "scaling",
                "description": f"{TARGET} = {FACTOR} * {TARGET}",
                "arrays": [],
            },
        }
    ]
    assert json.loads(str(problem.to_json()))["hybridizations"][0]["model"] == "model"


def test_a_parallel_fit_with_hooks(definition_hctz_iv: FitDefinition) -> None:
    """The workers unpickle the hooks and fit what the serial fit fits.

    The residuals do not depend on earlier evaluations, so the runs agree
    with the serial run up to the first load of the model by roadrunner.
    """
    definition = replace(
        definition_hctz_iv, parameters=[_factor(2.0)], hybridizations=[Scaling()]
    )
    serial = run_fit(definition=definition, opid="serial", size=1, n_cores=1, seed=1)
    parallel = run_fit(
        definition=definition, opid="parallel", size=2, n_cores=2, seed=1
    )
    result, expected = parallel["parallel"].result, serial["serial"].result
    assert result.size == 2
    # the factor moved away from its start value in the workers
    assert abs(result.xopt[0] - 2.0) > 0.5
    np.testing.assert_allclose(result.xopt, expected.xopt, rtol=1e-8)
    np.testing.assert_allclose(
        result.df_fits["cost"], expected.df_fits["cost"].iloc[0], rtol=1e-8
    )


def test_an_external_target_names_something() -> None:
    """The prefix alone is not a target."""
    with pytest.raises(ValueError, match="names nothing"):
        FitParameter("x", 1.0, unit="dimensionless", target=EXTERNAL_PREFIX)
    parameter = FitParameter("x", 1.0, unit="dimensionless", target="sciml:x")
    assert parameter.is_external
    assert parameter.entity_id == "x"
    assert not FitParameter("x", 1.0, unit="dimensionless").is_external


def test_the_summary_of_a_hook_and_the_groups() -> None:
    from sbmlsim.fit.derived import (
        HookSummary,
        ParameterGroup,
        describe,
        group_parameters,
        hook_summaries,
    )

    scaling = Scaling()
    (summary,) = hook_summaries([scaling])
    assert summary == HookSummary(
        name="scaling",
        kind="scaling",
        description=f"{TARGET} = {FACTOR} * {TARGET}",
        targets=(TARGET,),
        groups=(),
    )
    a, b, c = FitParameter("a", 1.0), FitParameter("b", 2.0), FitParameter("c", 3.0)
    grouped = HookSummary(
        "net",
        "rhs",
        "layer1 (Linear)",
        ("x",),
        (ParameterGroup("net.l.w", ("b", "z")),),
    )
    single, groups = group_parameters([a, b, c], [summary, grouped])
    assert single == [a, c]
    assert groups == [(ParameterGroup("net.l.w", ("b", "z")), [b])]
    assert describe(scaling)["summary"]["name"] == "scaling"
