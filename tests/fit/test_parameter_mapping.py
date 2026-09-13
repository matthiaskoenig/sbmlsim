"""Tests of the binding of the parameters to the simulations."""

import dataclasses
import pickle

import numpy as np
import pytest

from conftest import is_intravenous, is_oral  # ty: ignore[unresolved-import]
from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.objects import FitParameter, MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.parameter_mapping import ParameterMapping


def _parameter(
    pid: str, target: str | None = None, versioned: bool = False
) -> FitParameter:
    """Build a parameter, versioned when it selects a part of the data."""

    def _select(key: str, mapping: object) -> bool:
        return True

    return FitParameter(
        pid,
        1.0,
        0.1,
        10.0,
        "1/hr",
        target=target,
        mappings=_select if versioned else None,
    )


#: three mappings in two groups: mappings 0 and 1 share a simulation
GROUPS = [[0, 1], [2]]
KEYS = ["fm_a", "fm_b", "fm_c"]


def test_a_parameter_without_a_selector_covers_every_group() -> None:
    """An ordinary parameter is applied everywhere, as it is today."""
    parameters = [_parameter("Ka")]
    mapping = ParameterMapping(parameters, {}, GROUPS, KEYS)

    assert mapping.indices_for(0) == {"Ka": 0}
    assert mapping.indices_for(1) == {"Ka": 0}
    assert not mapping.is_versioned


def test_a_version_covers_the_groups_of_its_mappings() -> None:
    """A selector on mapping 2 reaches the group which holds it."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Ka_c", target="Ka", versioned=True),
    ]
    mapping = ParameterMapping(parameters, {0: {0, 1}, 1: {2}}, GROUPS, KEYS)

    assert mapping.indices_for(0) == {"Ka": 0}
    assert mapping.indices_for(1) == {"Ka": 1}
    assert mapping.is_versioned


def test_a_group_no_version_covers_gets_no_change() -> None:
    """An uncovered simulation keeps the value of the model."""
    parameters = [_parameter("Ka_a", target="Ka", versioned=True)]
    mapping = ParameterMapping(parameters, {0: {0, 1}}, GROUPS, KEYS)

    assert mapping.indices_for(0) == {"Ka": 0}
    assert mapping.indices_for(1) == {}


def test_two_versions_on_one_mapping_are_an_error() -> None:
    """A simulation cannot have two values for one entity."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Ka_b", target="Ka", versioned=True),
    ]
    with pytest.raises(ValueError, match=r"Ka_a.*Ka_b|Ka_b.*Ka_a"):
        ParameterMapping(parameters, {0: {0}, 1: {0}}, GROUPS, KEYS)


def test_a_parameter_without_a_selector_overlaps_every_version() -> None:
    """There is no fallback: a target is global or it is versioned."""
    parameters = [
        _parameter("Ka"),
        _parameter("Ka_a", target="Ka", versioned=True),
    ]
    with pytest.raises(ValueError, match="Ka"):
        ParameterMapping(parameters, {1: {0}}, GROUPS, KEYS)


def test_a_selector_must_not_split_a_simulation() -> None:
    """Mappings 0 and 1 share a simulation, so they share the value."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Ka_b", target="Ka", versioned=True),
    ]
    with pytest.raises(ValueError, match="one simulation"):
        ParameterMapping(parameters, {0: {0}, 1: {1}}, GROUPS, KEYS)


def test_the_versions_of_a_target_agree_on_the_unit() -> None:
    """The unit is how the value reaches the model, so it is one unit."""
    a = FitParameter(
        "Ka_a", 1.0, 0.1, 10.0, "1/hr", target="Ka", mappings=lambda k, m: True
    )
    b = FitParameter(
        "Ka_b", 1.0, 0.1, 10.0, "1/min", target="Ka", mappings=lambda k, m: True
    )
    with pytest.raises(ValueError, match="unit"):
        ParameterMapping([a, b], {0: {0, 1}, 1: {2}}, GROUPS, KEYS)


def test_changes_are_the_quantities_of_the_bound_parameters() -> None:
    """`changes_for` is what a simulation of a group is run with."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Vd"),
    ]
    mapping = ParameterMapping(parameters, {0: {0, 1}}, GROUPS, KEYS)
    quantities = ["Q_Ka_a", "Q_Vd"]

    assert mapping.changes_for(0, quantities) == {"Ka": "Q_Ka_a", "Vd": "Q_Vd"}
    assert mapping.changes_for(1, quantities) == {"Vd": "Q_Vd"}


def test_the_coverage_names_the_groups_a_version_does_not_reach() -> None:
    """The report says which simulations keep the value of the model."""
    parameters = [_parameter("Ka_a", target="Ka", versioned=True)]
    mapping = ParameterMapping(
        parameters, {0: {0, 1}}, GROUPS, KEYS, group_names=["po_tablet", "iv"]
    )

    (row,) = mapping.coverage()
    assert row.pid == "Ka_a"
    assert row.target == "Ka"
    assert row.n_covered == 1
    assert row.n_groups == 2
    assert row.uncovered_groups == ["iv"]


def test_the_problem_resolves_its_selectors(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """A versioned problem knows which simulation gets which parameter."""
    # `definition_hctz_pk` is the shared `FIT_DEFINITIONS["PK"]`, so a copy is
    # built here rather than mutating it in place, which would leak the
    # custom parameters into the other tests that use the same definition
    definition = dataclasses.replace(definition_hctz_pk)
    definition.parameters = [
        FitParameter(
            "Ka_po",
            0.35,
            0.01,
            10.0,
            "1/hr",
            target="Ka_dis_hctz",
            mappings=is_oral,
        ),
    ]
    problem = definition.problem(opid="versions")
    problem.initialize(fit_settings)

    mapping = problem.parameter_mapping
    assert mapping is not None
    assert mapping.is_versioned
    (row,) = mapping.coverage()
    assert row.pid == "Ka_po"
    assert row.target == "Ka_dis_hctz"
    # some simulations are oral and some are not
    assert 0 < row.n_covered < row.n_groups
    assert row.uncovered_groups


def test_an_unversioned_problem_binds_every_parameter_everywhere(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Nothing changes for a problem which has no versions."""
    op_hctz_pk.initialize(fit_settings)
    mapping = op_hctz_pk.parameter_mapping

    assert mapping is not None
    assert not mapping.is_versioned
    for group in range(len(op_hctz_pk.mapping_groups)):
        assert set(mapping.indices_for(group)) == set(op_hctz_pk.pids)


def test_a_versioned_problem_is_picklable(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """The workers of a parallel fit unpickle the definition of a problem.

    A selector is a callable, so it must be a module level function; a lambda
    would make a parallel fit fail when the workers start.
    """
    # see `test_the_problem_resolves_its_selectors` for why this is a copy
    definition = dataclasses.replace(definition_hctz_pk)
    definition.parameters = [
        FitParameter(
            "Ka_po",
            0.35,
            0.01,
            10.0,
            "1/hr",
            target="Ka_dis_hctz",
            mappings=is_oral,
        ),
    ]
    problem = definition.problem(opid="versions")
    problem.initialize(fit_settings)

    restored = pickle.loads(pickle.dumps(problem))
    assert restored.parameters[0].target_id == "Ka_dis_hctz"
    assert restored.parameters[0].mappings is is_oral


def test_an_unversioned_fit_is_unchanged(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The cost of a problem without versions is what it was.

    This is the regression test of the whole feature: the resolution must not
    move a single digit of an ordinary fit.
    """
    op_hctz_pk.initialize(fit_settings)
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)

    assert op_hctz_pk.cost_least_square(x) == pytest.approx(
        op_hctz_pk.cost_least_square(x)
    )
    # the residuals cover the training data and nothing else
    assert len(op_hctz_pk.residuals(x)) == sum(
        len(op_hctz_pk.y_references[k]) for k in op_hctz_pk.training_indices
    )


def _versioned_definition(definition: FitDefinition) -> FitDefinition:
    """Build a definition where `Ka_dis_hctz` is estimated once per route.

    A copy of the shared definition is built through `dataclasses.replace`
    rather than mutating `definition.parameters` in place, which would leak
    the custom parameters into the other tests sharing the same fixture.
    """
    return dataclasses.replace(
        definition,
        parameters=[
            FitParameter(
                "Ka_po",
                0.35,
                0.01,
                10.0,
                "1/hr",
                target="Ka_dis_hctz",
                mappings=is_oral,
            ),
            FitParameter(
                "Ka_iv",
                0.35,
                0.01,
                10.0,
                "1/hr",
                target="Ka_dis_hctz",
                mappings=is_intravenous,
            ),
        ],
    )


def test_the_versions_reach_their_own_simulations(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """Two versions of one entity give two different simulations."""
    problem = _versioned_definition(definition_hctz_pk).problem(opid="versions")
    problem.initialize(fit_settings)

    # the two versions with clearly different values
    x = problem.to_scale(np.array([0.1, 5.0]))
    res_data = problem.residuals(x, complete_data=True)

    # every mapping was simulated and the simulations are not all the same
    assert len(res_data["y_obs"]) == len(problem.mapping_keys)
    mapping = problem.parameter_mapping
    assert mapping is not None
    bound = {
        tuple(sorted(mapping.indices_for(g).items()))
        for g in range(len(problem.mapping_groups))
    }
    assert len(bound) > 1, "the two versions must not resolve to the same binding"


def test_a_version_counts_as_a_parameter_everywhere(
    definition_hctz_pk: FitDefinition, fit_settings: FitSettings
) -> None:
    """The metrics charge for both versions and the profiles cover both."""
    from sbmlsim.fit.identifiability import ProfileSettings, profile_likelihood
    from sbmlsim.fit.metrics import FitMetrics

    problem = _versioned_definition(definition_hctz_pk).problem(opid="versions")
    problem.initialize(fit_settings)

    metrics = FitMetrics(problem=problem, parameter_set=problem.parameter_set_model())
    # both versions are fitted parameters, so both are charged for
    assert metrics.n_parameters == 2
    assert metrics.summary(kind=MappingKind.TRAINING)["k"] == 2

    result = profile_likelihood(
        problem=problem,
        settings=fit_settings,
        parameter_set=problem.parameter_set_model(),
        profile_settings=ProfileSettings(
            reoptimize=False,
            initial_step=0.5,
            min_step=0.1,
            max_step=2.0,
            max_points=3,
        ),
        serial=True,
        show_progress=False,
    )
    assert set(result.profiles) == {"Ka_po", "Ka_iv"}
