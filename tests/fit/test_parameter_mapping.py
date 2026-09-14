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
from sbmlsim.fit.parameter_mapping import ParameterMapping, has_renamed_targets


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
    assert not has_renamed_targets(mapping.parameters)


def test_a_version_covers_the_groups_of_its_mappings() -> None:
    """A selector on mapping 2 reaches the group which holds it."""
    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Ka_c", target="Ka", versioned=True),
    ]
    mapping = ParameterMapping(parameters, {0: {0, 1}, 1: {2}}, GROUPS, KEYS)

    assert mapping.indices_for(0) == {"Ka": 0}
    assert mapping.indices_for(1) == {"Ka": 1}
    assert has_renamed_targets(mapping.parameters)


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


def test_a_selector_matching_nothing_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A versioned parameter which covers no simulation is a silent trap.

    It stays in the parameter vector and never changes the model, so the
    objective is flat in it -- a warning naming the parameter and its target
    is the only sign a user gets, short of reading the coverage table.
    """
    parameters = [_parameter("Ka_a", target="Ka", versioned=True)]
    with caplog.at_level("WARNING", logger="sbmlsim.fit.parameter_mapping"):
        mapping = ParameterMapping(parameters, {}, GROUPS, KEYS)

    (row,) = mapping.coverage()
    assert row.n_covered == 0
    assert any(
        "Ka_a" in record.getMessage() and "Ka" in record.getMessage()
        for record in caplog.records
    )


def test_a_selector_matching_something_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A version which covers at least one simulation is not a trap."""
    parameters = [_parameter("Ka_a", target="Ka", versioned=True)]
    with caplog.at_level("WARNING", logger="sbmlsim.fit.parameter_mapping"):
        ParameterMapping(parameters, {0: {0}}, GROUPS, KEYS)

    assert not caplog.records


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
    assert has_renamed_targets(mapping.parameters)
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
    assert not has_renamed_targets(mapping.parameters)
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


def test_an_unversioned_fit_is_deterministic(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Calling the cost twice for the same parameters gives the same number.

    This does not pin an absolute value -- `test_an_unversioned_fit_cost_is_pinned`
    does that -- it only guards that resolving the parameter mapping on every
    call to `residuals` did not introduce nondeterminism (e.g. through
    dict/set ordering).
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


def test_an_unversioned_fit_cost_is_pinned(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The cost of the HCTZ PK problem at the model values is what it was.

    This is the regression test of the whole feature: an unversioned problem
    must resolve to exactly the same changes it always did, so its cost must
    not move by orders of magnitude. The literal is a pin, not a derived
    fact: it was obtained by running this exact computation against the code
    on `develop` before Task 5 touched `residuals`/`_simulate_groups`. To
    regenerate it (only after a deliberate, reviewed change to the model,
    the data, or the fit settings), run:

        op = FIT_DEFINITIONS["PK"].problem(opid="hctz_pk")
        op.initialize(fit_settings)
        op.cost_least_square(op.to_scale(op.xmodel))

    The tolerance is `rel=1e-3`, not pytest's default `1e-6`: this exact
    problem is on record for needing headroom of that order across
    platforms, see `COST_RTOL` in `tests/fit/test_identifiability.py` -- a
    cost differs from the integrator by about its own relative tolerance
    (`1e-6` in `fit_settings`), and that differs between linux, macOS and
    windows. `1e-3` still separates a systematic mis-binding, which moves
    this cost by orders of magnitude, from integrator noise.
    """
    op_hctz_pk.initialize(fit_settings)
    x = op_hctz_pk.to_scale(op_hctz_pk.xmodel)

    assert op_hctz_pk.cost_least_square(x) == pytest.approx(23.96168054253333, rel=1e-3)


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
    """Two versions of one entity give two different simulations.

    `Ka_dis_hctz` is the dissolution rate of the oral tablet: the
    intravenous route bypasses dissolution, so its simulations do not depend
    on it. Swapping the two version values must move every oral prediction
    well beyond solver noise and leave every intravenous one at the level of
    solver noise -- a numeric fact that fails if `_simulate_groups` bound
    both versions into one shared dict (the "reach every mapping" and
    "bindings differ" checks in an earlier version of this test did not: they
    would not have failed even if no simulated value ever changed).
    """
    problem = _versioned_definition(definition_hctz_pk).problem(opid="versions")
    problem.initialize(fit_settings)

    mapping = problem.parameter_mapping
    assert mapping is not None
    group_of_mapping = {
        k: k_group
        for k_group, group in enumerate(problem.mapping_groups)
        for k in group
    }
    # Ka_po is pids[0], Ka_iv is pids[1]; group the fit mappings by which of
    # the two versions ParameterMapping actually bound them to
    oral = [
        k
        for k in range(len(problem.mapping_keys))
        if mapping.indices_for(group_of_mapping[k]).get("Ka_dis_hctz") == 0
    ]
    intravenous = [
        k
        for k in range(len(problem.mapping_keys))
        if mapping.indices_for(group_of_mapping[k]).get("Ka_dis_hctz") == 1
    ]
    assert oral and intravenous

    # the two versions with clearly different values, then swapped
    x_a = problem.to_scale(np.array([0.1, 5.0]))
    x_b = problem.to_scale(np.array([5.0, 0.1]))
    res_a = problem.residuals(x_a, complete_data=True)
    res_b = problem.residuals(x_b, complete_data=True)

    # every mapping was simulated
    assert len(res_a["y_obs"]) == len(problem.mapping_keys)

    def _relative_change(k: int) -> float:
        """Get the largest relative change of one mapping's interpolated curve."""
        a = np.asarray(res_a["y_obsip"][k])
        b = np.asarray(res_b["y_obsip"][k])
        scale = np.max(np.abs(a)) or 1.0
        return float(np.max(np.abs(a - b)) / scale)

    # swapping the versions must move the oral curves well beyond solver noise
    assert max(_relative_change(k) for k in oral) > 0.1
    # and must leave the intravenous curves at the level of solver noise,
    # since they do not depend on Ka_dis_hctz at all
    assert max(_relative_change(k) for k in intravenous) < 1e-3


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


def test_two_groups_sharing_a_simulation_object_must_bind_alike() -> None:
    """A `TimecourseSim` shared by two groups must not carry disagreeing changes.

    `_group_mappings` keys a group on `(id(model), id(simulation))`, not on
    the simulation object alone: `Task` allows one `simulation_id` to be
    combined with several `model_id`s to "execute the same simulation with
    different model variants" (`sbmlsim.task.task.Task`), which puts the same
    `TimecourseSim` object into two distinct groups. The fit path takes
    `sim_experiment._simulations[task.simulation_id]` directly, with no
    `deepcopy` (unlike the experiment path), so `_simulate_groups` mutating
    that object's `changes` in place would leak a change bound in one group
    into the other, silently, because `dict.update` never removes a key.
    `_check_shared_simulation_bindings` refuses this at `initialize` instead.
    """
    problem = OptimizationProblem.__new__(OptimizationProblem)
    problem.opid = "shared-simulation"

    parameters = [
        _parameter("Ka_a", target="Ka", versioned=True),
        _parameter("Ka_b", target="Ka", versioned=True),
    ]
    shared_simulation = object()
    # two model variants share one TimecourseSim object: mapping 0 (its own
    # group) is bound by Ka_a and mapping 1 (a different group, same object)
    # by Ka_b -- the pathological shape `_group_mappings` allows
    problem.simulations = [shared_simulation, shared_simulation, object()]
    problem.mapping_groups = [[0], [1], [2]]
    problem.parameter_mapping = ParameterMapping(
        parameters, {0: {0}, 1: {1}}, problem.mapping_groups, KEYS
    )

    with pytest.raises(ValueError, match="share one"):
        problem._check_shared_simulation_bindings()
