"""Tests of the binding of the parameters to the simulations."""

import pytest

from sbmlsim.fit.objects import FitParameter
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
