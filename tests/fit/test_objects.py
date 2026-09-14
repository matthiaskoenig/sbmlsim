"""Test fit objects."""

from sbmlsim.fit import FitParameter


def test_fit_parameter() -> None:
    """Test FitParameter creation."""
    p = FitParameter(
        pid="p1",
        start_value=1.0,
        lower_bound=1e-2,
        upper_bound=1e2,
        unit="dimensionless",
    )
    assert p.pid == "p1"
    assert p.start_value == 1.0
    assert p.lower_bound == 1e-2
    assert p.upper_bound == 1e2
    assert p.unit == "dimensionless"


def test_fit_parameter_serialization() -> None:
    """Test FitParameter serialization."""
    p = FitParameter(
        pid="p1",
        start_value=1.0,
        lower_bound=1e-2,
        upper_bound=1e2,
        unit="dimensionless",
    )

    p_str = p.to_json()

    q = FitParameter.from_json(json_info=p_str)
    assert p.pid == q.pid
    assert p.start_value == q.start_value
    assert p.lower_bound == q.lower_bound
    assert p.upper_bound == q.upper_bound
    assert p.unit == q.unit

    assert p == q


def test_a_parameter_is_its_own_target_by_default() -> None:
    """A parameter without a target writes the entity it is named after."""
    p = FitParameter("Ka_dis_hctz", 0.35, 0.01, 10.0, "1/hr")
    assert p.target is None
    assert p.target_id == "Ka_dis_hctz"
    assert not p.is_versioned


def test_a_versioned_parameter_writes_another_entity() -> None:
    """A version is estimated under its own id and written to the target."""

    def only_tablets(key: str, mapping: object) -> bool:
        return key.endswith("tablet")

    p = FitParameter(
        "Ka_dis_tablet",
        0.35,
        0.01,
        10.0,
        "1/hr",
        target="Ka_dis_hctz",
        mappings=only_tablets,
    )
    assert p.pid == "Ka_dis_tablet"
    assert p.target_id == "Ka_dis_hctz"
    assert p.is_versioned


def test_the_target_is_part_of_the_identity_of_a_parameter() -> None:
    """Two parameters which write different entities are not the same."""
    a = FitParameter("p", 1.0, 0.1, 10.0, "1/hr")
    b = FitParameter("p", 1.0, 0.1, 10.0, "1/hr", target="q")
    assert a != b


def test_the_target_is_serialized_and_the_selector_is_not() -> None:
    """A selector is a callable and cannot be written to JSON.

    The resolution it produces is what a PEtab problem stores, see the
    design; `to_dict` therefore carries the target and not the selector.
    """

    def every(key: str, mapping: object) -> bool:
        return True

    p = FitParameter("p", 1.0, 0.1, 10.0, "1/hr", target="q", mappings=every)
    d = p.to_dict()
    assert d["target"] == "q"
    assert "mappings" not in d
    # the round trip through JSON keeps the target
    assert FitParameter.from_json(p.to_json()).target_id == "q"
