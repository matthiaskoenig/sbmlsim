from sbmlsim.fit import FitParameter


def test_paramter():
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


def test_serialization():
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
