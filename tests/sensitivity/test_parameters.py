import math

import pandas as pd
import pytest

from sbmlsim.sensitivity import ParameterType, SensitivityParameter

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def simple_parameters():
    return [
        SensitivityParameter(
            uid="k1",
            name="rate constant 1",
            value=1.0,
            type=ParameterType.NA,
        ),
        SensitivityParameter(
            uid="k2",
            name="rate constant 2",
            value=2.0,
            type=ParameterType.NA,
        ),
    ]


# -----------------------------------------------------------------------------
# Basic model behavior
# -----------------------------------------------------------------------------


def test_parameter_type_enum_values():
    assert ParameterType.DATA.value == "data"
    assert ParameterType.SCALING.value == "scaling"
    assert ParameterType.NA.value == "na"
    assert ParameterType.FIT.value == "fitted"


def test_sensitivity_parameter_defaults():
    p = SensitivityParameter(uid="p1", name="param")

    assert p.uid == "p1"
    assert p.name == "param"
    assert math.isnan(p.value)
    assert math.isnan(p.lower_bound)
    assert math.isnan(p.upper_bound)
    assert p.unit is None
    assert p.type == ParameterType.NA
    assert p.reference == ""


def test_hash_is_based_on_uid():
    p1 = SensitivityParameter(uid="x", name="a")
    p2 = SensitivityParameter(uid="x", name="b")
    p3 = SensitivityParameter(uid="y", name="a")

    assert hash(p1) == hash(p2)
    assert hash(p1) != hash(p3)


# -----------------------------------------------------------------------------
# Bounds handling
# -----------------------------------------------------------------------------


def test_parameters_set_bounds(simple_parameters):
    bounds = [
        ("k1", 0.1, 10.0, ParameterType.FIT),
        ("k2", 1.0, 5.0, ParameterType.SCALING),
    ]

    SensitivityParameter.parameters_set_bounds(simple_parameters, bounds)

    p1, p2 = simple_parameters

    assert p1.lower_bound == 0.1
    assert p1.upper_bound == 10.0
    assert p1.type == ParameterType.FIT

    assert p2.lower_bound == 1.0
    assert p2.upper_bound == 5.0
    assert p2.type == ParameterType.SCALING


def test_parameters_to_df(simple_parameters):
    df = SensitivityParameter.parameters_to_df(simple_parameters, sort=False)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "uid",
        "name",
        "value",
        "lower_bound",
        "upper_bound",
        "unit",
        "type",
        "reference",
    }

    assert len(df) == 2
    assert df.loc[0, "type"] == "na"


def test_parameter_to_latex_creates_file(tmp_path):
    params = [
        SensitivityParameter(uid="k_cat", name="k_cat", value=1.23456),
    ]

    tex_path = tmp_path / "params.tex"
    SensitivityParameter.parameter_to_latex(tex_path, params)

    assert tex_path.exists()

    content = tex_path.read_text()
    assert "\\begin{tabular}" in content
    assert "k\\_cat" in content  # underscore escaped
    assert "1.23" in content  # formatted float
