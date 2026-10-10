"""The formula of a Data of type FUNCTION is the math of PEtab on labelled arrays."""

import numpy as np
import pytest
import xarray as xr

from sbmlsim.data import evaluate_function
from sbmlsim.units import ureg


def _tc(values: list[float], unit: str | None = "dimensionless") -> xr.DataArray:
    return xr.DataArray(np.array(values), dims=("time",), attrs={"units": unit})


def test_a_ratio_of_quantities_keeps_the_units() -> None:
    x = _tc([1.0, 2.0], "mmol/l")
    y = _tc([2.0, 4.0], "mmol/l")
    ratio = evaluate_function("x / y", {"x": x, "y": y}, ureg)
    assert ratio.dims == ("time",)
    q = ureg.Quantity(ratio.values, ratio.attrs["units"])
    np.testing.assert_allclose(q.to("dimensionless").magnitude, [0.5, 0.5])
    v = xr.DataArray(2.0, attrs={"units": "l"})
    amount = evaluate_function("x * v", {"x": x, "v": v}, ureg)
    q = ureg.Quantity(amount.values, amount.attrs["units"])
    assert q.to("mmol").magnitude.tolist() == pytest.approx([2.0, 4.0])


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Y/max(Y)", [0.25, 0.5, 1.0]),
        ("Y - min(Y)", [0.0, 1.0, 3.0]),
        ("max(Y, 2)", [2.0, 2.0, 4.0]),
        ("min(Y, 2)", [1.0, 2.0, 2.0]),
        ("Y/max(Y + Z)", [1 / 8, 2 / 8, 4 / 8]),
        ("max(max(Y), 2) + 0*Y", [4.0, 4.0, 4.0]),
        ("Ymax/max(Y)", [0.25, 0.25, 0.25]),
        ("Y^2 + ln(Z)", [1.0, 4.0, 16.0 + np.log(4.0)]),
        ("piecewise(1, Y > 1.5, 0)", [0.0, 1.0, 1.0]),
    ],
)
def test_reductions_and_petab_math(formula: str, expected: list[float]) -> None:
    variables = {
        "Y": _tc([1.0, 2.0, 4.0], None),
        "Z": _tc([1.0, 1.0, 4.0], None),
        "Ymax": _tc([1.0, 1.0, 1.0], None),
    }
    np.testing.assert_allclose(
        evaluate_function(formula, variables, ureg).values, expected
    )


def test_a_reduction_ignores_the_padding() -> None:
    y = xr.DataArray([1.0, 2.0, np.nan], dims=("_point",), attrs={"units": None})
    np.testing.assert_allclose(
        evaluate_function("Y/max(Y)", {"Y": y}, ureg).values, [0.5, 1.0, np.nan]
    )


def test_a_reduction_of_quantities_keeps_the_units() -> None:
    y = _tc([1.0, 2.0, 4.0], "mmol/l")
    shifted = evaluate_function("Y - min(Y)", {"Y": y}, ureg)
    assert ureg.Unit(shifted.attrs["units"]) == ureg.Unit("mmol/l")


def test_a_formula_of_parameters_is_a_number() -> None:
    value = evaluate_function("2 * k", {"k": 3.0}, ureg)
    assert value.dims == () and float(value) == pytest.approx(6.0)
    assert value.attrs["units"] == "dimensionless"


@pytest.mark.parametrize("formula", ["Y +", "max(Y", "foo(Y)"])
def test_invalid_math_is_reported(formula: str) -> None:
    with pytest.raises(ValueError):
        evaluate_function(formula, {"Y": _tc([1.0])}, ureg)


def test_an_unknown_identifier_is_reported() -> None:
    with pytest.raises(ValueError, match="W"):
        evaluate_function("Y / W", {"Y": _tc([1.0])}, ureg)


def test_a_reduction_of_a_scan_is_per_simulation_by_name() -> None:
    # the time is first here: the reduction finds it by its name
    y = xr.DataArray(
        np.array([[1.0, 1.0], [2.0, 1.0], [4.0, 2.0]]),
        dims=("time", "dose"),
        attrs={"units": None},
    )
    normalized = evaluate_function("Y/max(Y)", {"Y": y}, ureg)
    assert set(normalized.dims) == {"time", "dose"}
    np.testing.assert_allclose(
        normalized.transpose("dose", "time").values, [[0.25, 0.5, 1.0], [0.5, 0.5, 1.0]]
    )
    peak = evaluate_function("max(Y)", {"Y": y}, ureg)
    assert peak.dims == ("dose",)
    np.testing.assert_allclose(peak.values, [4.0, 2.0])


def test_arrays_broadcast_by_dimension_name() -> None:
    y = xr.DataArray(np.ones((2, 3)), dims=("dose", "time"), attrs={"units": "mM"})
    dose = xr.DataArray([1.0, 2.0], dims=("dose",), attrs={"units": "mg"})
    per_dose = evaluate_function("y / d", {"y": y, "d": dose}, ureg)
    assert per_dose.dims == ("dose", "time")
    np.testing.assert_allclose(per_dose.values[:, 0], [1.0, 0.5])


def test_different_coordinates_of_a_dimension_raise() -> None:
    a = xr.DataArray(
        [1.0, 2.0], dims=("time",), coords={"time": [0.0, 1.0]}, attrs={"units": None}
    )
    b = xr.DataArray(
        [1.0, 2.0], dims=("time",), coords={"time": [0.0, 2.0]}, attrs={"units": None}
    )
    with pytest.raises(ValueError, match="coordinates"):
        evaluate_function("a + b", {"a": a, "b": b}, ureg)


@pytest.mark.parametrize("formula", ["mean(Y)", "at(Y, 1)"])
def test_mean_and_at_are_no_reductions_of_data(formula: str) -> None:
    with pytest.raises(ValueError, match="observable"):
        evaluate_function(formula, {"Y": _tc([1.0, 2.0])}, ureg)
