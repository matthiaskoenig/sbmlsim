"""Test scans."""

import numpy as np

from examples import scan as example_scan


def test_scan0d() -> None:
    res = example_scan.run_scan0d()
    assert res.dims == ()
    assert res["PX"].dims == ("time",)


def test_scan1d() -> None:
    res = example_scan.run_scan1d()
    assert res["PX"].dims == ("dim1", "time")
    np.testing.assert_allclose(res["n"].values[:, 0], np.linspace(2, 10, 8))


def test_scan2d() -> None:
    res = example_scan.run_scan2d()
    assert res["PX"].shape == (8, 4, 101)


def test_scan1d_distribution() -> None:
    """The sample has a seed, the scan is the same every time."""
    res = example_scan.run_scan1d_distribution()
    assert res.ds.sizes["dim1"] == 50
    again = example_scan.run_scan1d_distribution()
    np.testing.assert_array_equal(res["n"].values, again["n"].values)
