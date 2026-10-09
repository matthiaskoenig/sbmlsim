"""The global sensitivity analyses."""

import copy
from pathlib import Path

import numpy as np
import pytest
from SALib.analyze import sobol as salib_sobol

from sbmlsim import sensitivity
from sbmlsim.result import ScanResult
from sbmlsim.sensitivity.result import PARAMETER, PARAMETER_2
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation, sampling
from sbmlsim.simulation.sampling import Uniform
from sbmlsim.simulator import Simulator
from tests.sensitivity.models import ISHIGAMI
from tests.simulator.models import BLOWUP, sbml

PI = np.pi
BOUNDS = {"x1": Uniform(-PI, PI), "x2": Uniform(-PI, PI), "x3": Uniform(-PI, PI)}
OBSERVABLES = [Formula("y_max", "max(y)")]


def _run(design: Dimension, *others: Dimension) -> ScanResult:
    model = Simulator().load(sbml(ISHIGAMI))
    return Simulator(n_workers=1).run(
        model, Scan(Simulation(end=1, steps=1), [*others, design]), OBSERVABLES
    )


def test_sobol_of_the_ishigami_function() -> None:
    res = _run(sampling.sobol(BOUNDS, 1024, seed=1))
    s = sensitivity.sobol(res)
    np.testing.assert_allclose(s["y_max.S1"].values, [0.314, 0.442, 0.0], atol=0.05)
    np.testing.assert_allclose(s["y_max.ST"].values, [0.558, 0.442, 0.244], atol=0.05)
    assert "y_max.ST_conf" in s and s.method == "sobol"


def test_sobol_second_order() -> None:
    res = _run(sampling.sobol(BOUNDS, 1024, seed=1, second_order=True))
    s = sensitivity.sobol(res)
    assert s["y_max.S2"].dims == (PARAMETER, PARAMETER_2)
    assert "y_max.S2_conf" in s
    assert s["y_max.S2"].values[0, 2] == pytest.approx(0.244, abs=0.06)


def test_sobol_equals_salib_on_the_same_values() -> None:
    design = sampling.sobol(BOUNDS, 64, seed=2)
    assert design.design is not None
    res = _run(design)
    s = sensitivity.sobol(res)
    problem = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[0.0, 1.0]] * 3}
    y = res["y_max"].values
    expected = salib_sobol.analyze(
        problem,
        y,
        calc_second_order=False,
        num_resamples=100,
        conf_level=0.95,
        print_to_console=False,
        seed=design.design.options["seed"],
    )
    np.testing.assert_allclose(s["y_max.S1"].values, expected["S1"])
    np.testing.assert_allclose(s["y_max.ST"].values, expected["ST"])


def test_fast_and_morris_rank_the_ishigami_parameters() -> None:
    f = sensitivity.fast(_run(sampling.fast(BOUNDS, 1025, seed=3)))
    np.testing.assert_allclose(f["y_max.S1"].values, [0.314, 0.442, 0.0], atol=0.06)
    m = sensitivity.morris(_run(sampling.morris(BOUNDS, 100, seed=4)))
    mu_star = m["y_max.mu_star"].values
    assert mu_star[0] > mu_star[2] > 0.0 and mu_star[1] > 0.0
    assert set(m.observables) == {"y_max"}
    assert {"y_max.mu", "y_max.sigma", "y_max.mu_star_conf"} <= set(m.ds.data_vars)


def test_indices_per_label_of_another_dimension() -> None:
    shift = Dimension("shift", values={"x3": [0.0, 0.0]})  # changes nothing, two labels
    res = _run(
        sampling.sobol({"x1": Uniform(-PI, PI), "x2": Uniform(-PI, PI)}, 64, seed=5),
        shift,
    )
    s = sensitivity.sobol(res)
    assert s["y_max.S1"].dims == (PARAMETER, "shift")


def test_timecourses_and_scalars() -> None:
    model = Simulator().load(sbml(ISHIGAMI))
    design = sampling.sobol(BOUNDS, 16, seed=6)
    res = Simulator(n_workers=1).run(
        model,
        Scan(Simulation(end=1, steps=2), [design]),
        [Formula("yt", "y"), Formula("y_max", "max(y)")],
    )
    s = sensitivity.sobol(res)
    assert s["yt.ST"].dims == (PARAMETER, "time")
    assert s["y_max.ST"].dims == (PARAMETER,)


def test_a_failed_point_gives_nan_indices(caplog: pytest.LogCaptureFixture) -> None:
    model = Simulator().load(sbml(BLOWUP))
    design = sampling.sobol({"k": Uniform(0.1, 3.0)}, 8, seed=7)
    res = Simulator(n_workers=1).run(
        model,
        Scan(Simulation(end=1, steps=2), [design]),
        [Formula("s_max", "max(S)")],
        on_error="flag",
    )
    assert res["status"].values.any()
    s = sensitivity.sobol(res)
    assert np.isnan(s["s_max.S1"].values).all()
    assert sum(r.name == "sbmlsim.sensitivity.indices" for r in caplog.records) == 1


def test_a_constant_element_gives_nan_variance_indices_and_zero_effects() -> None:
    model = Simulator().load(sbml(ISHIGAMI))
    obs = [Formula("one", "1 + 0 * max(y)")]
    scan = Scan(Simulation(end=1, steps=1), [sampling.sobol(BOUNDS, 16, seed=1)])
    s = sensitivity.sobol(Simulator(n_workers=1).run(model, scan, obs))
    assert np.isnan(s["one.S1"].values).all() and np.isnan(s["one.ST"].values).all()
    scan = Scan(Simulation(end=1, steps=1), [sampling.fast(BOUNDS, 65, seed=1)])
    f = sensitivity.fast(Simulator(n_workers=1).run(model, scan, obs))
    assert np.isnan(f["one.S1"].values).all()
    scan = Scan(Simulation(end=1, steps=1), [sampling.morris(BOUNDS, 4, seed=1)])
    m = sensitivity.morris(Simulator(n_workers=1).run(model, scan, obs))
    for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
        np.testing.assert_array_equal(m[f"one.{key}"].values, 0.0)


def test_a_stored_result_is_analysed(tmp_path: Path) -> None:
    res = _run(sampling.sobol(BOUNDS, 32, seed=8))
    path = tmp_path / "r.nc"
    res.to_netcdf(path)
    again = sensitivity.sobol(ScanResult.from_netcdf(path))
    np.testing.assert_allclose(
        again["y_max.ST"].values, sensitivity.sobol(res)["y_max.ST"].values
    )


def test_a_wrong_number_of_points_raises() -> None:
    res = _run(sampling.sobol(BOUNDS, 16, seed=9))
    ds = res.ds.copy(deep=True)
    ds.attrs = copy.deepcopy(res.ds.attrs)
    for d in ds.attrs["scan"]["dimensions"]:
        if d.get("design"):
            d["design"]["options"]["n"] = 32
    with pytest.raises(ValueError, match="points"):
        sensitivity.sobol(ScanResult(ds))
