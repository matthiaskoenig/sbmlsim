"""The global sensitivity analyses."""

import copy
import ctypes
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from SALib.analyze import fast as salib_fast
from SALib.analyze import morris as salib_morris
from SALib.analyze import sobol as salib_sobol
from SALib.sample import morris as salib_morris_sampler

from sbmlsim import sensitivity
from sbmlsim.result import ScanResult
from sbmlsim.sensitivity.result import PARAMETER, PARAMETER_2, SensitivityResult
from sbmlsim.simulation import (
    Custom,
    Dimension,
    Formula,
    Observable,
    Scan,
    Simulation,
    sampling,
)
from sbmlsim.simulation.sampling import Uniform
from sbmlsim.simulation.sampling.designs import unit_problem
from sbmlsim.simulator import Simulator
from tests.sensitivity.models import CHAIN, ISHIGAMI, s2_end_fails_for_high_s1_and_k1
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
    np.testing.assert_allclose(s["y_max.S1"].values[:, 0], s["y_max.S1"].values[:, 1])
    assert np.isfinite(s["y_max.S1"].values).all()


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


def _take_c_output(capfd: pytest.CaptureFixture[str]) -> None:
    """Take the messages of SUNDIALS of the points which failed, see test_simulator.

    The streams of C are flushed with POSIX ctypes while the capture runs;
    on Windows the output of C which is still buffered is not taken.
    """
    if sys.platform != "win32":
        ctypes.CDLL(None).fflush(None)
    capfd.readouterr()


def test_a_failed_point_gives_nan_indices(
    caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]
) -> None:
    """S' = k S^2 blows up before t = 1 for k S0 > 1: only S0 = 1 has failed points."""
    model = Simulator().load(sbml(BLOWUP))
    design = sampling.sobol({"k": Uniform(0.1, 3.0)}, 8, seed=7)
    initial = Dimension("S0", values={"S": [0.1, 1.0]})
    try:
        res = Simulator(n_workers=1).run(
            model,
            Scan(Simulation(end=1, steps=2), [initial, design]),
            [Formula("s_max", "max(S)")],
            on_error="flag",
        )
        caplog.clear()
        s = sensitivity.sobol(res)
        alone = sensitivity.sobol(res.isel(S0=[0]))
    finally:
        _take_c_output(capfd)
    assert res["status"].sel(S0=1).values.any()
    assert not res["status"].sel(S0=0).values.any()
    assert np.isnan(s["s_max.S1"].sel(S0=1).values).all()
    # the element without a failed point is the one of its analysis alone
    np.testing.assert_array_equal(
        s["s_max.S1"].sel(S0=0).values, alone["s_max.S1"].sel(S0=0).values
    )
    assert np.isfinite(s["s_max.S1"].sel(S0=0).values).all()
    records = [r for r in caplog.records if r.name == "sbmlsim.sensitivity.indices"]
    assert len(records) == 1 and records[0].getMessage().startswith("1 elements")


def test_a_constant_element_gives_nan_variance_indices_and_zero_effects() -> None:
    model = Simulator().load(sbml(ISHIGAMI))
    obs = [Formula("one", "1 + 1e-15 * max(y)")]
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


def _conf(s: SensitivityResult, name: str, key: str) -> np.ndarray:
    return s[f"{name}.{key}"].values


@pytest.mark.parametrize("seed", [0, 3])
def test_the_intervals_are_reproducible_and_independent_of_other_elements(
    seed: int,
) -> None:
    model = Simulator().load(sbml(ISHIGAMI))
    for design, analysis, n in (
        (sampling.sobol(BOUNDS, 64, seed=seed), sensitivity.sobol, 64),
        (sampling.fast(BOUNDS, 65, seed=seed), sensitivity.fast, 65),
    ):
        assert len(design) >= n
        scan = Scan(Simulation(end=1, steps=1), [design])
        both = Simulator(n_workers=1).run(
            model, scan, [Formula("y_min", "min(y)"), *OBSERVABLES]
        )
        first = analysis(both)
        again = analysis(both)
        alone = analysis(both, observables=["y_max"])
        np.testing.assert_array_equal(
            _conf(first, "y_max", "S1_conf"), _conf(again, "y_max", "S1_conf")
        )
        np.testing.assert_array_equal(
            _conf(first, "y_max", "ST_conf"), _conf(alone, "y_max", "ST_conf")
        )
        assert np.isfinite(_conf(first, "y_max", "S1_conf")).all()


def test_fast_leaves_the_global_random_state() -> None:
    res = _run(sampling.fast(BOUNDS, 65, seed=3))
    np.random.seed(123)
    before = np.random.get_state()
    sensitivity.fast(res)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_the_points_are_analysed_in_the_order_of_their_labels() -> None:
    res = _run(sampling.sobol(BOUNDS, 64, seed=2))
    permutation = np.random.default_rng(0).permutation(len(res.ds["sobol"]))
    s = sensitivity.sobol(res)
    shuffled = sensitivity.sobol(res.isel(sobol=permutation))
    np.testing.assert_allclose(shuffled["y_max.S1"].values, s["y_max.S1"].values)
    np.testing.assert_allclose(shuffled["y_max.ST"].values, s["y_max.ST"].values)
    with pytest.raises(ValueError, match="sobol"):
        sensitivity.sobol(res.isel(sobol=slice(0, 100)))


def test_the_options_hold_the_design_and_the_analysis(tmp_path: Path) -> None:
    res = _run(sampling.sobol(BOUNDS, 32, seed=0))
    s = sensitivity.sobol(res, conf_level=0.9, num_resamples=50)
    assert s.ds.attrs["options"]["n"] == 32
    assert s.ds.attrs["options"]["conf_level"] == 0.9
    assert s.ds.attrs["options"]["num_resamples"] == 50
    assert "bootstrap_seed" in s.ds.attrs["options"]
    path = tmp_path / "s.nc"
    s.to_netcdf(path)
    again = SensitivityResult.from_netcdf(path)
    assert again.ds.attrs["options"] == s.ds.attrs["options"]


@pytest.mark.parametrize(
    "kwargs", [{"conf_level": 1.5}, {"conf_level": 0.0}, {"num_resamples": 0}]
)
def test_the_arguments_are_validated(kwargs: dict[str, float]) -> None:
    res = _run(sampling.sobol(BOUNDS, 16, seed=1))
    with pytest.raises(ValueError, match=r"conf_level|num_resamples"):
        sensitivity.sobol(res, **kwargs)  # ty: ignore[invalid-argument-type]


def test_morris_with_other_levels_equals_salib_on_its_grid() -> None:
    design = sampling.morris(BOUNDS, 20, levels=6, seed=4)
    s = sensitivity.morris(_run(design))
    problem = unit_problem(3)
    grid = salib_morris_sampler.sample(problem, 20, num_levels=6, seed=4)
    # the grid is the one of the design, the values are a function of the centred cube
    res = _run(design)
    expected = salib_morris.analyze(
        problem,
        grid,
        res["y_max"].values,
        num_levels=6,
        scaled=False,
        print_to_console=False,
        seed=4,
    )
    np.testing.assert_allclose(s["y_max.mu"].values, expected["mu"])
    np.testing.assert_allclose(s["y_max.mu_star"].values, expected["mu_star"])


def test_fast_with_another_m_equals_salib() -> None:
    res = _run(sampling.fast(BOUNDS, 257, m=2, seed=3))
    f = sensitivity.fast(res)
    assert f.ds.attrs["options"]["m"] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = salib_fast.analyze(
            unit_problem(3), res["y_max"].values, M=2, print_to_console=False
        )
    np.testing.assert_allclose(f["y_max.S1"].values, expected["S1"])
    np.testing.assert_allclose(f["y_max.ST"].values, expected["ST"])


#: the times of the chain: before the saturation of [S3] and ten after it
CHAIN_TIMES = [0.0, 2.0, 5.0, 10.0, *range(100, 1001, 100)]


def _chain_run(design: Dimension) -> ScanResult:
    model = Simulator().load(sbml(CHAIN))
    return Simulator(n_workers=1).run(
        model,
        Scan(Simulation(end=1000, times=CHAIN_TIMES), [design]),
        [
            Formula("s3", "[S3]"),
            Formula("total", "[S1] + [S2] + [S3]"),
            Formula("total_end", "at([S1] + [S2] + [S3], 1000)"),
        ],
    )


CHAIN_BOUNDS = {"k1": Uniform(relative=0.15), "k2": Uniform(relative=0.15)}


@pytest.mark.parametrize("method", ["sobol", "fast"])
def test_the_noise_of_the_integrator_gives_no_variance_indices(method: str) -> None:
    """[S3] saturates and the total is conserved: they vary by the error only."""
    model = Simulator().load(sbml(CHAIN))
    if method == "sobol":
        design = sampling.sobol(CHAIN_BOUNDS, 64, seed=1, model=model)
    else:
        design = sampling.fast(CHAIN_BOUNDS, 65, seed=1, model=model)
    res = _chain_run(design)
    analysis = sensitivity.sobol if method == "sobol" else sensitivity.fast
    s = analysis(res)
    assert s.ds.attrs["options"]["tolerance"] == pytest.approx(1e-7)
    late = s["s3.ST"].sel(time=slice(100, None))
    assert late.sizes["time"] == 10 and np.isnan(late.values).all()
    assert np.isnan(s["total.ST"].values).all()
    assert np.isnan(s["total_end.S1"].values).all()
    # before the saturation [S3] depends on both rates
    early = s["s3.ST"].sel(time=slice(1, 50))
    assert np.isfinite(early.values).all()
    # without a tolerance the noise gives indices
    noisy = analysis(res, tolerance=0.0)
    assert np.isfinite(noisy["total_end.ST"].values).all()


def test_the_noise_of_the_integrator_gives_no_elementary_effects() -> None:
    model = Simulator().load(sbml(CHAIN))
    design = sampling.morris(CHAIN_BOUNDS, 10, seed=1, model=model)
    m = sensitivity.morris(_chain_run(design))
    for key in ("mu", "mu_star", "sigma", "mu_star_conf"):
        np.testing.assert_array_equal(m[f"total_end.{key}"].values, 0.0)
        late = m[f"s3.{key}"].sel(time=slice(100, None)).values
        np.testing.assert_array_equal(late, 0.0)
    assert (m["s3.mu_star"].sel(time=slice(1, 50)).values > 0).all()


def test_a_small_real_effect_keeps_its_indices() -> None:
    model = Simulator().load(
        sbml("model small\n  x1 = 0; x2 = 0\n  y := 1 + 1e-5 * x1 + 1e-6 * x2\nend\n")
    )
    bounds = {"x1": Uniform(0.0, 1.0), "x2": Uniform(0.0, 1.0)}
    scan = Scan(Simulation(end=1, steps=1), [sampling.sobol(bounds, 256, seed=1)])
    res = Simulator(n_workers=1).run(model, scan, OBSERVABLES)
    assert np.ptp(res["y_max"].values) < 2e-5
    s = sensitivity.sobol(res)
    np.testing.assert_allclose(s["y_max.S1"].values, [100 / 101, 1 / 101], atol=0.02)
    # an explicit tolerance wins
    coarse = sensitivity.sobol(res, tolerance=1e-4)
    assert np.isnan(coarse["y_max.S1"].values).all()
    assert coarse.ds.attrs["options"]["tolerance"] == 1e-4


def test_the_tolerance_follows_the_integrator(tmp_path: Path) -> None:
    model = Simulator().load(sbml(ISHIGAMI))
    scan = Scan(Simulation(end=1, steps=1), [sampling.morris(BOUNDS, 4, seed=1)])
    res = Simulator(n_workers=1, relative_tolerance=1e-6).run(model, scan, OBSERVABLES)
    assert sensitivity.morris(res).ds.attrs["options"]["tolerance"] == pytest.approx(
        1e-3
    )
    path = tmp_path / "r.nc"
    res.to_netcdf(path)
    stored = ScanResult.from_netcdf(path)
    assert sensitivity.morris(stored).ds.attrs["options"]["tolerance"] == pytest.approx(
        1e-3
    )
    # below 1e-10 the error of the integrator does not decrease
    tight = Simulator(n_workers=1, relative_tolerance=1e-12).run(
        model, scan, OBSERVABLES
    )
    assert sensitivity.morris(tight).ds.attrs["options"]["tolerance"] == pytest.approx(
        1e-7
    )
    # a result without the settings of its integrator has the one of the default
    ds = res.ds.copy()
    ds.attrs = {k: v for k, v in res.ds.attrs.items() if k != "integrator_settings"}
    assert sensitivity.morris(ScanResult(ds)).ds.attrs["options"][
        "tolerance"
    ] == pytest.approx(1e-7)
    with pytest.raises(ValueError, match="tolerance"):
        sensitivity.morris(res, tolerance=-1.0)


def _chain_with_conditions(design: Dimension, *, fail: bool) -> ScanResult:
    """Run the chain under three conditions; with `fail` the high one has failed points."""
    model = Simulator().load(sbml(CHAIN))
    conditions = Dimension(
        "S1_0",
        values={"[S1]": [0.1, 1.0, 10.0]},
        labels=["low", "reference", "high"],
    )
    observables: list[Observable] = [
        Formula("s2", "[S2]"),
        Formula("s2_max", "max([S2])"),
    ]
    if fail:
        observables.append(
            Custom(
                "s2_end",
                s2_end_fails_for_high_s1_and_k1,
                "dimensionless",
                symbols=["[S1]", "[S2]", "k1"],
            )
        )
    return Simulator(n_workers=1).run(
        model,
        Scan(Simulation(end=4, steps=4), [conditions, design]),
        observables,
        on_error="flag",
    )


@pytest.mark.parametrize("method", ["local", "sobol", "fast", "morris"])
def test_the_chain_with_conditions_timecourses_and_a_failed_point(
    method: str, caplog: pytest.LogCaptureFixture
) -> None:
    model = Simulator().load(sbml(CHAIN))
    designs = {
        "local": sampling.local(["k1", "k2"], 0.01, model=model),
        "sobol": sampling.sobol(CHAIN_BOUNDS, 64, seed=1, model=model),
        "fast": sampling.fast(CHAIN_BOUNDS, 65, seed=1, model=model),
        "morris": sampling.morris(CHAIN_BOUNDS, 10, seed=1, model=model),
    }
    analysis = getattr(sensitivity, method)
    failed = _chain_with_conditions(designs[method], fail=True)
    assert failed["status"].sel(S1_0="high").values.any()
    assert not failed["status"].sel(S1_0=["low", "reference"]).values.any()
    caplog.clear()
    s = analysis(failed)
    expected = analysis(_chain_with_conditions(designs[method], fail=False))
    key = {"local": "normalized", "sobol": "ST", "fast": "ST", "morris": "mu_star"}[
        method
    ]
    assert s[f"s2.{key}"].dims == (PARAMETER, "S1_0", "time")
    assert s[f"s2_max.{key}"].dims == (PARAMETER, "S1_0")
    # the conditions without a failed point are the ones of a run without one
    for name in ("s2", "s2_max"):
        for variable in expected.ds.data_vars:
            if str(variable).startswith(f"{name}."):
                np.testing.assert_array_equal(
                    s[str(variable)].sel(S1_0=["low", "reference"]).values,
                    expected[str(variable)].sel(S1_0=["low", "reference"]).values,
                )
        later = expected[f"{name}.{key}"].sel(S1_0=["low", "reference"])
        if "time" in later.dims:
            # at t = 0 [S2] = 0 for every point, constant
            later = later.sel(time=slice(1, None))
        assert np.isfinite(later.values).all()
    high = s[f"s2.{key}"].sel(S1_0="high")
    if method == "local":
        # only the point k1+ failed
        assert np.isnan(high.sel(parameter="k1").values).all()
        assert np.isfinite(high.sel(parameter="k2", time=slice(1, None)).values).all()
    else:
        assert np.isnan(high.values).all()
    records = [r for r in caplog.records if r.name == "sbmlsim.sensitivity.indices"]
    # s2 at five time points, s2_max and s2_end of the high condition
    assert len(records) == 1 and records[0].getMessage().startswith("7 elements")
