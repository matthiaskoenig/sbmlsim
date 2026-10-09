"""The designs of a fit: Fisher, profiles, repeats."""

import numpy as np
import pytest
from scipy import stats

from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.identifiability import (
    IdentifiabilityResult,
    ParameterProfile,
    ProfileSettings,
)
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.simulation.sampling import fit_parameters, fit_repeats, profile_parameters


def _fisher(
    matrix: np.ndarray, targets: list[str] | None = None, n: int = 102
) -> FisherInformation:
    return FisherInformation(
        opid="op",
        sid="best",
        pids=["k1", "k2"],
        values=np.array([1.0, 10.0]),
        scale=ParameterScaleType.LOG10,
        matrix=matrix,
        cost=1.0,
        n=n,
        units=["1/min", None],
        targets=targets or [],
    )


def test_fit_parameters_follow_the_fisher_covariance() -> None:
    fisher = _fisher(np.array([[400.0, 100.0], [100.0, 900.0]]))
    dimension = fit_parameters(fisher, 20000, seed=1)
    logs = np.column_stack(
        [
            np.log10(np.asarray(dimension.values["k1"].magnitude)),
            np.log10(dimension.values["k2"]),
        ]
    )
    np.testing.assert_allclose(
        np.cov(logs, rowvar=False), fisher.covariance, rtol=0.05, atol=1e-6
    )
    np.testing.assert_allclose(np.median(logs, axis=0), [0.0, 1.0], atol=0.01)
    assert str(dimension.values["k1"].units) == "1 / minute"
    assert dimension.design is not None
    assert dimension.design.method == "fit_parameters"
    assert dimension.design is not None
    assert dimension.design.options["opid"] == "op"


def test_a_rank_deficient_covariance_warns_once_and_draws(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fisher = _fisher(np.array([[1.0, 1.0], [1.0, 1.0]]))
    dimension = fit_parameters(fisher, 100, seed=1)
    assert np.isfinite(np.asarray(dimension.values["k2"])).all()
    assert sum("rank" in r.message for r in caplog.records) == 1


def test_targets_map_parameters_and_versions_raise() -> None:
    fisher = _fisher(np.eye(2) * 100.0)
    assert set(fit_parameters(fisher, 5, seed=1, targets={"k1": "kcat"}).values) == {
        "kcat",
        "k2",
    }
    with pytest.raises(ValueError, match="target"):
        fit_parameters(fisher, 5, seed=1, targets={"k1": "k", "k2": "k"})


def _profile(
    values: np.ndarray, costs: np.ndarray, lower: float | None, upper: float | None
) -> ParameterProfile:
    return ParameterProfile(
        pid="k1",
        values=values,
        costs=costs,
        paths=values[:, None],
        converged=np.ones(len(values), dtype=bool),
        index_optimum=int(np.argmin(costs)),
        ci_lower=lower,
        ci_upper=upper,
    )


def _identifiability(
    profile: ParameterProfile, bounds: tuple[float, float]
) -> IdentifiabilityResult:
    parameter = FitParameter(
        pid="k1", start_value=1.0, lower_bound=bounds[0], upper_bound=bounds[1]
    )
    return IdentifiabilityResult(
        opid="op",
        parameter_set=ParameterSet(sid="best", values={"k1": 1.0}, cost=0.0),
        parameters=[parameter],
        settings=ProfileSettings(),
        fit_settings=FitSettings(parameter_scale=ParameterScaleType.LOG10),
        cost=0.0,
        profiles={"k1": profile},
    )


def test_profile_parameters_follow_the_likelihood_of_the_profile() -> None:
    # a quadratic profile in log10 space: the likelihood ratio is a normal of sd 0.1
    x = np.linspace(-0.5, 0.5, 201)
    profile = _profile(10.0**x, 0.5 * (x / 0.1) ** 2, 10.0**-0.2, 10.0**0.2)
    dimension = profile_parameters(_identifiability(profile, (1e-3, 1e3)), 4000, seed=2)
    logs = np.log10(np.asarray(dimension.values["k1"]))
    assert stats.kstest(logs, stats.norm(0.0, 0.1).cdf).pvalue > 0.01
    assert dimension.design is not None
    assert dimension.design.options["alpha"] == pytest.approx(0.95)


def test_a_flat_side_reaches_the_bound() -> None:
    # flat above the optimum: the likelihood stays high up to the upper bound
    x = np.linspace(-0.5, 0.5, 101)
    costs = np.where(x < 0.0, 0.5 * (x / 0.1) ** 2, 0.0)
    profile = _profile(10.0**x, costs, 10.0**-0.2, None)
    dimension = profile_parameters(_identifiability(profile, (1e-3, 1e2)), 4000, seed=3)
    values = np.asarray(dimension.values["k1"])
    assert values.max() > 10.0**0.5 and values.max() <= 1e2


def test_a_profile_without_a_converged_optimum_raises() -> None:
    x = np.linspace(-0.5, 0.5, 11)
    profile = _profile(10.0**x, x**2, None, None)
    profile.converged[profile.index_optimum] = False
    with pytest.raises(ValueError, match="converged"):
        profile_parameters(_identifiability(profile, (1e-3, 1e3)), 10, seed=1)


class _Result:
    """A stand-in of OptimizationResult with three repeats."""

    def __init__(self, target: str | None = None) -> None:
        self.parameters = [
            FitParameter(pid="k1", start_value=1.0, unit="1/min", target=target),
            FitParameter(pid="k2", start_value=1.0, unit="mM"),
        ]

    def parameter_sets(self, size: int = 1) -> list[ParameterSet]:
        sets = [
            ParameterSet(
                sid=f"run{k}",
                values={"k1": 1.0 + k, "k2": 2.0 * k},
                units={"k1": "1/min", "k2": None},
                cost=float(k),
            )
            for k in range(3)
        ]
        return sets[:size]


def test_fit_repeats_take_the_best_sets() -> None:
    dimension = fit_repeats(_Result(), 2)
    assert dimension.labels.tolist() == ["run0", "run1"]
    np.testing.assert_allclose(dimension.values["k1"].magnitude, [1.0, 2.0])
    np.testing.assert_allclose(dimension.values["k2"], [0.0, 2.0])
    assert dimension.design is not None
    assert dimension.design.options["costs"] == [0.0, 1.0]


def test_fit_repeats_write_the_target_of_a_parameter() -> None:
    dimension = fit_repeats(_Result(target="kcat"), 2)
    assert set(dimension.values) == {"kcat", "k2"}
    assert set(fit_repeats(_Result(), 2, targets={"k1": "x"}).values) == {"x", "k2"}
    with pytest.raises(ValueError, match="positive integer"):
        fit_repeats(_Result(), 0)


def test_the_fisher_targets_are_written_and_overridden() -> None:
    fisher = _fisher(np.eye(2) * 100.0, targets=["kcat", "k2"])
    assert set(fit_parameters(fisher, 5, seed=1).values) == {"kcat", "k2"}
    assert set(fit_parameters(fisher, 5, seed=1, targets={"k2": "y"}).values) == {
        "kcat",
        "y",
    }
    with pytest.raises(ValueError, match="no fitted parameters"):
        fit_parameters(fisher, 5, seed=1, targets={"nope": "y"})
    twice = _fisher(np.eye(2) * 100.0, targets=["k", "k"])
    with pytest.raises(ValueError, match="more than once"):
        fit_parameters(twice, 5, seed=1)


def test_a_near_singular_information_has_no_spread_where_it_is_unconstrained() -> None:
    fisher = _fisher(np.diag([1.0, 1e-12]))
    dimension = fit_parameters(fisher, 500, seed=1)
    k1, k2 = (
        np.asarray(dimension.values["k1"].magnitude),
        np.asarray(dimension.values["k2"]),
    )
    assert np.isfinite(k1).all() and np.isfinite(k2).all()
    assert np.log10(k1).std() > 0.01
    np.testing.assert_allclose(k2, 10.0)


def test_no_degrees_of_freedom_raise() -> None:
    with pytest.raises(ValueError, match="degrees of freedom"):
        fit_parameters(_fisher(np.eye(2), n=2), 5, seed=1)


def test_a_profile_writes_the_target_and_versions_raise() -> None:
    x = np.linspace(-0.5, 0.5, 201)
    profile = _profile(10.0**x, 0.5 * (x / 0.1) ** 2, 10.0**-0.2, 10.0**0.2)
    result = _identifiability(profile, (1e-3, 1e3))
    result.parameters[0].target = "kcat"
    assert set(profile_parameters(result, 5, seed=1).values) == {"kcat"}
    with pytest.raises(ValueError, match="no fitted parameters"):
        profile_parameters(result, 5, seed=1, targets={"nope": "y"})
    other = _profile(10.0**x, 0.5 * (x / 0.1) ** 2, 10.0**-0.2, 10.0**0.2)
    other.pid = "k2"
    result.profiles["k2"] = other
    result.parameters.append(
        FitParameter(
            pid="k2", start_value=1.0, lower_bound=1e-3, upper_bound=1e3, target="kcat"
        )
    )
    with pytest.raises(ValueError, match="more than once"):
        profile_parameters(result, 5, seed=1)


def test_a_profile_cut_at_the_threshold_keeps_its_spread() -> None:
    # profile_likelihood stops at the first point at or above the threshold
    x = np.linspace(-0.196, 0.196, 99)
    profile = _profile(10.0**x, 0.5 * (x / 0.1) ** 2, 10.0**-0.196, 10.0**0.196)
    dimension = profile_parameters(
        _identifiability(profile, (1e-3, 1e3)), 20000, seed=4
    )
    logs = np.log10(np.asarray(dimension.values["k1"]))
    assert logs.std() == pytest.approx(0.1, rel=0.03)
    assert stats.kstest(logs, stats.norm(0.0, 0.1).cdf).pvalue > 0.01
