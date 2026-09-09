"""Test the profile likelihood analysis of the parameters of a fit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from sbmlsim.fit import FitSettings, ParameterSet, ParameterSets
from sbmlsim.fit.cli import FitDefinition, identifiability_cli
from sbmlsim.fit.identifiability import (
    Identifiability,
    IdentifiabilityResult,
    ParameterProfile,
    ProfileSettings,
    cost_threshold,
    plot_profile,
    plot_profiles,
    profile_likelihood,
)
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.report import FitReport

#: settings of a fast analysis of the reference problem, plain scans with a
#: few large steps
SCAN_SETTINGS = ProfileSettings(
    reoptimize=False, initial_step=0.5, min_step=0.1, max_step=2.0, max_points=6
)

#: tolerance the cost of a scan is compared with between processes. A worker
#: integrates on a fresh roadrunner instance while the serial scans reuse one,
#: so a cost differs by about the relative tolerance of the integrator, `1e-6`
#: in `fit_settings`; this is that with headroom. It still separates the scans:
#: a scan which took another path differs by the threshold of the test, i.e.
#: `1.92` in the cost, not by a millionth of it
COST_RTOL = 1e-3


def _profile(
    values: list[float], costs: list[float], index_optimum: int
) -> ParameterProfile:
    """Create a profile of a single parameter from values and costs."""
    return ParameterProfile(
        pid="p",
        values=np.array(values),
        costs=np.array(costs),
        paths=np.array(values).reshape(-1, 1),
        converged=np.ones(len(values), dtype=bool),
        index_optimum=index_optimum,
    )


# ---------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------
def test_cost_threshold() -> None:
    """The threshold is half the chi-square quantile above the minimal cost."""
    # 3.84 on -2 log L, i.e., 1.92 on the cost 0.5 * sum(r**2)
    assert cost_threshold(cost=1.0) == pytest.approx(1.0 + 3.841459 / 2, rel=1e-5)
    assert cost_threshold(cost=0.0, alpha=0.95, df=3) == pytest.approx(
        chi2.ppf(0.95, df=3) / 2
    )


def test_profile_settings_threshold() -> None:
    """The settings give the same threshold."""
    settings = ProfileSettings(alpha=0.9, degrees_of_freedom=2)
    assert settings.delta == pytest.approx(chi2.ppf(0.9, df=2))
    assert settings.threshold(2.0) == pytest.approx(2.0 + chi2.ppf(0.9, df=2) / 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 1.0},
        {"degrees_of_freedom": 0},
        {"min_step": 0.5, "initial_step": 0.1},
        {"initial_step": 2.0, "max_step": 1.0},
        {"step_factor": 1.0},
        {"max_cost_fraction": 0.0},
        {"max_points": 0},
        {"flatness": 1.0},
    ],
)
def test_profile_settings_invalid(kwargs: dict) -> None:
    """Settings out of their range are reported."""
    with pytest.raises(ValueError):
        ProfileSettings(**kwargs)


def test_profile_settings_round_trip() -> None:
    """The settings survive the round trip through a dictionary."""
    settings = ProfileSettings(alpha=0.9, max_points=7, reoptimize=False)
    assert ProfileSettings.from_dict(settings.to_dict()) == settings


# ---------------------------------------------------------------------------
# classification of synthetic profiles
# ---------------------------------------------------------------------------
def test_profile_identifiable() -> None:
    """A profile which crosses the threshold on both sides is identifiable."""
    profile = _profile(
        values=[0.1, 1.0 / 3.0, 1.0, 3.0, 10.0],
        costs=[5.0, 2.0, 1.0, 2.0, 5.0],
        index_optimum=2,
    )
    profile.evaluate(threshold=3.0, flatness_cost=0.1)
    assert profile.identifiability is Identifiability.IDENTIFIABLE
    assert profile.ci_lower is not None and 0.1 < profile.ci_lower < 1.0 / 3.0
    assert profile.ci_upper is not None and 3.0 < profile.ci_upper < 10.0
    # the crossing is interpolated in logarithmic space, symmetric here
    assert profile.ci_lower * profile.ci_upper == pytest.approx(1.0)


def test_profile_crossing_interpolation() -> None:
    """The crossing is interpolated linearly in the logarithm of the parameter."""
    profile = _profile(values=[1.0, 100.0], costs=[1.0, 3.0], index_optimum=0)
    # halfway in the cost is halfway in log10, i.e., at 10
    assert profile.crossing(threshold=2.0, direction=+1) == pytest.approx(10.0)
    assert profile.crossing(threshold=2.0, direction=-1) is None


def test_profile_non_identifiable_lower() -> None:
    """A profile which stays below the threshold towards small values is open."""
    profile = _profile(
        values=[0.01, 0.1, 1.0, 10.0], costs=[1.2, 1.1, 1.0, 5.0], index_optimum=2
    )
    profile.evaluate(threshold=3.0, flatness_cost=0.05)
    assert profile.identifiability is Identifiability.NON_IDENTIFIABLE_LOWER
    assert profile.ci_lower is None
    assert profile.ci_upper is not None


def test_profile_non_identifiable_upper() -> None:
    """A profile which stays below the threshold towards large values is open."""
    profile = _profile(
        values=[0.01, 1.0, 10.0, 100.0], costs=[5.0, 1.0, 1.5, 1.6], index_optimum=1
    )
    profile.evaluate(threshold=3.0, flatness_cost=0.05)
    assert profile.identifiability is Identifiability.NON_IDENTIFIABLE_UPPER
    assert profile.ci_lower is not None
    assert profile.ci_upper is None


def test_profile_non_identifiable_both() -> None:
    """A profile with a minimum which stays below the threshold on both sides."""
    profile = _profile(
        values=[0.01, 0.1, 1.0, 10.0, 100.0],
        costs=[2.0, 1.5, 1.0, 1.5, 2.0],
        index_optimum=2,
    )
    profile.evaluate(threshold=3.0, flatness_cost=0.05)
    assert profile.identifiability is Identifiability.NON_IDENTIFIABLE
    assert profile.ci_lower is None and profile.ci_upper is None


def test_profile_structural() -> None:
    """A flat profile is structurally non-identifiable."""
    profile = _profile(
        values=[0.01, 0.1, 1.0, 10.0, 100.0],
        costs=[1.0, 1.001, 1.0, 1.0, 1.002],
        index_optimum=2,
    )
    profile.evaluate(threshold=3.0, flatness_cost=0.05)
    assert profile.identifiability is Identifiability.STRUCTURAL
    assert profile.identifiability.label == "structurally non-identifiable"
    assert not profile.identifiability.is_identifiable


def test_profile_inconsistent_points() -> None:
    """Inconsistent points are reported."""
    with pytest.raises(ValueError, match="inconsistent"):
        ParameterProfile(
            pid="p",
            values=np.array([1.0, 2.0]),
            costs=np.array([1.0]),
            paths=np.array([[1.0], [2.0]]),
            converged=np.array([True, True]),
            index_optimum=0,
        )
    with pytest.raises(ValueError, match="ascending"):
        _profile(values=[2.0, 1.0], costs=[1.0, 1.0], index_optimum=0)


# ---------------------------------------------------------------------------
# the analysis of the reference problem
# ---------------------------------------------------------------------------
def _result_of_scan(
    problem: OptimizationProblem, settings: FitSettings
) -> IdentifiabilityResult:
    """Run the fast scan of the model parameters of the problem."""
    problem.initialize(settings)
    return profile_likelihood(
        problem=problem,
        settings=settings,
        parameter_set=problem.parameter_set_model(),
        profile_settings=SCAN_SETTINGS,
        serial=True,
        show_progress=False,
    )


def test_scan_of_reference_problem(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The scans profile every parameter and classify it."""
    result = _result_of_scan(op_hctz_pkiv, fit_settings)

    assert set(result.profiles) == set(op_hctz_pkiv.pids)
    assert result.threshold == pytest.approx(result.cost_min + 1.9207, abs=1e-3)
    for pid, profile in result.profiles.items():
        assert profile.pid == pid
        assert profile.identifiability is not None
        # the optimum is a point of the profile
        assert profile.value_optimum == pytest.approx(result.parameter_set.values[pid])
        assert profile.cost_optimum == pytest.approx(result.cost)
        assert profile.paths.shape == (len(profile), len(op_hctz_pkiv.pids))
        # the scan stops at a bound, at the threshold or after max_points
        assert len(profile) <= 2 * SCAN_SETTINGS.max_points + 1
        # without re-optimization the other parameters stay at the optimum
        others = [k for k, p in enumerate(op_hctz_pkiv.pids) if p != pid]
        x_others = result.parameter_set.x(op_hctz_pkiv.pids)[others]
        assert np.allclose(profile.paths[:, others], x_others)

    # the absorption parameters are not observed by iv data, their profiles
    # are flat
    assert result.profiles["Ka_dis_hctz"].identifiability is Identifiability.STRUCTURAL
    assert result.n_identifiable <= len(result.profiles)

    df = result.summary_df()
    assert list(df.parameter) == op_hctz_pkiv.pids
    assert {"ci_lower", "ci_upper", "identifiability"} <= set(df.columns)
    assert "Identifiability" in result.report()


def test_scan_subset_of_parameters(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """Only the given parameters are profiled."""
    op_hctz_pkiv.initialize(fit_settings)
    result = profile_likelihood(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_set=op_hctz_pkiv.parameter_set_model(),
        profile_settings=SCAN_SETTINGS,
        pids=["KI__HCTZEX_k"],
        serial=True,
        show_progress=False,
    )
    assert list(result.profiles) == ["KI__HCTZEX_k"]
    assert result.pids == op_hctz_pkiv.pids


def test_unknown_parameter(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A parameter which is not a parameter of the problem is reported."""
    op_hctz_pkiv.initialize(fit_settings)
    with pytest.raises(KeyError, match="not parameters of the problem"):
        profile_likelihood(
            problem=op_hctz_pkiv,
            settings=fit_settings,
            parameter_set=op_hctz_pkiv.parameter_set_model(),
            pids=["unknown"],
            serial=True,
            show_progress=False,
        )


def test_parameter_set_outside_bounds(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A parameter set outside of the bounds of the problem is reported."""
    op_hctz_pkiv.initialize(fit_settings)
    pset = op_hctz_pkiv.parameter_set_model()
    pset.values["Ka_dis_hctz"] = 1000.0
    with pytest.raises(ValueError, match="outside of the bounds"):
        profile_likelihood(
            problem=op_hctz_pkiv,
            settings=fit_settings,
            parameter_set=pset,
            serial=True,
            show_progress=False,
        )


def test_profile_with_reoptimization(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The other parameters are optimized at every point of the profile."""
    op_hctz_pkiv.initialize(fit_settings)
    result = profile_likelihood(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_set=op_hctz_pkiv.parameter_set_model(),
        profile_settings=ProfileSettings(
            initial_step=0.5,
            min_step=0.1,
            max_step=2.0,
            max_points=2,
            optimizer_kwargs={"diff_step": 0.05, "max_nfev": 10},
        ),
        pids=["KI__HCTZEX_k"],
        serial=True,
        show_progress=False,
    )
    profile = result.profiles["KI__HCTZEX_k"]
    assert 1 < len(profile) <= 5
    assert profile.identifiability is not None
    # the profile is never above the plain scan at the same points, the other
    # parameters can only lower the cost
    assert profile.cost_min <= result.cost + 1e-9


def test_parallel_equals_serial(
    op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The scans of the workers give the result of the serial scans.

    The scans take the same path, i.e. the same parameter values, and their
    costs agree up to the integrator, see `COST_RTOL`.
    """
    serial = _result_of_scan(op_hctz_pkiv, fit_settings)
    parallel = profile_likelihood(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_set=op_hctz_pkiv.parameter_set_model(),
        profile_settings=SCAN_SETTINGS,
        n_cores=2,
        show_progress=False,
    )
    for pid in op_hctz_pkiv.pids:
        values, other = serial.profiles[pid].values, parallel.profiles[pid].values
        # the same number of points, i.e. the adaptive steps did the same
        assert values.shape == other.shape, pid
        assert np.allclose(values, other)
        assert np.allclose(
            serial.profiles[pid].costs,
            parallel.profiles[pid].costs,
            rtol=COST_RTOL,
        )


def test_result_json_round_trip(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The result survives the round trip through JSON."""
    result = _result_of_scan(op_hctz_pkiv, fit_settings)
    path = result.to_json(tmp_path / "identifiability.json")
    loaded = IdentifiabilityResult.from_json(path)

    assert loaded.opid == result.opid
    assert loaded.settings == result.settings
    assert loaded.fit_settings == result.fit_settings
    assert loaded.cost == result.cost
    assert loaded.threshold == pytest.approx(result.threshold)
    assert [p.pid for p in loaded.parameters] == result.pids
    for pid, profile in result.profiles.items():
        other = loaded.profiles[pid]
        assert np.allclose(other.values, profile.values)
        assert np.allclose(other.costs, profile.costs)
        assert np.allclose(other.paths, profile.paths)
        assert other.index_optimum == profile.index_optimum
        assert other.identifiability is profile.identifiability
        assert other.ci_lower == profile.ci_lower
        assert other.ci_upper == profile.ci_upper
    pd.testing.assert_frame_equal(loaded.summary_df(), result.summary_df())


def test_plots(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The overview and the profile of a parameter are written."""
    result = _result_of_scan(op_hctz_pkiv, fit_settings)
    plot_profiles(result, path=tmp_path / "profiles.svg")
    plot_profile(result, pid="KI__HCTZEX_k", path=tmp_path / "profile.svg")
    assert (tmp_path / "profiles.svg").exists()
    assert (tmp_path / "profile.svg").exists()


# ---------------------------------------------------------------------------
# report and command line
# ---------------------------------------------------------------------------
def test_report_with_identifiability(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report with an analysis has the identifiability section and files."""
    result = _result_of_scan(op_hctz_pkiv, fit_settings)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=ParameterSets([result.parameter_set]),
        identifiability=result,
    )
    results_dir = report.create(output_dir=tmp_path, name="report")

    assert (results_dir / "identifiability.json").exists()
    assert (results_dir / "identifiability.tsv").exists()
    assert (results_dir / "plots" / "profiles.svg").exists()
    for pid in op_hctz_pkiv.pids:
        assert (results_dir / "plots" / f"profile_{pid}.svg").exists()
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert 'id="identifiability"' in html
    assert "structurally non-identifiable" in html
    assert "Identifiability" in (results_dir / "report.txt").read_text(encoding="utf-8")


def test_report_without_identifiability(
    tmp_path: Path, op_hctz_pkiv: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """A report without an analysis has no identifiability section."""
    op_hctz_pkiv.initialize(fit_settings)
    report = FitReport(
        problem=op_hctz_pkiv,
        settings=fit_settings,
        parameter_sets=op_hctz_pkiv.parameter_set_model(),
    )
    results_dir = report.create(output_dir=tmp_path, name="report")
    assert not (results_dir / "identifiability.json").exists()
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert 'id="identifiability"' not in html


def test_identifiability_cli(
    tmp_path: Path, definition_hctz_pkiv: FitDefinition
) -> None:
    """The command line tool profiles stored parameters and reports them."""
    problem = definition_hctz_pkiv.problem(opid="PKIV")
    problem.initialize(definition_hctz_pkiv.settings)
    parameters_path = tmp_path / "parameters.json"
    ParameterSets([problem.parameter_set_model()]).to_json(path=parameters_path)

    results_dir = identifiability_cli(
        {"PKIV": definition_hctz_pkiv},
        args=[
            str(parameters_path),
            "--subset=PKIV",
            "--name=identifiability",
            "--no-reoptimize",
            "--max-points=3",
            "--initial-step=1.0",
            "--min-step=0.5",
            "--max-step=2.0",
            f"--output_dir={tmp_path}",
        ],
    )
    assert results_dir == tmp_path / "identifiability"
    assert (results_dir / "identifiability.json").exists()
    assert (results_dir / "index.html").exists()
    result = IdentifiabilityResult.from_json(results_dir / "identifiability.json")
    assert set(result.profiles) == set(problem.pids)
    assert not result.settings.reoptimize

    # the Fisher information comes with the profiles
    assert (results_dir / "fisher.json").exists()
    assert (results_dir / "fisher.tsv").exists()
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert "Fisher information" in html


def test_identifiability_cli_without_fisher(
    tmp_path: Path, definition_hctz_pkiv: FitDefinition
) -> None:
    """`--no-fisher` reports the profiles alone."""
    problem = definition_hctz_pkiv.problem(opid="PKIV")
    problem.initialize(definition_hctz_pkiv.settings)
    parameters_path = tmp_path / "parameters.json"
    ParameterSets([problem.parameter_set_model()]).to_json(path=parameters_path)

    results_dir = identifiability_cli(
        {"PKIV": definition_hctz_pkiv},
        args=[
            str(parameters_path),
            "--subset=PKIV",
            "--name=identifiability",
            "--no-reoptimize",
            "--no-fisher",
            "--max-points=3",
            "--initial-step=1.0",
            "--min-step=0.5",
            "--max-step=2.0",
            f"--output_dir={tmp_path}",
        ],
    )
    assert (results_dir / "identifiability.json").exists()
    assert not (results_dir / "fisher.json").exists()
    html = (results_dir / "index.html").read_text(encoding="utf-8")
    assert "Fisher information" not in html


def test_parameter_set_of_result() -> None:
    """A result is created by hand from profiles."""
    parameters = [FitParameter("p", 1.0, 0.01, 100.0, "1/min")]
    profile = _profile(values=[0.1, 1.0, 10.0], costs=[3.0, 1.0, 3.0], index_optimum=1)
    result = IdentifiabilityResult(
        opid="test",
        parameter_set=ParameterSet(sid="fit", values={"p": 1.0}),
        parameters=parameters,
        settings=ProfileSettings(),
        fit_settings=FitSettings(),
        cost=1.0,
        profiles={"p": profile},
    )
    assert result.n_identifiable == 1
    assert not result.better_optimum
    assert result.summary_df().loc[0, "identifiability"] == "identifiable"
