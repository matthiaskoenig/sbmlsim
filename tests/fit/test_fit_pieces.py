"""Pieces of a fit and of a profile likelihood merge to the whole run.

A fit runs given repeats of all its repeats, every one with the start value
and the seed it has in the whole fit, and a profile likelihood reports every
profile as it completes, so that a fit and an identifiability analysis can be
split into units and the merge of the units equals the whole run.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from sbmlsim.fit import FitSettings
from sbmlsim.fit.identifiability import (
    IdentifiabilityResult,
    ProfileSettings,
    profile_likelihood,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.result import OptimizationResult
from sbmlsim.fit.runner import run_optimization


def _fit(
    problem: OptimizationProblem,
    settings: FitSettings,
    short_fit: dict[str, Any],
    runs_dir: Path,
    indices: list[int] | None = None,
    n_cores: int = 1,
) -> OptimizationResult:
    """Run the fit of four repeats, or the given ones of them."""
    return run_optimization(
        problem=problem,
        settings=settings,
        size=4,
        seed=1,
        n_cores=n_cores,
        show_progress=False,
        runs_dir=runs_dir,
        indices=indices,
        **short_fit,
    )


def _run(directory: Path, k: int) -> OptimizeResult:
    """Read the fit of the stored repeat `k` of the problem `hctz_iv`."""
    (fit,) = OptimizationResult.from_json(directory / f"hctz_iv_run_{k}.json").fits
    return fit


def test_given_repeats_equal_the_repeats_of_the_whole_fit(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A repeat run on its own is the repeat of the whole fit, bitwise."""
    a, b = tmp_path / "a", tmp_path / "b"
    _fit(op_hctz_iv, fit_settings, short_fit, a, indices=[1, 3])
    _fit(op_hctz_iv, fit_settings, short_fit, b)

    for k in (1, 3):
        fit_a, fit_b = _run(a, k), _run(b, k)
        np.testing.assert_array_equal(fit_a.x, fit_b.x)
        np.testing.assert_array_equal(fit_a.x0, fit_b.x0)
        assert fit_a.cost == fit_b.cost
    assert not (a / "hctz_iv_run_0.json").exists()
    assert sorted(p.name for p in a.glob("*.json")) == [
        "hctz_iv_run_1.json",
        "hctz_iv_run_3.json",
    ]


def test_given_repeats_of_a_parallel_fit(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A parallel fit runs only the given repeats, the ones of a serial fit."""
    a, b = tmp_path / "a", tmp_path / "b"
    pooled = _fit(op_hctz_iv, fit_settings, short_fit, a, indices=[3, 1], n_cores=2)
    serial = _fit(op_hctz_iv, fit_settings, short_fit, b, indices=[1, 3])

    assert sorted(p.name for p in a.glob("*.json")) == [
        "hctz_iv_run_1.json",
        "hctz_iv_run_3.json",
    ]
    assert len(pooled.fits) == 2
    for k in (1, 3):
        np.testing.assert_array_equal(_run(a, k).x, _run(b, k).x)
        assert _run(a, k).cost == _run(b, k).cost
    assert sorted(fit.cost for fit in pooled.fits) == sorted(
        fit.cost for fit in serial.fits
    )


def test_pieces_merge_to_the_whole_fit(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The repeats of two pieces in one directory are the whole fit."""
    pieces = tmp_path / "pieces"
    _fit(op_hctz_iv, fit_settings, short_fit, pieces, indices=[0, 1])
    _fit(op_hctz_iv, fit_settings, short_fit, pieces, indices=[2, 3])
    whole = _fit(op_hctz_iv, fit_settings, short_fit, tmp_path / "whole")

    merged = OptimizationResult.from_directory(pieces)
    np.testing.assert_array_equal(
        merged.df_fits[["cost"]].to_numpy(), whole.df_fits[["cost"]].to_numpy()
    )
    np.testing.assert_array_equal(
        np.array([fit.x for fit in merged.fits]),
        np.array([fit.x for fit in whole.fits]),
    )


def test_from_directory_orders_by_run_index(
    op_hctz_iv: OptimizationProblem, tmp_path: Path
) -> None:
    """The runs are in the order of their index, `run_10` after `run_9`."""
    n = len(op_hctz_iv.parameters)
    for k in reversed(range(12)):
        OptimizationResult.write_run(
            directory=tmp_path,
            parameters=op_hctz_iv.parameters,
            fit=OptimizeResult(
                x=np.ones(n),
                x0=np.ones(n),
                cost=float(k),
                status=1,
                success=True,
                message="",
                nfev=1,
                njev=1,
                duration=0.0,
            ),
            trajectory=[float(k)],
            sid=f"p_run_{k}",
            opid="p",
        )

    result = OptimizationResult.from_directory(tmp_path)
    assert [fit.cost for fit in result.fits] == [float(k) for k in range(12)]


@pytest.mark.parametrize("indices", [[4], [1, 1], [-1]])
def test_indices_out_of_range_raise(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
    tmp_path: Path,
    indices: list[int],
) -> None:
    """An index outside of the repeats or a repeated index is refused."""
    with pytest.raises(ValueError, match="indices"):
        _fit(op_hctz_iv, fit_settings, short_fit, tmp_path, indices=indices)
    op_hctz_iv.initialize(fit_settings)
    with pytest.raises(ValueError, match="indices"):
        op_hctz_iv.optimize(size=4, seed=1, indices=indices, **short_fit)


def test_write_run_leaves_no_temporary_file(
    op_hctz_iv: OptimizationProblem, tmp_path: Path
) -> None:
    """The run is written atomically, nothing but its file is left."""
    n = len(op_hctz_iv.parameters)
    OptimizationResult.write_run(
        directory=tmp_path,
        parameters=op_hctz_iv.parameters,
        fit=OptimizeResult(
            x=np.ones(n), x0=np.ones(n), cost=1.0, success=True, duration=0.0
        ),
        trajectory=[1.0],
        sid="p_run_0",
        opid="p",
    )
    assert [p.name for p in tmp_path.iterdir()] == ["p_run_0.json"]


@pytest.fixture
def profiles(
    op_hctz_iv: OptimizationProblem,
    fit_settings: FitSettings,
    short_fit: dict[str, Any],
) -> tuple[OptimizationProblem, ParameterSet, list[str]]:
    """Get the problem, the parameter set of a short fit and two parameters."""
    result = run_optimization(
        problem=op_hctz_iv,
        settings=fit_settings,
        size=1,
        seed=1,
        n_cores=1,
        show_progress=False,
        **short_fit,
    )
    return op_hctz_iv, result.parameter_set(0), op_hctz_iv.pids[:2]


def _profile(
    problem: OptimizationProblem,
    settings: FitSettings,
    parameter_set: ParameterSet,
    pids: list[str],
    on_profile: Any = None,
    n_cores: int = 1,
) -> IdentifiabilityResult:
    """Profile the given parameters with two points per direction."""
    return profile_likelihood(
        problem=problem,
        settings=settings,
        parameter_set=parameter_set,
        profile_settings=ProfileSettings(max_points=2),
        pids=pids,
        n_cores=n_cores,
        serial=n_cores == 1,
        show_progress=False,
        on_profile=on_profile,
    )


def _without_duration(result: IdentifiabilityResult) -> dict[str, Any]:
    """Get the dictionary of a result without its duration."""
    d = result.to_dict()
    d.pop("duration")
    return d


def test_profiles_combine_to_the_profile_of_both(
    profiles: tuple[OptimizationProblem, ParameterSet, list[str]],
    fit_settings: FitSettings,
) -> None:
    """The profiles of one parameter each combine to the profile of both."""
    problem, parameter_set, (pid0, pid1) = profiles
    reported: list[IdentifiabilityResult] = []

    part0 = _profile(problem, fit_settings, parameter_set, [pid0])
    part1 = _profile(problem, fit_settings, parameter_set, [pid1])
    whole = _profile(
        problem, fit_settings, parameter_set, [pid0, pid1], on_profile=reported.append
    )

    # the scans left the optimum, the profiles are more than their optimum
    assert all(len(profile) > 1 for profile in whole.profiles.values())
    combined = IdentifiabilityResult.combine([part0, part1])
    assert list(combined.profiles) == [pid0, pid1]
    assert _without_duration(combined) == _without_duration(whole)
    assert combined.duration == part0.duration + part1.duration

    assert [list(part.profiles) for part in reported] == [[pid0], [pid1]]
    assert _without_duration(IdentifiabilityResult.combine(reported)) == (
        _without_duration(whole)
    )


def test_parallel_profiles_are_reported_in_the_calling_process(
    profiles: tuple[OptimizationProblem, ParameterSet, list[str]],
    fit_settings: FitSettings,
) -> None:
    """The profiles of the workers are reported, they combine to the result."""
    problem, parameter_set, pids = profiles
    reported: list[IdentifiabilityResult] = []
    pid_reported: list[int] = []

    def on_profile(part: IdentifiabilityResult) -> None:
        reported.append(part)
        pid_reported.append(os.getpid())

    whole = _profile(
        problem, fit_settings, parameter_set, pids, on_profile=on_profile, n_cores=2
    )
    assert sorted(pid for part in reported for pid in part.profiles) == sorted(pids)
    assert set(pid_reported) == {os.getpid()}
    combined = IdentifiabilityResult.combine(
        sorted(reported, key=lambda part: pids.index(next(iter(part.profiles))))
    )
    assert _without_duration(combined) == _without_duration(whole)


def test_combine_refuses_a_parameter_twice(
    profiles: tuple[OptimizationProblem, ParameterSet, list[str]],
    fit_settings: FitSettings,
) -> None:
    """A parameter profiled in two parts is refused, as are no parts."""
    problem, parameter_set, (pid0, _) = profiles
    part = _profile(problem, fit_settings, parameter_set, [pid0])
    with pytest.raises(ValueError, match="twice"):
        IdentifiabilityResult.combine([part, part])
    with pytest.raises(ValueError, match="no part"):
        IdentifiabilityResult.combine([])
