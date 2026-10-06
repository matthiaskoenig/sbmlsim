"""Test the example analysis."""

from pathlib import Path

import pytest

from examples.sensitivity.sensitivity_example import (
    sensitivity_groups,
    sensitivity_parameters,
    sensitivity_simulation,
)
from sbmlsim.sensitivity import (
    FASTSensitivityAnalysis,
    LocalSensitivityAnalysis,
    MorrisSensitivityAnalysis,
    SamplingSensitivityAnalysis,
    SobolSensitivityAnalysis,
)

#: the output of an analysis is pristine, a warning of a library is a failure
pytestmark = pytest.mark.filterwarnings("error")


def test_sampling_sensitivity_analysis(tmp_path: Path):
    """Test sampling sensitivity analysis."""

    sa = SamplingSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=[sensitivity_groups[0]],
        results_path=tmp_path,
        N=5,
        cache_results=False,
        n_cores=1,
        seed=1234,
    )
    assert sa

    # run simulation
    sa.execute()
    assert sa.samples is not None
    assert sa.results is not None
    assert sa.sensitivity is not None

    # execute plots
    sa.plot()


@pytest.mark.parametrize(
    ("analysis", "kwargs"),
    [
        (SamplingSensitivityAnalysis, {"N": 5}),
        (SobolSensitivityAnalysis, {"N": 4}),
        (FASTSensitivityAnalysis, {"N": 65}),
        (
            MorrisSensitivityAnalysis,
            {"N": 4, "num_levels": 4, "optimal_trajectories": 2},
        ),
    ],
)
def test_the_seed_decides_the_samples(tmp_path: Path, analysis: type, kwargs: dict):
    """Two analyses with the same seed draw the same samples."""

    def samples(name: str) -> dict:
        sa = analysis(
            sensitivity_simulation=sensitivity_simulation,
            parameters=sensitivity_parameters,
            groups=[sensitivity_groups[0]],
            results_path=tmp_path / name,
            cache_results=False,
            n_cores=1,
            seed=1234,
            **kwargs,
        )
        sa.create_samples()
        return sa.samples

    first, second = samples("first"), samples("second")
    for uid, data in first.items():
        assert data.equals(second[uid]), uid


def test_the_samples_are_simulated_in_parallel_as_in_one_process(tmp_path: Path):
    """Two processes simulate what a single core simulates, for every group."""

    def results(n_cores: int) -> dict:
        sa = SamplingSensitivityAnalysis(
            sensitivity_simulation=sensitivity_simulation,
            parameters=sensitivity_parameters,
            groups=sensitivity_groups,
            results_path=tmp_path / str(n_cores),
            N=5,
            cache_results=False,
            n_cores=n_cores,
            seed=1234,
        )
        sa.execute()
        return sa.results

    serial = results(n_cores=1)
    parallel = results(n_cores=2)
    assert list(parallel) == [group.uid for group in sensitivity_groups]
    for uid, data in serial.items():
        assert data.equals(parallel[uid]), uid


def test_local_sensitivity_analysis(tmp_path: Path):
    """Test local sensitivity analysis."""

    sa = LocalSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=[sensitivity_groups[0]],
        results_path=tmp_path,
        difference=0.01,
        cache_results=False,
        n_cores=1,
        seed=1234,
    )
    assert sa

    # run simulation
    sa.execute()
    assert sa.samples is not None
    assert sa.results is not None
    assert sa.sensitivity is not None

    # execute plots
    sa.plot()


def test_sobol_sensitivity_analysis(tmp_path: Path):
    """Test sobol sensitivity analysis."""

    sa = SobolSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=[sensitivity_groups[0]],
        results_path=tmp_path,
        N=4,
        cache_results=False,
        n_cores=1,
        seed=1234,
    )
    assert sa

    # run simulation
    sa.execute()
    assert sa.samples is not None
    assert sa.results is not None
    assert sa.sensitivity is not None

    # execute plots
    sa.plot()


def test_fast_sensitivity_analysis(tmp_path: Path):
    """Test fast sensitivity analysis."""

    sa = FASTSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=[sensitivity_groups[0]],
        results_path=tmp_path,
        N=100,
        cache_results=False,
        n_cores=1,
        seed=1234,
    )
    assert sa

    # run simulation
    sa.execute()
    assert sa.samples is not None
    assert sa.results is not None
    assert sa.sensitivity is not None

    # execute plots
    sa.plot()


def test_morris_sensitivity_analysis(tmp_path: Path):
    """Test morris sensitivity analysis."""

    sa = MorrisSensitivityAnalysis(
        sensitivity_simulation=sensitivity_simulation,
        parameters=sensitivity_parameters,
        groups=[sensitivity_groups[0]],
        results_path=tmp_path,
        N=4,
        num_levels=2,
        optimal_trajectories=2,
        cache_results=False,
        n_cores=1,
        seed=1234,
    )

    assert sa

    # run simulation
    sa.execute()
    assert sa.samples is not None
    assert sa.results is not None
    assert sa.sensitivity is not None

    # execute plots
    sa.plot()
