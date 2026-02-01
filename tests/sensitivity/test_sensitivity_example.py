"""Test the example analysis."""

from pathlib import Path

from sbmlsim.sensitivity import (
    SobolSensitivityAnalysis,
    LocalSensitivityAnalysis,
    SamplingSensitivityAnalysis,
    FASTSensitivityAnalysis,
    MorrisSensitivityAnalysis,
)
from sbmlsim.sensitivity.example.sensitivity_example import (
    sensitivity_simulation,
    sensitivity_parameters,
    sensitivity_groups,
)


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
