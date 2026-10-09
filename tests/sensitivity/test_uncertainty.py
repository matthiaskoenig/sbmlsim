"""The uncertainty analysis: bands and distributions."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy import stats

from sbmlsim.result.scan import ScanResult
from sbmlsim.sensitivity.uncertainty import plot_bands, plot_distribution
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulation.sampling import LogNormal, random
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml


@pytest.fixture(scope="module")
def result() -> ScanResult:
    draws = random({"k1": LogNormal(0.8, 0.3)}, 2000, seed=1)
    doses = Dimension("a", values={"a0": [1.0, 2.0]})
    observables = [Formula("rate", "k1 * a0 + 0 * time"), Formula("k", "max(k1)")]
    scan = Scan(Simulation(end=1, steps=4), [draws, doses])
    return Simulator(n_workers=1).run(sbml(), scan, observables)


def test_the_bands_of_a_lognormal_parameter_are_its_quantiles(
    result: ScanResult,
) -> None:
    summary = result.summary("random", quantiles=[0.05, 0.5, 0.95])
    sigma = np.sqrt(np.log(1.0 + 0.3**2))
    expected = stats.lognorm(sigma, scale=0.8).ppf([0.05, 0.5, 0.95])
    # the first label of the dimension a is a0 = 1, so the rate is k1
    band = summary["rate"].isel(a=0, time=0)
    np.testing.assert_allclose(
        band.sel(statistic=["q0.05", "q0.5", "q0.95"]).values, expected, rtol=0.05
    )


def test_plot_bands_draws_a_band_per_label(result: ScanResult) -> None:
    summary = result.summary("random", quantiles=[0.05, 0.5, 0.95])
    figure = plot_bands(summary, "rate")
    assert isinstance(figure, Figure)
    (ax,) = figure.axes
    assert len(ax.collections) == 2 and len(ax.lines) == 2
    assert ax.get_xlabel().startswith("time")


def test_plot_distribution_per_label(result: ScanResult) -> None:
    figure = plot_distribution(result, "k", dim="random")
    assert isinstance(figure, Figure)
    assert len(figure.axes[0].patches) > 0
    box = plot_distribution(result, "k", dim="random", kind="box")
    assert isinstance(box, Figure)
    with pytest.raises(ValueError, match="kind"):
        plot_distribution(result, "k", dim="random", kind="violin")
    with pytest.raises(ValueError, match="time"):
        plot_distribution(result, "rate", dim="random")
