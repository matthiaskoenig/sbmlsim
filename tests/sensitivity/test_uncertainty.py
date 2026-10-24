"""The uncertainty analysis: bands and distributions."""

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import stats

from sbmlsim.result.scan import ScanResult
from sbmlsim.sensitivity.uncertainty import plot_bands, plot_distribution
from sbmlsim.simulation import Change, Dimension, Formula, Scan, Simulation
from sbmlsim.simulation.sampling import LogNormal, random
from sbmlsim.simulator import Simulator
from sbmlsim.units import unit_label
from tests.simulator.models import sbml, sbml_pk


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
    patches = figure.axes[0].patches
    assert len(patches) > 0
    # the labels share the bin edges, so every bar has one width
    widths = [p.get_width() for p in patches if isinstance(p, Rectangle)]
    assert len(widths) == len(patches) and np.allclose(widths, widths[0])
    box = plot_distribution(result, "k", dim="random", kind="box")
    assert isinstance(box, Figure)
    with pytest.raises(ValueError, match="kind"):
        plot_distribution(result, "k", dim="random", kind="violin")
    with pytest.raises(ValueError, match="time"):
        plot_distribution(result, "rate", dim="random")


@pytest.fixture(scope="module")
def volumes() -> dict[str, ScanResult]:
    """Get a scan over draws of the elimination per volume, in ng/ml."""
    draws = random({"ke": LogNormal(0.2, 0.3)}, 40, seed=1)
    observables = [
        Formula("cp", "[C]", unit="ng/ml"),
        Formula("cmax", "max(cp)", unit="ng/ml"),
    ]
    return {
        name: Simulator(n_workers=1).run(
            sbml_pk(),
            Scan(
                Simulation(
                    end=24,
                    steps=24,
                    preinit_changes={"V": volume},
                    changes=[Change(0, {"PODOSE": 100.0})],
                ),
                [draws],
            ),
            observables,
        )
        for name, volume in (("small", 10.0), ("large", 20.0))
    }


def test_a_unit_is_written_only_when_there_is_one() -> None:
    assert unit_label("k", "1/s") == "k [1/s]"
    assert unit_label("k", "") == "k"
    assert unit_label("k", None) == "k"
    assert unit_label("k", "dimensionless") == "k"


def test_a_unit_is_written_in_its_short_form() -> None:
    assert unit_label("cp", "nanogram / milliliter") == "cp [ng/ml]"
    assert unit_label("time", "hour") == "time [h]"
    assert unit_label("k", "1 / hour") == "k [1/h]"
    # a unit pint does not parse is written as given
    assert unit_label("n", "item/(31557600.0*s)") == "n [item/(31557600.0*s)]"
    assert unit_label("x", "IU_insulin") == "x [IU_insulin]"


def test_the_axes_have_the_short_units(volumes: dict[str, ScanResult]) -> None:
    summary = volumes["small"].summary("random", quantiles=[0.05, 0.5, 0.95])
    (ax,) = plot_bands(summary, "cp").axes
    assert ax.get_ylabel() == "cp [ng/ml]" and ax.get_xlabel() == "time [h]"
    (ax,) = plot_distribution(volumes["small"], "cmax", dim="random").axes
    assert ax.get_xlabel() == "cmax [ng/ml]"
    (ax,) = plot_distribution(volumes["small"], "cmax", dim="random", kind="box").axes
    assert ax.get_ylabel() == "cmax [ng/ml]"


def _legend(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


def test_plot_bands_labels_a_summary_per_call(volumes: dict[str, ScanResult]) -> None:
    """Every call on one axes adds its line with its label to the legend."""
    figure = Figure()
    ax = figure.add_subplot()
    for name, result in volumes.items():
        summary = result.summary("random", quantiles=[0.05, 0.5, 0.95])
        plot_bands(summary, "cp", ax=ax, label=name)
    assert _legend(ax) == ["small", "large"]
    # the color of a band is the one of its line
    for line, band in zip(ax.lines, ax.collections, strict=True):
        face = np.asarray(band.get_facecolor(), dtype=float).reshape(-1, 4)[0]
        assert np.allclose(to_rgba(line.get_color())[:3], face[:3])
    # a single call with a label has its legend
    summary = volumes["small"].summary("random", quantiles=[0.05, 0.5, 0.95])
    (ax,) = plot_bands(summary, "cp", label="small").axes
    assert _legend(ax) == ["small"]


def test_plot_bands_prefixes_the_labels_of_other_dimensions(
    result: ScanResult,
) -> None:
    summary = result.summary("random", quantiles=[0.05, 0.5, 0.95])
    (ax,) = plot_bands(summary, "rate", label="mid").axes
    assert _legend(ax) == ["mid: a=0", "mid: a=1"]
    (ax,) = plot_bands(summary, "rate").axes
    assert _legend(ax) == ["a=0", "a=1"]


def test_plot_distribution_labels_a_result_per_call(
    volumes: dict[str, ScanResult],
) -> None:
    figure = Figure()
    ax = figure.add_subplot()
    for name, result in volumes.items():
        plot_distribution(result, "cmax", dim="random", ax=ax, label=name)
    assert _legend(ax) == ["small", "large"]
    (ax,) = plot_distribution(
        volumes["small"], "cmax", dim="random", kind="box", label="small"
    ).axes
    assert [t.get_text() for t in ax.get_xticklabels()] == ["small"]


def test_plot_bands_draws_quantities_in_the_units_of_the_axes(
    volumes: dict[str, ScanResult],
) -> None:
    """A band converts into the units of an axis, its own or the ones asked for."""
    summary = volumes["small"].summary("random", quantiles=[0.05, 0.5, 0.95])
    figure = plot_bands(summary, "cp", time_unit="min", unit="µg/l", name="C")
    (ax,) = figure.axes
    # pint writes micro as the greek letter mu
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("time [min]", "C [\u03bcg/l]")
    hours = summary.q["time"].to("hr").magnitude
    np.testing.assert_allclose(np.asarray(ax.lines[0].get_xydata())[:, 0], 60 * hours)
    # a second band on the axes is converted into their units
    other = volumes["large"].summary("random", quantiles=[0.05, 0.5, 0.95])
    plot_bands(other, "cp", ax=ax, label="large")
    np.testing.assert_allclose(
        np.asarray(ax.lines[1].get_xydata())[:, 0], 60 * hours, rtol=1e-12
    )
    assert ax.get_ylabel() == "cp [\u03bcg/l]"


def test_plot_distribution_converts_into_a_unit(
    volumes: dict[str, ScanResult],
) -> None:
    """The values of a distribution are drawn in the unit asked for."""
    figure = plot_distribution(volumes["small"], "cmax", dim="random", unit="µg/ml")
    ax = figure.axes[0]
    assert ax.get_xlabel() == "cmax [\u03bcg/ml]"
    cmax = volumes["small"].q["cmax"].to("µg/ml").magnitude
    edges = [p.get_x() for p in ax.patches if isinstance(p, Rectangle)]
    assert min(edges) == pytest.approx(np.nanmin(cmax))
