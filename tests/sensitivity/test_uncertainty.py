"""The uncertainty analysis: bands and distributions, charts of the renderer."""

from typing import Any

import numpy as np
import pytest
from scipy import stats

from sbmlsim.plot.chart import Chart
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
    observables = [Formula("rate", "k1 * a0"), Formula("k", "max(k1)")]
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


def _plot(chart: Chart) -> dict[str, Any]:
    (plot,) = chart.spec()["plots"]
    return plot


def _elements(chart: Chart, kind: str) -> list[dict[str, Any]]:
    return [e for e in _plot(chart)["elements"] if e["kind"] == kind]


def _labels(chart: Chart) -> list[str]:
    assert _plot(chart)["legend"]
    return [e["label"] for e in _plot(chart)["elements"] if "label" in e]


def test_plot_bands_draws_a_band_per_label(result: ScanResult) -> None:
    summary = result.summary("random", quantiles=[0.05, 0.5, 0.95])
    chart = plot_bands(summary, "rate")
    assert isinstance(chart, Chart)
    bands = _elements(chart, "band")
    assert len(bands) == 2 and all("median" in b for b in bands)
    band = summary["rate"].isel(a=1)
    np.testing.assert_allclose(bands[1]["low"], band.sel(statistic="q0.05").values)
    np.testing.assert_allclose(bands[1]["median"], band.sel(statistic="q0.5").values)
    assert _plot(chart)["x"]["label"] == "time"
    chart.svg()


def test_plot_distribution_per_label(result: ScanResult) -> None:
    chart = plot_distribution(result, "k", dim="random")
    assert isinstance(chart, Chart)
    histograms = _elements(chart, "histogram")
    # the labels share the bin edges
    assert len(histograms) == 2 and len(histograms[0]["edges"]) == 31
    assert histograms[0]["edges"] == histograms[1]["edges"]
    assert sum(histograms[0]["counts"]) == 2000
    chart.svg()
    chart = plot_distribution(result, "k", dim="random", kind="box")
    # a box per label on a category axis of the labels
    (box,) = _elements(chart, "box")
    assert _plot(chart)["x"]["categories"] == ["a=0", "a=1"]
    assert box["positions"] == [0.0, 1.0]
    sample = result["k"].isel(a=0).values
    assert box["stats"][0]["median"] == pytest.approx(np.median(sample))
    chart.svg()
    chart = plot_distribution(result, "k", dim="random", kind="violin")
    (violin,) = _elements(chart, "violin")
    density = violin["densities"][0]
    assert len(density["values"]) == len(density["density"]) == 128
    assert density["values"][0] == pytest.approx(sample.min())
    assert density["values"][-1] == pytest.approx(sample.max())
    kde = stats.gaussian_kde(sample)
    np.testing.assert_allclose(density["density"], kde(density["values"]))
    chart.svg()
    with pytest.raises(ValueError, match="kind"):
        plot_distribution(result, "k", dim="random", kind="strip")
    with pytest.raises(ValueError, match="time"):
        plot_distribution(result, "rate", dim="random")


def test_a_violin_needs_a_spread(volumes: dict[str, ScanResult]) -> None:
    """A sample of one value has no density and no violin, the others have one."""
    result = volumes["small"]
    constant = result.ds.assign(cmax=result.ds["cmax"] * 0 + 1.0)
    chart = plot_distribution(
        {"varied": result, "constant": ScanResult(constant)},
        "cmax",
        dim="random",
        kind="violin",
    )
    (violin,) = _elements(chart, "violin")
    assert violin["positions"] == [0.0]
    assert _plot(chart)["x"]["categories"] == ["varied", "constant"]
    chart.svg()


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
    plot = _plot(plot_bands(summary, "cp"))
    assert (plot["y"]["label"], plot["y"]["unit"]) == ("cp", "ng/ml")
    assert (plot["x"]["label"], plot["x"]["unit"]) == ("time", "h")
    plot = _plot(plot_distribution(volumes["small"], "cmax", dim="random"))
    assert (plot["x"]["label"], plot["x"]["unit"]) == ("cmax", "ng/ml")
    assert plot["y"]["label"] == "count"
    for kind in ("box", "violin"):
        plot = _plot(
            plot_distribution(volumes["small"], "cmax", dim="random", kind=kind)
        )
        assert (plot["y"]["label"], plot["y"]["unit"]) == ("cmax", "ng/ml")


def test_plot_bands_labels_a_summary_per_scenario(
    volumes: dict[str, ScanResult],
) -> None:
    """The summaries of a mapping are labelled by their keys in one legend."""
    summaries = {
        name: result.summary("random", quantiles=[0.05, 0.5, 0.95])
        for name, result in volumes.items()
    }
    chart = plot_bands(summaries, "cp")
    assert _labels(chart) == ["small", "large"]
    # a band is one element with its median, so it has the color of its line
    assert all("median" in band for band in _elements(chart, "band"))
    chart.svg()
    # a single summary with a label has its legend, one without none
    assert _labels(plot_bands(summaries["small"], "cp", label="small")) == ["small"]
    assert not _plot(plot_bands(summaries["small"], "cp"))["legend"]


def test_plot_bands_prefixes_the_labels_of_other_dimensions(
    result: ScanResult,
) -> None:
    summary = result.summary("random", quantiles=[0.05, 0.5, 0.95])
    assert _labels(plot_bands(summary, "rate", label="mid")) == [
        "mid: a=0",
        "mid: a=1",
    ]
    assert _labels(plot_bands({"mid": summary}, "rate")) == ["mid: a=0", "mid: a=1"]
    assert _labels(plot_bands(summary, "rate")) == ["a=0", "a=1"]


def test_plot_distribution_labels_a_result_per_scenario(
    volumes: dict[str, ScanResult],
) -> None:
    chart = plot_distribution(volumes, "cmax", dim="random")
    assert _labels(chart) == ["small", "large"]
    histograms = _elements(chart, "histogram")
    assert histograms[0]["edges"] == histograms[1]["edges"]
    chart = plot_distribution(
        volumes["small"], "cmax", dim="random", kind="box", label="small"
    )
    assert _plot(chart)["x"]["categories"] == ["small"]
    chart = plot_distribution(volumes, "cmax", dim="random", kind="box")
    assert _plot(chart)["x"]["categories"] == ["small", "large"]
    assert len(_elements(chart, "box")[0]["stats"]) == 2


def test_plot_bands_draws_quantities_in_the_units_of_the_axes(
    volumes: dict[str, ScanResult],
) -> None:
    """A band converts into the units of an axis, its own or the ones asked for."""
    summary = volumes["small"].summary("random", quantiles=[0.05, 0.5, 0.95])
    chart = plot_bands(summary, "cp", time_unit="min", unit="µg/l", name="C")
    plot = _plot(chart)
    # pint writes micro as the greek letter mu
    assert (plot["x"]["unit"], plot["y"]["unit"]) == ("min", "\u03bcg/l")
    assert plot["y"]["label"] == "C"
    hours = summary.quantity("time").to("hr").magnitude
    (band,) = _elements(chart, "band")
    np.testing.assert_allclose(band["x"], 60 * hours)
    cp = summary.sel(statistic="q0.5").quantity("cp").to("µg/l").magnitude
    np.testing.assert_allclose(band["median"], cp)
    # a second band is converted into the units of the axes
    other = volumes["large"].summary("random", quantiles=[0.05, 0.5, 0.95])
    chart = plot_bands({"small": summary, "large": other}, "cp", time_unit="min")
    _, second = _elements(chart, "band")
    np.testing.assert_allclose(second["x"], 60 * hours, rtol=1e-12)
    assert _plot(chart)["y"]["unit"] == "ng/ml"


def test_plot_distribution_converts_into_a_unit(
    volumes: dict[str, ScanResult],
) -> None:
    """The values of a distribution are drawn in the unit asked for."""
    chart = plot_distribution(volumes["small"], "cmax", dim="random", unit="µg/ml")
    plot = _plot(chart)
    assert (plot["x"]["label"], plot["x"]["unit"]) == ("cmax", "\u03bcg/ml")
    cmax = volumes["small"].quantity("cmax").to("µg/ml").magnitude
    (histogram,) = _elements(chart, "histogram")
    assert min(histogram["edges"]) == pytest.approx(np.nanmin(cmax))
