"""The band of the draws of the uncertainty of a panel, in the service and in a bundle.

The HCTZ model of the other tests carries no distribution of its parameters, so
these tests write one: a copy of its SBML whose absorption rate `Ka_dis_hctz`
has the distrib uncertainty of a fit, a lognormal of its own value as the
median, which is what the factory of a model writes from an adopted fit
(`sbmlsim.fit.records.AdoptedRecords.parameter`). The panel of the tests
declares a sample of four draws of it, so a request for two runs a prefix of
that sample.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import libsbml
import numpy as np
import pytest

from examples.panels.hctz import HCTZ_RENAL_PLASMA, PROJECT
from sbmlsim.panel import Panel, PanelUncertainty
from sbmlsim.plot.points import nan_quantile
from sbmlsim.project import Project
from sbmlsim.serve.runner import (
    DrawsError,
    ModelEntry,
    ModelRegistry,
    PanelRunner,
    check_draws,
)
from sbmlsim.simulation.sampling.model_distributions import (
    uncertainty_dimension,
    uncertainty_name,
)
from tests.serve.conftest import SIMULATE

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

#: the parameter of the HCTZ model whose fit the band is drawn from, the rate
#: of the absorption of the dose, which every curve of the panel follows
PARAMETER = "Ka_dis_hctz"
#: the name of the uncertainties of that model, the fit they come from
FIT = "fit HCTZ_1"
#: the uncertainty the panel of these tests declares
UNCERTAINTY = PanelUncertainty(draws=2, max_draws=4, level=0.8, seed=0)


def with_distribution(source: Path, path: Path) -> Path:
    """Write the model of a source with a distrib distribution of `PARAMETER`.

    The distribution is the lognormal of the value of the parameter as its
    median and a coefficient of variation of about 20 %, truncated to the
    bounds a fit would carry, as `sbmlsim.distrib` writes it.
    """
    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(source))
    document.enablePackage(libsbml.DistribExtension.getXmlnsL3V1V1(), "distrib", True)
    document.setPackageRequired("distrib", False)
    model: libsbml.Model = document.getModel()
    parameter: libsbml.Parameter = model.getParameter(PARAMETER)
    plugin = parameter.getPlugin("distrib")
    uncertainty = plugin.createUncertainty()
    assert uncertainty.setName(FIT) == libsbml.LIBSBML_OPERATION_SUCCESS
    child = uncertainty.createUncertParameter()
    child.setType(libsbml.DISTRIB_UNCERTTYPE_DISTRIBUTION)
    value = parameter.getValue()
    formula = f"lognormal({math.log(value)}, 0.2, {value / 4}, {value * 4})"
    assert child.setMath(libsbml.parseL3Formula(formula)) == (
        libsbml.LIBSBML_OPERATION_SUCCESS
    )
    assert libsbml.writeSBMLToFile(document, str(path)) == 1
    return path


@pytest.fixture(scope="module")
def sbml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The SBML of the HCTZ model whose absorption rate carries a distribution."""
    return with_distribution(
        PROJECT.variant_path("default"),
        tmp_path_factory.mktemp("uncertain") / "hctz_body_flat.xml",
    )


@pytest.fixture(scope="module")
def panel() -> Panel:
    """The plasma panel of the HCTZ model, which declares a band of four draws."""
    simulation = HCTZ_RENAL_PLASMA.simulation.model_copy(
        update={"uncertainty": UNCERTAINTY}
    )
    return HCTZ_RENAL_PLASMA.model_copy(update={"simulation": simulation})


@pytest.fixture(scope="module")
def project(sbml: Path, panel: Panel) -> Project:
    """The project of the panel, on the model which carries the distribution."""
    return PROJECT.replace(
        variants={"default": sbml},
        panels={**dict(PROJECT.panels), panel.id: panel},
    )


@pytest.fixture(scope="module")
def registry(project: Project) -> ModelRegistry:
    return ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=project, title="Hydrochlorothiazide")}
    )


@pytest.fixture(scope="module")
def runner(registry: ModelRegistry) -> PanelRunner:
    return PanelRunner(registry)


def test_the_model_of_the_tests_carries_the_distribution_of_its_fit(
    sbml: Path,
) -> None:
    assert uncertainty_name(sbml) == FIT
    dimension = uncertainty_dimension(sbml, draws=4, seed=0)
    draws = np.asarray(dimension.values[PARAMETER], dtype=float)
    assert draws.shape == (4,)
    # the draws are around the value of the parameter, which is their median
    value = 0.351832524244915
    assert np.all((draws > value / 2) & (draws < value * 2))
    assert float(np.median(draws)) == pytest.approx(value, rel=0.1)


def test_the_band_of_a_point_is_the_percentiles_of_its_draws(
    runner: PanelRunner, panel: Panel
) -> None:
    result = runner.run("hctz", panel, {}, {}, draws=4)
    band = result.uncertainty
    assert band is not None
    assert (band.draws, band.fit, band.level) == (4, FIT, 0.8)
    assert band.failed == ()
    # a request of one point carries the curve of every draw as well
    assert band.curves is not None
    assert sorted(band.curves) == sorted(curve.id for curve in panel.curves)
    for curve in panel.curves:
        drawn = np.asarray(band.curves[curve.id], dtype=float)
        assert drawn.shape == (4, len(result.x))
        (one,) = band.band[curve.id]
        # the band is the 10th and the 90th percentile of those four draws
        np.testing.assert_allclose(one.low, nan_quantile(drawn.T, 0.1))
        np.testing.assert_allclose(one.high, nan_quantile(drawn.T, 0.9))
        low = np.asarray(one.low, dtype=float)
        high = np.asarray(one.high, dtype=float)
        assert np.all(high >= low)
        # the draws differ, so the band has a width where the curve is not 0
        nominal = np.asarray(result.curves[curve.id][0], dtype=float)
        peak = int(np.argmax(nominal))
        assert high[peak] > low[peak]
        assert low[peak] <= nominal[peak] <= high[peak]


def test_the_draws_a_request_asks_for_are_the_first_ones_of_the_sample(
    runner: PanelRunner, panel: Panel
) -> None:
    """A prefix of the sample is a sample of its own, so two draws are the first two."""
    four = runner.run("hctz", panel, {}, {}, draws=4)
    two = runner.run("hctz", panel, {}, {}, draws=2)
    assert four.uncertainty is not None
    assert two.uncertainty is not None
    assert two.uncertainty.curves is not None
    assert four.uncertainty.curves is not None
    for curve in panel.curves:
        first = np.asarray(four.uncertainty.curves[curve.id], dtype=float)[:2]
        np.testing.assert_array_equal(
            np.asarray(two.uncertainty.curves[curve.id], dtype=float), first
        )
    # and the curves of the panel are the ones of the run without a band
    plain = runner.run("hctz", panel, {}, {})
    assert plain.uncertainty is None
    assert plain.curves == four.curves


def test_a_band_travels_for_every_point_of_a_grid(
    runner: PanelRunner, panel: Panel
) -> None:
    # the dose of the panel is a choice knob, so its grid runs its choices
    doses = len(next(knob for knob in panel.knobs if knob.id == "dose").choices)
    result = runner.run("hctz", panel, {}, {"dose": doses}, draws=4)
    band = result.uncertainty
    assert band is not None
    # the draws themselves stay behind: over a grid the band travels alone
    assert band.curves is None
    for curve in panel.curves:
        points = band.band[curve.id]
        assert len(points) == doses
        for point, row in zip(points, result.curves[curve.id], strict=True):
            nominal = np.asarray(row, dtype=float)
            low = np.asarray(point.low, dtype=float)
            high = np.asarray(point.high, dtype=float)
            peak = int(np.argmax(nominal))
            assert low[peak] <= nominal[peak] <= high[peak]
        # a larger dose is a larger band, as it is a larger curve
        first = np.asarray(points[0].high, dtype=float)
        last = np.asarray(points[-1].high, dtype=float)
        assert float(np.nanmax(last)) > float(np.nanmax(first))


def test_the_draws_of_a_request_are_the_ones_the_panel_declares(
    runner: PanelRunner, panel: Panel
) -> None:
    assert check_draws(panel, 0) == 0
    assert check_draws(panel, 4) == 4
    with pytest.raises(DrawsError, match="carries 4 draws of its uncertainty, 5"):
        check_draws(panel, 5)
    with pytest.raises(DrawsError, match="carries 4 draws"):
        runner.run("hctz", panel, {}, {}, draws=5)
    # a panel which declares no uncertainty runs no draw, whatever it was asked
    plain = HCTZ_RENAL_PLASMA
    assert plain.simulation.uncertainty is None
    assert check_draws(plain, 4) == 0
    assert runner.run("hctz", plain, {}, {}, draws=4).uncertainty is None


def test_the_route_answers_the_band_and_hashes_the_draws(client: TestClient) -> None:
    pytest.importorskip("fastapi")
    plain = client.post(SIMULATE, json={})
    assert plain.status_code == 200
    assert plain.json()["uncertainty"] is None

    answer = client.post(SIMULATE, json={"draws": 2})
    assert answer.status_code == 200
    body = answer.json()
    uncertainty = body["uncertainty"]
    assert uncertainty["draws"] == 2
    assert uncertainty["fit"] == FIT
    assert uncertainty["level"] == 0.8
    assert uncertainty["failed"] == []
    assert sorted(uncertainty["band"]) == sorted(body["curves"])
    (one,) = uncertainty["band"]["[Cve_hctz]"]
    assert len(one["low"]) == len(body["x"])
    assert len(uncertainty["curves"]["[Cve_hctz]"]) == 2
    # the draws are part of what was simulated, so they are part of its hash
    assert body["spec_hash"] != plain.json()["spec_hash"]
    assert (
        client.post(SIMULATE, json={"draws": 4}).json()["spec_hash"]
        != (body["spec_hash"])
    )
    # a request which asks a panel of the curves alone for draws is the
    # request without them, the answer it gets
    other = client.post("/api/models/hctz/panels/renal_urine/simulate", json={})
    drawn = client.post(
        "/api/models/hctz/panels/renal_urine/simulate", json={"draws": 2}
    )
    assert drawn.json()["spec_hash"] == other.json()["spec_hash"]
    assert drawn.json()["uncertainty"] is None

    over = client.post(SIMULATE, json={"draws": 5})
    assert over.status_code == 422
    assert "carries 4 draws of its uncertainty, 5" in over.json()["detail"]
    # and a number which is no count of draws is no request at all
    assert client.post(SIMULATE, json={"draws": -1}).status_code == 422
    assert client.post(SIMULATE, json={"draws": 1.5}).status_code == 422


def test_the_sample_of_a_bundle_is_the_sample_of_the_service(
    registry: ModelRegistry, sbml: Path, panel: Panel
) -> None:
    """A static bundle carries the sample, so it answers the band of the service."""
    from sbmlsim.serve.export import build_bundle

    bundle = build_bundle(registry, "hctz")
    run: dict[str, Any] = bundle.content["answers"][""]["panels"][panel.id]["run"]
    sample = run["uncertainty"]
    assert sample["fit"] == FIT
    assert (sample["level"], sample["seed"]) == (0.8, 0)
    assert (sample["default_draws"], sample["max_draws"]) == (2, 4)
    assert sample["parameters"] == [PARAMETER]
    expected = uncertainty_dimension(sbml, draws=4, seed=0)
    np.testing.assert_allclose(
        np.asarray(sample["draws"], dtype=float).ravel(),
        np.asarray(expected.values[PARAMETER], dtype=float),
    )
    # the declaration of the panel says what a client may ask for
    declaration = bundle.content["answers"][""]["panels"][panel.id]["declaration"]
    assert declaration["simulation"]["uncertainty"] == {
        "draws": 2,
        "max_draws": 4,
        "level": 0.8,
        "seed": 0,
    }
    # a panel which declares none carries none
    other = bundle.content["answers"][""]["panels"]["renal_urine"]
    assert other["run"]["uncertainty"] is None
    assert other["declaration"]["simulation"]["uncertainty"] is None
