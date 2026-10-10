"""The report of a fit with values per simulation and values over a dimension."""

from pathlib import Path

import numpy as np
import pytest

from sbmlsim.fit import FitMappingCollection, FitParameter, FitSettings, ParameterSet
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.report import FitReport
from tests.fit.scalar_experiment import TRUE_V, DoseStudy, ScalarStudy


@pytest.fixture
def report(tmp_path: Path) -> FitReport:
    """The report of parameters near the true ones (a perfect fit has no MSE)."""
    problem = OptimizationProblem(
        "scalar_report",
        [
            FitMappingCollection(experiment=ScalarStudy, mappings=["fm_cmax"]),
            FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"]),
        ],
        [
            FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    settings = FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    return FitReport(
        problem,
        settings,
        ParameterSet(sid="near", values={"V": 1.05 * TRUE_V}),
        mapping_figures=True,
    )


def test_the_report_files_of_both_kinds(report: FitReport, tmp_path: Path) -> None:
    """The html and the two figures of both mappings are written."""
    results_dir = report.create(output_dir=tmp_path / "out", name="scalar")
    assert (results_dir / "index.html").exists()
    plots = results_dir / "plots"
    for mapping in report.problem.mapping_keys:
        sid = report.problem.experiment_keys[report.problem.mapping_keys.index(mapping)]
        assert (plots / f"{sid}_{mapping}.svg").exists()
        assert (plots / f"fit_{sid}_{mapping}.svg").exists()
    assert (plots / "goodness_of_fit.svg").exists()
    assert (plots / "bland_altman.svg").exists()
    html = (results_dir / "index.html").read_text()
    assert "fm_cmax" in html
    assert "fm_dose" in html


def test_the_points_of_scalar_mappings(report: FitReport) -> None:
    """A scalar point has no x, a point over a dimension has the dose."""
    pset = report.reference_set
    points = report.points(pset)
    assert len(points) == 1 + 3
    assert np.isfinite(points["DV"]).all()
    assert np.isfinite(points["IPRED"]).all()
    scalar = points[points["mapping"] == "fm_cmax"]
    assert scalar["x"].isna().all()
    dose = points[points["mapping"] == "fm_dose"]
    assert np.isfinite(dose["x"]).all()

    np.testing.assert_allclose(points["DV"], points["IPRED"], rtol=0.1)

    frame = report._datapoints_df(pset)
    assert len(frame) == 4
    assert frame[frame["mapping"] == "fm_cmax"]["x_ref"].isna().all()
