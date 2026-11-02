"""Geometric means fitted with their geometric standard deviations (#403)."""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from sbmlsim.data import DataSet
from sbmlsim.fit import (
    FitData,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    FitSettings,
    ParameterSet,
)
from sbmlsim.fit.objects import GSD_ERROR, NoiseDistribution
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType, WeightingPointsType
from sbmlsim.fit.petab_v2 import to_petab
from sbmlsim.fit.petab_v2.likelihood import log_likelihood, noise_model_of
from sbmlsim.fit.petab_v2.reader import from_petab
from sbmlsim.fit.report import FitReport
from sbmlsim.model import AbstractModel
from tests.fit.scalar_experiment import CONC, TIME, TRUE_V, PKStudy
from tests.simulator.models import sbml_pk

#: the points of the timecourse, without the zero before the dose
POINTS = slice(40, None, 80)


class GmeanStudy(PKStudy):
    """Geometric means of `[C]` in g/l with their geometric SDs."""

    gsd: ClassVar[list[float]] = [1.2, 1.5, np.nan, 2.0, 1.3, 1.0]
    scale: ClassVar[float] = 1.0

    def datasets(self) -> dict[str, DataSet]:
        times = TIME[POINTS]
        df = pd.DataFrame(
            {
                "time": times,
                # mg/l of the model in g/l
                "C": self.scale * CONC[POINTS] / 1000,
                "C_gsd": self.gsd[: len(times)],
            }
        )
        return {
            "tc": DataSet.from_df(df, udict={"time": "hr", "C": "g/l"}, ureg=self.ureg)
        }

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_gmean": FitMapping(
                self,
                reference=FitData(
                    self, dataset="tc", xid="time", yid="C", yid_gsd="C_gsd"
                ),
                observable=FitData(self, task="task_single", xid="time", yid="[C]"),
            )
        }


def _problem(
    study: type[GmeanStudy], tmp_path: Path, settings: FitSettings | None = None
) -> OptimizationProblem:
    problem = OptimizationProblem(
        "gsd",
        [FitMappingCollection(experiment=study, mappings=["fm_gmean"])],
        [
            FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(
        settings or FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    )
    return problem


def test_a_geometric_sd_is_the_sd_of_the_logarithm(tmp_path: Path) -> None:
    """The SD of a point has the CV of the log-normal distribution of its gsd."""
    problem = _problem(GmeanStudy, tmp_path)
    n = len(TIME[POINTS])
    gsd = np.array(GmeanStudy.gsd[:n])
    sigma = np.log(gsd)
    # a missing and a geometric SD of 1 are no error, the largest one fills them
    measured = np.isfinite(sigma) & (sigma > 0)
    sigma[~measured] = np.nanmax(sigma)
    assert problem.y_errors_type[0] == GSD_ERROR
    log_errors = problem.y_errors_log[0]
    assert log_errors is not None
    np.testing.assert_allclose(log_errors, sigma)
    measured_points = problem.y_errors_measured[0]
    assert measured_points is not None
    np.testing.assert_array_equal(measured_points, measured)
    # the gmean in mg/l of the model, the gsd a factor in every unit
    gmean = CONC[POINTS]
    np.testing.assert_allclose(problem.y_references[0], gmean, rtol=1e-12)
    np.testing.assert_allclose(
        problem.y_errors[0], gmean * np.sqrt(np.exp(sigma**2) - 1), rtol=1e-12
    )


def test_the_weight_of_a_geometric_sd_is_the_inverse_geometric_cv(
    tmp_path: Path,
) -> None:
    settings = FitSettings(
        parameter_scale=ParameterScaleType.LINEAR,
        weighting_points=WeightingPointsType.ERROR_WEIGHTING,
    )
    problem = _problem(GmeanStudy, tmp_path, settings)
    sigma = problem.y_errors_log[0]
    assert sigma is not None
    np.testing.assert_allclose(
        problem.weights_points[0], 1 / np.sqrt(np.expm1(sigma**2)), rtol=1e-12
    )


def test_the_noise_of_a_geometric_sd_is_log_normal(tmp_path: Path) -> None:
    """The log-likelihood is the one of PEtab's log-normal noise with `log(gsd)`."""
    problem = _problem(GmeanStudy, tmp_path)
    noise = noise_model_of(problem, 0)
    assert noise.distribution is NoiseDistribution.LOG_NORMAL
    sigma = problem.y_errors_log[0]
    assert sigma is not None
    assert [value[0] for value in noise.placeholder_values] == sigma.tolist()
    # the data is the simulation at the true volume, the median of the noise
    m = problem.y_references[0]
    expected = np.sum(-0.5 * np.log(2 * np.pi * sigma**2 * m**2))
    llh = log_likelihood(problem, ParameterSet(sid="true", values={"V": TRUE_V}))
    assert llh == pytest.approx(expected, rel=1e-6)


def test_a_geometric_sd_round_trips_through_petab(tmp_path: Path) -> None:
    """The export writes log-normal noise, the reader reads geometric SDs."""
    model_path = tmp_path / "pk.xml"
    model_path.write_text(sbml_pk())

    class Study(GmeanStudy):
        def models(self) -> dict[str, AbstractModel | Path]:
            return {"m": AbstractModel(source=model_path)}

    problem = _problem(Study, tmp_path)
    settings = FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    yaml_file = to_petab(problem, tmp_path / "petab", settings=settings)
    observables = pd.read_csv(yaml_file.parent / "observables.tsv", sep="\t")
    assert observables["noiseDistribution"].tolist() == ["log-normal"]

    again, settings_again = from_petab(yaml_file, opid="gsd")
    again.initialize(settings_again)
    assert again.y_errors_type[0] == GSD_ERROR
    log_errors = again.y_errors_log[0]
    assert log_errors is not None
    expected = problem.y_errors_log[0]
    assert expected is not None
    np.testing.assert_allclose(log_errors, expected, rtol=1e-12)
    np.testing.assert_allclose(again.y_errors[0], problem.y_errors[0], rtol=1e-12)
    pset = ParameterSet(sid="near", values={"V": 1.1 * TRUE_V})
    assert log_likelihood(again, pset) == pytest.approx(
        log_likelihood(problem, pset), rel=1e-9
    )


def test_the_report_draws_the_geometric_sd(tmp_path: Path) -> None:
    problem = _problem(GmeanStudy, tmp_path)
    report = FitReport(
        problem,
        FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        [ParameterSet(sid="true", values={"V": TRUE_V})],
        mapping_figures=True,
    )
    assert report._data_label(0) == "data (gmean ×/÷ GSD)"
    (tmp_path / "plots").mkdir()
    report.plot_mapping(tmp_path / "plots")
    assert list((tmp_path / "plots").glob("*.svg"))


def test_a_geometric_sd_below_one_raises(tmp_path: Path) -> None:
    class Study(GmeanStudy):
        gsd: ClassVar[list[float]] = [1.2, 0.8, 1.5, 1.5, 1.5, 1.5]

    with pytest.raises(ValueError, match=r"rows \[1\] is below 1"):
        _problem(Study, tmp_path)


def test_a_geometric_mean_is_positive(tmp_path: Path) -> None:
    class Study(GmeanStudy):
        scale: ClassVar[float] = -1.0

    with pytest.raises(ValueError, match="no positive geometric means"):
        _problem(Study, tmp_path)


def test_a_geometric_sd_is_the_only_error() -> None:
    with pytest.raises(ValueError, match="set one error"):
        FitData(
            object(), dataset="tc", xid="time", yid="C", yid_sd="C_sd", yid_gsd="C_gsd"
        )


def test_geometric_means_are_fitted_with_their_errors(tmp_path: Path) -> None:
    """A fit with the weights of the geometric SDs recovers the volume."""
    settings = FitSettings(
        parameter_scale=ParameterScaleType.LINEAR,
        weighting_points=WeightingPointsType.ERROR_WEIGHTING,
    )
    problem = _problem(GmeanStudy, tmp_path, settings)
    assert problem.cost_least_square(np.array([TRUE_V])) == pytest.approx(
        0.0, abs=1e-10
    )
    fits, _ = problem.optimize(size=1, seed=1, max_nfev=30)
    assert abs(float(fits[0].x[0]) - TRUE_V) / TRUE_V < 1e-3
