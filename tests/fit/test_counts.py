"""The counts of the data points of a fit and their standard deviations."""

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
)
from sbmlsim.fit.optimization import OptimizationProblem
from tests.fit.scalar_experiment import CMAX, CMAX_UNIT, PKStudy


class _Study(PKStudy):
    """Two groups of 100 mg against the one simulated cmax."""

    columns: ClassVar[dict[str, list[float]]] = {}
    count: ClassVar[int | str | None] = "count"
    error: ClassVar[str] = "cmax_se"

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame({"cmax": [CMAX[1], CMAX[1]], **self.columns})
        units = {"cmax": CMAX_UNIT, "cmax_se": CMAX_UNIT, "cmax_sd": CMAX_UNIT}
        units = {key: unit for key, unit in units.items() if key in df.columns}
        return {"tab": DataSet.from_df(df, udict=units, ureg=self.ureg)}

    def fit_mappings(self) -> dict[str, FitMapping]:
        se = self.error.endswith("se")
        return {
            "fm_cmax": FitMapping(
                self,
                reference=FitData(
                    self,
                    dataset="tab",
                    xid=None,
                    yid="cmax",
                    count=self.count,
                    yid_se=self.error if se else None,
                    yid_sd=None if se else self.error,
                ),
                observable=FitData(self, task="task_single", xid=None, yid="pk.cmax"),
            )
        }


def _initialized(study: type[_Study], tmp_path: Path) -> OptimizationProblem:
    problem = OptimizationProblem(
        "counts",
        [FitMappingCollection(experiment=study, mappings=["fm_cmax"])],
        [
            FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings())
    return problem


def test_an_se_is_converted_to_an_sd(tmp_path: Path) -> None:
    class Study(_Study):
        columns: ClassVar[dict[str, list[float]]] = {
            "cmax_se": [1.0, 1.0],
            "count": [4.0, 9.0],
        }

    problem = _initialized(Study, tmp_path)
    np.testing.assert_allclose(problem.y_errors[0], [2.0, 3.0])
    assert problem.y_errors_type[0] == "SD"
    assert problem.y_counts[0] is not None
    np.testing.assert_allclose(problem.y_counts[0], [4.0, 9.0])


def test_a_constant_count(tmp_path: Path) -> None:
    class Study(_Study):
        columns: ClassVar[dict[str, list[float]]] = {"cmax_se": [0.5, 0.5]}
        count = 16

    problem = _initialized(Study, tmp_path)
    np.testing.assert_allclose(problem.y_errors[0], [2.0, 2.0])
    assert problem.y_counts[0] is not None
    np.testing.assert_allclose(problem.y_counts[0], [16.0, 16.0])


def test_an_sd_is_kept(tmp_path: Path) -> None:
    class Study(_Study):
        columns: ClassVar[dict[str, list[float]]] = {
            "cmax_sd": [1.5, 2.5],
            "count": [4.0, 9.0],
        }
        error = "cmax_sd"

    problem = _initialized(Study, tmp_path)
    np.testing.assert_allclose(problem.y_errors[0], [1.5, 2.5])
    assert problem.y_errors_type[0] == "SD"
    assert problem.y_counts[0] is not None
    np.testing.assert_allclose(problem.y_counts[0], [4.0, 9.0])


def test_data_without_a_count_has_no_counts(tmp_path: Path) -> None:
    class Study(_Study):
        columns: ClassVar[dict[str, list[float]]] = {"cmax_sd": [1.5, 2.5]}
        error = "cmax_sd"
        count = None

    problem = _initialized(Study, tmp_path)
    assert problem.y_counts[0] is None


def test_an_se_without_a_count_raises(tmp_path: Path) -> None:
    class Study(_Study):
        columns: ClassVar[dict[str, list[float]]] = {"cmax_se": [1.0, 1.0]}
        count = None

    with pytest.raises(
        ValueError, match="cannot be converted to a standard deviation without a count"
    ):
        _initialized(Study, tmp_path)


def test_an_se_with_a_missing_count_raises(tmp_path: Path) -> None:
    class Study(_Study):
        columns: ClassVar[dict[str, list[float]]] = {
            "cmax_se": [1.0, 1.0],
            "count": [4.0, np.nan],
        }

    with pytest.raises(ValueError, match=r"rows \[1\]"):
        _initialized(Study, tmp_path)
