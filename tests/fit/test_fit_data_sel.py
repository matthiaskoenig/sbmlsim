"""FitData selects the rows of a dataset or the points of a scan and may have no x."""

import pandas as pd
import pytest

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit.objects import FitData


class SelExperiment(SimulationExperiment):
    """An experiment of a dataset of two groups."""

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "dose": [1.0, 2.0, 1.0, 2.0],
                "dose_unit": "mg",
                "cmax": [1.0, 2.0, 4.0, 5.0],
                "cmax_unit": "mg/l",
                "cmax_sd": [0.1, 0.2, 0.4, 0.5],
                "cmax_sd_unit": "mg/l",
                "n": [6, 6, 8, 8],
                "n_unit": "dimensionless",
            }
        )
        return {"tab": DataSet.from_df(df, ureg=self.ureg)}


@pytest.fixture
def experiment() -> SelExperiment:
    """Get the initialized experiment."""
    exp = SelExperiment()
    exp.initialize()
    return exp


def test_a_reference_selects_its_rows(experiment: SelExperiment) -> None:
    fd = FitData(
        experiment,
        dataset="tab",
        xid=None,
        yid="cmax",
        yid_sd="cmax_sd",
        sel={"group": "b"},
    )
    assert fd.x is None
    data = fd.get_data()
    assert data.x is None
    assert data.y is not None
    assert data.y.magnitude.tolist() == [4.0, 5.0]
    assert data.y_sd is not None
    assert data.y_sd.magnitude.size == 2


def test_an_observable_of_a_scan_selects_its_point(experiment: SelExperiment) -> None:
    fd = FitData(experiment, task="task_scan", xid=None, yid="pk.cmax", sel={"dose": 1})
    assert fd.x is None
    assert fd.y.sel == {"dose": 1}
    assert fd.sel == {"dose": 1}


def test_the_count_of_a_selection(experiment: SelExperiment) -> None:
    fd = FitData(
        experiment, dataset="tab", xid="dose", yid="cmax", count="n", sel={"group": "a"}
    )
    assert fd.count == 6
    assert (
        FitData(
            experiment,
            dataset="tab",
            xid="dose",
            yid="cmax",
            count="n",
            sel={"group": "b"},
        ).count
        == 8
    )
