"""Test XResults."""
from pathlib import Path

import pandas as pd

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.rr_model import roadrunner
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.xresult import XResult


def test_xresult(repressilator_path: Path) -> None:
    """Test xresults."""
    r: roadrunner.RoadrunnerRunner = RoadrunnerSBMLModel(
        source=REPRESSILATOR_SBML
    ).model
    dfs = []
    num_sim = 10
    num_steps = 20
    for _ in range(num_sim):
        s = r.simulate(0, 10, steps=num_steps)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    xres: XResult = XResult.from_dfs(dfs)

    assert xres
    # check dimensions
    assert len(xres.dims) == 2
    assert "_time" in xres.dims
    assert "_dfs" in xres.dims

    # check coordinates
    assert "_time" in xres.coords
    assert "_dfs" in xres.coords
    assert len(xres.coords["_time"]) == (num_steps + 1)
    assert len(xres.coords["_dfs"]) == num_sim

    assert xres.X is not None
    assert xres.Y is not None


def test_xresults_netcdf(tmp_path: Path) -> None:
    """Test xresults in netcdf format."""
    r: roadrunner.RoadRunner = RoadrunnerSBMLModel(source=REPRESSILATOR_SBML).model
    print(r.timeCourseSelections)
    dfs = []
    for _ in range(10):
        s = r.simulate(0, 10, steps=10)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    xres = XResult.from_dfs(dfs)
    nc_path = tmp_path / "result.nc"
    xres.to_netcdf(nc_path)

    xres2 = XResult.from_netcdf(nc_path)
    assert xres2 is not None
    assert len(xres.dims) == len(xres2.dims)
    assert xres.equals(xres2)
