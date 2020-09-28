import pandas as pd

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import XResult
from sbmlsim.test import MODEL_REPRESSILATOR


def test_result():
    r = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)._model
    dfs = []
    num_sim = 10
    num_steps = 20
    for _ in range(num_sim):
        s = r.simulate(0, 10, steps=num_steps)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    xres = XResult.from_dfs(dfs)

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


def test_netcdf(tmp_path):
    r = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)._model
    dfs = []
    for _ in range(10):
        s = r.simulate(0, 10, steps=10)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    xres = XResult.from_dfs(dfs)
    nc_path = tmp_path / "result.nc"
    xres.to_netcdf(nc_path)

    xres2 = XResult.from_netcdf(nc_path)
    assert xres is not None
    assert len(xres.dims) == len(xres2.dims)
