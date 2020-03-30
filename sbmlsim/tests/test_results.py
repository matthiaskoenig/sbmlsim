import pandas as pd

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import XResult
from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def test_result():
    r = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)._model
    dfs = []
    for _ in range(10):
        s = r.simulate(0, 10, steps=10)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    result = XResult(dfs)
    assert result
    assert result.nframes == 10
    assert result.nrow == 11
    assert result.data is not None


def test_hdf5(tmp_path):
    r = RoadrunnerSBMLModel(source=MODEL_REPRESSILATOR)._model
    dfs = []
    for _ in range(10):
        s = r.simulate(0, 10, steps=10)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    result = XResult(dfs)
    h5_path = tmp_path / "result.h5"
    result.to_hdf5(h5_path)

    result2 = XResult.from_hdf5(h5_path)
    assert result
    assert result.nframes == 10
    assert result.nrow == 11
    assert result.data is not None
