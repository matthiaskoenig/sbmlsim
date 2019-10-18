
from sbmlsim.result import Result
import pandas as pd

from sbmlsim.tests.constants import MODEL_REPRESSILATOR

import roadrunner


def test_result():
    r = roadrunner.RoadRunner(MODEL_REPRESSILATOR)
    dfs = []
    for _ in range(10):
        s = r.simulate(0, 10, steps=10)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    result = Result(dfs)
    assert result
    assert result.nframes == 10
    assert result.nrow == 11
    assert result.data is not None


def test_hdf5(tmp_path):
    r = roadrunner.RoadRunner(MODEL_REPRESSILATOR)
    dfs = []
    for _ in range(10):
        s = r.simulate(0, 10, steps=10)
        dfs.append(pd.DataFrame(s, columns=s.colnames))

    result = Result(dfs)
    h5_path = tmp_path / "result.h5"
    result.to_hdf5(h5_path)

    result2 = Result.from_hdf5(h5_path)
    assert result
    assert result.nframes == 10
    assert result.nrow == 11
    assert result.data is not None
