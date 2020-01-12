import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataSeries(pd.Series):
    """DataSet - a pd.Series with additional unit information."""
    # additional properties
    _metadata = ['udict', 'ureg']

    @property
    def _constructor(self):
        return DataSeries

    @property
    def _constructor_expanddim(self):
        return DataSet


class DataSet(pd.DataFrame):
    """DataSet - a pd.DataFrame with additional unit information.
    """
    # additional properties
    _metadata = ['udict', 'ureg']

    @property
    def _constructor(self):
        return DataSet

    @property
    def _constructor_sliced(self):
        return DataSeries

    @classmethod
    def from_df(cls, data: pd.DataFrame, udict: dict, ureg):
        dset = DataSet(data)
        dset.udict = udict
        dset.ureg = ureg
        return dset


if __name__ == "__main__":
    df = pd.DataFrame({'col1': [1, 2, 3],
                       'col2': [2, 3, 4],
                       "col3": [4, 5, 6]})
    print(df)
    dset = DataSet.from_df(df, udict={"col1": "mM"}, ureg="test")
    print(dset)
    print(dset.udict)
    dset2 = dset[dset.col1 > 1]
    print(dset2)
    print(dset2.udict)
