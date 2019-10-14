"""
Helpers for working with timecourse results.
"""

import logging
import numpy as np
import pandas as pd


# FIXME: hashing

class TimecourseResult(object):
    """Result of a single timecourse simulation. """

    def __init__(self, dataframes):
        # empty array for storage
        df = dataframes[0]
        self.columns = df.columns
        Nt = len(df)
        Ncol = len(self.columns)
        Nsim = len(dataframes)
        data = np.empty((Nt, Ncol, Nsim)) * np.nan

        for k, df in enumerate(dataframes):
            data[:, :, k] = df.values

        self.data = data

    @property
    def mean(self):
        return pd.DataFrame(np.mean(self.data, axis=2), columns=self.columns)

    @property
    def std(self):
        return pd.DataFrame(np.std(self.data, axis=2), columns=self.columns)

    @property
    def min(self):
        return pd.DataFrame(np.min(self.data, axis=2), columns=self.columns)

    @property
    def max(self):
        return pd.DataFrame(np.max(self.data, axis=2), columns=self.columns)
