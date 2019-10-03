import logging
import numpy as np
import pandas as pd


class TimecourseResult(object):
    """Result of a timecourse simulation."""

    def __init__(self, data, selections, changeset):
        self.data = data
        self.changeset = changeset
        self.selections = selections

    @property
    def Nsel(self):
        return len(self.selections)

    @property
    def Nsim(self):
        return len(self.changeset)

    @property
    def mean(self):
        return pd.DataFrame(np.mean(self.data, axis=2), columns=self.selections)

    @property
    def std(self):
        return pd.DataFrame(np.std(self.data, axis=2), columns=self.selections)

    @property
    def min(self):
        return pd.DataFrame(np.min(self.data, axis=2), columns=self.selections)

    @property
    def max(self):
        return pd.DataFrame(np.max(self.data, axis=2), columns=self.selections)

