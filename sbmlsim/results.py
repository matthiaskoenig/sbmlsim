import logging
import numpy as np
import pandas as pd


class TimecourseResult(object):
    """Result of a single timecourse simulation. """

    def __init__(self, data, selections, changeset):

        # FIXME: what exactly is this result?
        # FIXME: what is going on with selections and changesets here?
        self.df = data

    @property
    def Nsel(self):
        return len(self.selections)

    @property
    def Nsim(self):
        return len(self.changeset)

    @property
    def mean(self):
        return pd.DataFrame(np.mean(self.df, axis=2), columns=self.selections)

    @property
    def std(self):
        return pd.DataFrame(np.std(self.df, axis=2), columns=self.selections)

    @property
    def min(self):
        return pd.DataFrame(np.min(self.df, axis=2), columns=self.selections)

    @property
    def max(self):
        return pd.DataFrame(np.max(self.df, axis=2), columns=self.selections)

    @staticmethod
    def append_results(results, offset_time=True):
        """ Append multiple timecourse results.

        Changeset and selections have to be identical.
        """
        # fix times in individual time frames
        tend = -1.0
        frames = []
        # FIXME
        for result in results:
            """
            df = result.df
            if "time" in df.columns:
                tend_new = df.time.values[-1]
                if tend < 0:
                    tend = tend_new
                else:
                    tend +=
                    
            # necessary to fix all the times:
            """
        df = pd.concat(frames)
        return





        return retult

