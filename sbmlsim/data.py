from pandas import pd
import logging

logger = logging.getLogger(__name__)


class DataSet(object):
    """DataSet"""

    def __init__(self, data: pd.DataFrame, udict=None, ureg=None):
        """

        :param frames: iterable of pd.DataFrame
        """

        # empty array for storage
        self.data = data
        # units dictionary for lookup and conversion
        self.udict = udict
        self.ureg = ureg