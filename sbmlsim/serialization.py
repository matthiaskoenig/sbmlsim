from json import JSONEncoder
from enum import Enum
from numpy import ndarray
from matplotlib.pyplot import Figure
from sbmlsim.data import Data


class ObjectJSONEncoder(JSONEncoder):
    def default(self, o):
        """json encoder"""
        # print(type(o))
        # print(o)

        # handle enums
        if isinstance(o, Enum):
            return o.name

        # no serialization of Matplotlib figures
        if isinstance(o, Figure):
            return o.__class__.__name__
        if isinstance(o, Data):
            return o.__class__.__name__

        # handle numpy ndarrays
        if isinstance(o, ndarray):
            return o.tolist()

        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            # handle pint
            return str(o)