from json import JSONEncoder
from enum import Enum
from numpy import ndarray
from matplotlib.pyplot import Figure as MPLFigure
from sbmlsim.plotting import Figure, Plot
from sbmlsim.data import Data


class ObjectJSONEncoder(JSONEncoder):
    def default(self, o):
        """json encoder"""
        # print(type(o))

        if isinstance(o, Enum):
            # handle enums
            return o.name

        if isinstance(o, MPLFigure):
            # no serialization of Matplotlib figures
            return o.__class__.__name__

        if isinstance(o, (Figure, Plot, Data)):
            # return o.__class__.__name__
            return o.to_dict()

        # handle numpy ndarrays
        if isinstance(o, ndarray):
            return o.tolist()

        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            # handle pint
            return str(o)