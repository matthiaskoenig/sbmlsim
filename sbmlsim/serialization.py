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
        # print(o)
        # print("-" * 80)

        if isinstance(o, Enum):
            # handle enums
            return o.name

        if isinstance(o, MPLFigure):
            # no serialization of Matplotlib figures
            return o.__class__.__name__

        # handle numpy ndarrays
        if isinstance(o, ndarray):
            return o.tolist()

        # custom serializer (Figure, Plot, Data, ...)
        if hasattr(o, "to_dict"):
            return o.to_dict()

        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            # handle pint
            return str(o)