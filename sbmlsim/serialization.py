"""
Helpers for JSON serialization of experiments.
"""
from json import JSONEncoder
from enum import Enum
from numpy import ndarray
from matplotlib.pyplot import Figure as MPLFigure


class ObjectJSONEncoder(JSONEncoder):
    def default(self, o):
        """json encoder"""

        if isinstance(o, Enum):
            # handle enums
            return o.name

        if isinstance(o, MPLFigure):
            # no serialization of Matplotlib figures
            return o.__class__.__name__

        if isinstance(o, ndarray):
            # handle numpy ndarrays
            return o.tolist()

        if hasattr(o, "to_dict"):
            # custom serializer
            return o.to_dict()

        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            # handle pint
            return str(o)
