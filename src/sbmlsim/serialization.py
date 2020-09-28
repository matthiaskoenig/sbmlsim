"""
Helpers for JSON serialization of experiments.
"""
import json
from enum import Enum
from json import JSONEncoder

from matplotlib.pyplot import Figure as MPLFigure
from numpy import ndarray


class ObjectJSONEncoder(JSONEncoder):
    def to_json(self, path=None):
        """Convert definition to JSON for exchange.

        :param path: path for file, if None JSON str is returned
        :return:
        """
        if path is None:
            return json.dumps(self, cls=ObjectJSONEncoder, indent=2)
        else:
            with open(path, "w") as f_json:
                json.dump(self, fp=f_json, cls=ObjectJSONEncoder, indent=2)

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
