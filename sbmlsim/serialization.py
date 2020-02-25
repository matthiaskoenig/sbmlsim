from json import JSONEncoder
from enum import Enum
from numpy import ndarray


class ObjectJSONEncoder(JSONEncoder):
    def default(self, o):
        """json encoder"""

        # handle enums
        if isinstance(o, Enum):
            return o.name

        # handle numpy ndarrays
        if isinstance(o, ndarray):
            return o.tolist()

        if hasattr(o, "__dict__"):
            return o.__dict__
        else:
            # handle pint
            return str(o)