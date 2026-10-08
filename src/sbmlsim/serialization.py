"""Helpers for JSON serialization of experiments."""

import json
from enum import Enum
from json import JSONEncoder
from pathlib import Path
from typing import Any

from matplotlib.pyplot import Figure as MPLFigure
from numpy import ndarray


def from_json(json_info: str | Path) -> dict[Any, Any]:
    """Load data from JSON."""
    d: dict[Any, Any]
    if isinstance(json_info, Path):
        with open(json_info, encoding="utf-8") as f_json:
            d = json.load(f_json)
    else:
        d = json.loads(json_info)
    return d


def to_json(object, path: Path | None = None) -> str | Path:
    """Serialize to JSON."""
    if path is None:
        return json.dumps(object, cls=ObjectJSONEncoder, indent=2)
    with open(path, "w", encoding="utf-8") as f_json:
        json.dump(object, fp=f_json, cls=ObjectJSONEncoder, indent=2)
    return path


class ObjectJSONEncoder(JSONEncoder):
    """Class for encoding in JSON."""

    def default(self, o):
        """JSON encoder."""
        if isinstance(o, Enum):
            # handle enums
            return o.name

        if isinstance(o, MPLFigure):
            # no serialization of Matplotlib figures
            return o.__class__.__name__

        if isinstance(o, ndarray):
            # handle numpy ndarrays
            return o.tolist()

        if isinstance(o, type):
            # a class, e.g. the experiment class of a fit, is its import path
            return f"{o.__module__}.{o.__qualname__}"

        if hasattr(o, "to_dict"):
            # custom serializer
            return o.to_dict()

        if hasattr(o, "__dict__"):
            return o.__dict__
        # handle pint
        return str(o)
