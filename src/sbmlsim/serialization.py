"""Helpers for JSON serialization of experiments."""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from json import JSONEncoder
from pathlib import Path
from typing import Any, overload, override

from matplotlib.pyplot import Figure as MPLFigure
from numpy import generic, ndarray

from sbmlsim.units import Quantity


def from_json(json_info: str | Path) -> dict[Any, Any]:
    """Load data from JSON."""
    d: dict[Any, Any]
    if isinstance(json_info, Path):
        with open(json_info, encoding="utf-8") as f_json:
            d = json.load(f_json)
    else:
        d = json.loads(json_info)
    return d


@overload
def to_json(object: Any, path: None = None) -> str: ...


@overload
def to_json(object: Any, path: Path) -> Path: ...


def to_json(object: Any, path: Path | None = None) -> str | Path:
    """Serialize to JSON.

    The object is encoded before the file is written, so a value which cannot
    be encoded leaves no truncated file.

    Args:
        object: the object to serialize.
        path: the file to write, `None` for the JSON as a string.

    Returns:
        The JSON, or the path of the file it was written to.

    Raises:
        TypeError: for a value which `ObjectJSONEncoder` cannot encode.
    """
    text = json.dumps(object, cls=ObjectJSONEncoder, indent=2)
    if path is None:
        return text
    path.write_text(text, encoding="utf-8")
    return path


class ObjectJSONEncoder(JSONEncoder):
    """Class for encoding in JSON."""

    @override
    def default(self, o: Any) -> Any:
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

        if isinstance(o, generic):
            value = o.item()
            # Extended precision scalars can return another numpy scalar;
            # reject unsupported precision rather than recursing forever.
            return super().default(o) if isinstance(value, generic) else value

        if isinstance(o, Path):
            return str(o)

        if isinstance(o, Quantity):
            # Preserve the existing quantity representation without exposing
            # arbitrary internal attributes of the quantity or its registry.
            return {"_magnitude": o.magnitude, "_units": str(o.units)}

        if isinstance(o, type):
            # a class, e.g. the experiment class of a fit, is its import path
            return f"{o.__module__}.{o.__qualname__}"

        if callable(getattr(o, "to_dict", None)):
            # custom serializer
            return o.to_dict()

        if is_dataclass(o):
            return {field.name: getattr(o, field.name) for field in fields(o)}

        if isinstance(o, Mapping):
            return dict(o)

        return super().default(o)
