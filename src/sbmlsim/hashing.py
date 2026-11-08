"""The hashes which identify a run, a scan or a file.

A hash of this module is persisted and compared across runs, e.g. the
`spec_hash` of the run layer in `run.json`, the hash of a scan in a planned
spec and the checksums of a provenance, so its serialization never changes.
The module imports only the standard library.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def canonical_json_sha256(
    value: Any, default: Callable[[Any], Any] | None = None
) -> str:
    """Get the sha256 of the canonical JSON of a value.

    The canonical JSON has sorted keys, no whitespace (`separators=(",",
    ":")`), escapes every character which is not ASCII (`ensure_ascii`, the
    default of `json.dumps`) and writes `NaN` and the infinities as
    `NaN`/`Infinity`; it is encoded as UTF-8.

    Args:
        value: the value, of the types `json.dumps` writes.
        default: converts a value of another type, see `json.dumps`.

    Returns:
        The hexadecimal digest.
    """
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Get the sha256 of the bytes of a file.

    Args:
        path: the file.

    Returns:
        The hexadecimal digest.
    """
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
