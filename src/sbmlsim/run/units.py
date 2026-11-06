"""The units of a split run: blocks of consecutive items.

A splittable kind cuts its items (the repeats of a fit, the points of a scan,
the draws of a propagation) into at most `max_units` units of consecutive
items, `unit_blocks`, so the number of units does not grow with the size of an
analysis, and the merge raises `MissingUnits` naming the units it misses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import override

__all__ = ["MAX_UNITS", "MissingUnits", "unit_blocks"]

#: the number of units an analysis is split into at most
MAX_UNITS = 64

#: the number of missing units a message names at most
_NAMED = 20


def unit_blocks(n: int, max_units: int = MAX_UNITS) -> list[range]:
    """Split items into blocks of consecutive items, one block per unit.

    Every block has `max(1, ceil(n / max_units))` items, the last one the
    rest, so there are at most `max_units` blocks.

    Args:
        n: the number of items.
        max_units: the number of blocks at most.

    Returns:
        The blocks in the order of the items, none for no items.

    Raises:
        ValueError: for a negative `n` or `max_units` below 1.
    """
    if n < 0:
        raise ValueError(f"The number of items must not be negative, it is {n}.")
    if max_units < 1:
        raise ValueError(f"There must be at least one unit, not {max_units}.")
    size = max(1, math.ceil(n / max_units))
    return [range(start, min(start + size, n)) for start in range(0, n, size)]


class MissingUnits(RuntimeError):
    """A merge misses units which were not computed.

    Attributes:
        missing: the indices of the missing units.
        n_units: the number of units.
    """

    def __init__(self, missing: Sequence[int], n_units: int) -> None:
        """Name the missing units.

        Args:
            missing: the indices of the missing units.
            n_units: the number of units.
        """
        self.missing = tuple(missing)
        self.n_units = n_units
        named = ", ".join(str(i) for i in self.missing[:_NAMED])
        if len(self.missing) > _NAMED:
            named += ", ..."
        super().__init__(
            f"{len(self.missing)} of {n_units} units are missing: {named}; "
            "compute them before the merge."
        )

    @override
    def __reduce__(self) -> tuple[type[MissingUnits], tuple[tuple[int, ...], int]]:
        """Pickle the error with its arguments, e.g. from a worker of a pool."""
        return type(self), (self.missing, self.n_units)
