"""Scan of a simulation over dimensions of changes.

A `ScanSim` runs a `Simulation` for every combination of the values of its
dimensions. The values of a dimension replace the values of their targets
wherever the simulation sets them, i.e. a scanned dose replaces the dose of
every `Change` which sets it, and a target the simulation does not set is a
pre-initialization change, see `Simulation.with_values`. A dimension with `at`
applies its values as a `Change` at that time instead.
"""

import logging
from typing import Any

from sbmlsim.simulation.definition import Change, Simulation
from sbmlsim.simulation.range import Dimension

logger = logging.getLogger(__name__)


class ScanSim:
    """A scan of a simulation over the dimensions of its changes."""

    def __init__(
        self,
        simulation: Simulation,
        dimensions: list[Dimension] | None = None,
    ):
        """Scan a simulation.

        Args:
            simulation: the simulation which is scanned.
            dimensions: the dimensions of the scan, every combination of their
                values is a simulation.

        Raises:
            ValueError: if two dimensions have the same id.
        """
        self.simulation: Simulation = simulation
        self.dimensions: list[Dimension] = list(dimensions or [])
        dimension_keys = [dim.dimension for dim in self.dimensions]
        if len(dimension_keys) > len(set(dimension_keys)):
            raise ValueError(f"duplicate dimension keys in scan: {dimension_keys}")

    def __repr__(self) -> str:
        """Get representation."""
        return (
            f"Scan({self.simulation!r}: "
            f"[{', '.join([str(d) for d in self.dimensions])}])"
        )

    def indices(self) -> list[tuple[Any, ...]]:
        """Get indices of all combinations."""
        return Dimension.indices_from_dimensions(self.dimensions)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON types."""
        return {
            "type": self.__class__.__name__,
            "simulation": self.simulation.to_dict(),
            "dimensions": [str(dim) for dim in self.dimensions],
        }

    def to_simulations(self) -> tuple[list[tuple[Any, ...]], list[Simulation]]:
        """Get the simulations of the scan.

        Returns:
            The indices of every combination of the dimensions and its
            simulation, see the module.
        """
        indices = self.indices()
        simulations: list[Simulation] = []
        for index_list in indices:
            values: dict[str, Any] = {}
            timed: list[Change] = []
            for k_dim, k_index in enumerate(index_list):
                dim = self.dimensions[k_dim]
                dim_values = {key: dim.changes[key][k_index] for key in dim.changes}
                if dim.at is None:
                    values.update(dim_values)
                else:
                    timed.append(Change(dim.at, dim_values))
            simulation = self.simulation.with_values(values)
            simulation.changes.extend(timed)
            simulations.append(simulation)
        return indices, simulations
