"""Scan simulation.

Allows scans over other simulations.
"""

import logging
from copy import deepcopy
from typing import Any

import numpy as np

from sbmlsim.simulation.definition import Change, Simulation
from sbmlsim.simulation.range import Dimension
from sbmlsim.simulation.simulation import AbstractSim
from sbmlsim.simulation.timecourse import TimecourseSim
from sbmlsim.units import UnitsInformation

logger = logging.getLogger(__name__)


class ScanSim(AbstractSim):
    """A scan simulation over another AbstractSim.

    FIXME: probably not necessary to make this a simulation.
    """

    def __init__(
        self,
        simulation: "AbstractSim | Simulation",
        dimensions: list[Dimension] | None = None,
        mapping: dict[str, int] | None = None,
    ):
        """Scan a simulation.

        Parameters or initial conditions can be scanned.
        Multiple parameters will result in a multi-dimensional scan.
        If the changes should be applied to a later timecourse in the
        timecourse simulation the mapping dictionary can be used to map
        the changes of a given dimension on the respective timecourse
        index (starting from index 0). E.g.,
            mapping = {dim_1: 1}
        will apply the changes of 'dim_1' on the second timecouse in the
        timecourse simulation.

        :param simulation: simulation to scan over the given parameters
        :param scan: dictionary of parameters or conditions to scan
        :param mapping: map of changes to parts of simulations
        """
        self.simulation: AbstractSim | Simulation = simulation
        if dimensions is None:
            # handling the simple simulation case
            dimensions = []
        self.dimensions: list[Dimension] = dimensions
        dimension_keys = [dim.dimension for dim in self.dimensions]
        if len(dimension_keys) > len(set(dimension_keys)):
            raise ValueError(f"duplicate dimension keys in scan: {dimension_keys}")

        if mapping is None:
            # if no mapping is provided than the changes map on the
            # initial part of the simulation
            mapping = {dim.dimension: 0 for dim in self.dimensions}
        if len(mapping) != len(dimensions):
            raise ValueError(
                f"mapping '{mapping}' incompatible with dimensions '{dimensions}'."
            )
        self.mapping: dict[str, int] = mapping

    def __repr__(self) -> str:
        """Get representation."""
        return (
            f"Scan({self.simulation.__class__.__name__}: "
            f"[{', '.join([str(d) for d in self.dimensions])}])"
        )

    def get_dimension(self, key: str) -> Dimension:
        """Get dimension by key."""
        for dim in self.dimensions:
            if dim.dimension == key:
                return dim
        raise KeyError(f"Dimension with key '{key}' does not exist.")

    def indices(self) -> list[tuple[Any, ...]]:
        """Get indices of all combinations."""
        return Dimension.indices_from_dimensions(self.dimensions)

    def add_model_changes(self, model_changes: dict[str, Any]) -> None:
        """Add model changes to first timecourse."""
        if self.simulation and isinstance(self.simulation, TimecourseSim):
            self.simulation.add_model_changes(model_changes)

    def normalize(self, uinfo: UnitsInformation) -> None:
        """Normalize units in scan.

        Requires normalization of timecourse simulation as well
        as all dimensions in the scan.
        """
        # normalize simulation, a `Simulation` is converted when it is compiled
        if isinstance(self.simulation, AbstractSim):
            self.simulation.normalize(uinfo=uinfo)

        # normalize changes in all dimensions
        for scan_dim in self.dimensions:
            scan_dim.changes = UnitsInformation.normalize_changes(
                scan_dim.changes, uinfo=uinfo
            )

    def to_simulations(self) -> tuple[list[tuple[Any, ...]], list[Any]]:
        """Flatten the scan to individual simulations, of `Simulation` or `TimecourseSim`.

        Here the changes are appended.
        Scan should be normalized before calling this function.
        Necessary to track the results.
        """
        if isinstance(self.simulation, Simulation):
            return self._simulations_of_definition(self.simulation)
        if not isinstance(self.simulation, TimecourseSim):
            raise NotImplementedError(
                f"Only TimecourseSim supported in scan, but '{type(self.simulation)}'"
            )

        # create all combinations of the scan
        indices = self.indices()
        # create respective simulations
        simulations: list[TimecourseSim] = []
        for index_list in indices:
            sim_new = deepcopy(self.simulation)

            for k_dim, k_index in enumerate(index_list):
                # add all changes for the given dimension and index
                dim = self.dimensions[k_dim]
                changes = dim.changes
                map_index = self.mapping[dim.dimension]
                # changes have to be applied to correct part of simulation
                tc = sim_new.timecourses[map_index]
                for key in changes:
                    value = changes[key][k_index]
                    tc.add_change(key, value)

            simulations.append(sim_new)

        return indices, simulations

    def _simulations_of_definition(
        self, simulation: Simulation
    ) -> tuple[list[tuple[Any, ...]], list[Simulation]]:
        """Get the simulations of a scan of a `Simulation`.

        The values of a dimension replace the values of their targets wherever
        the simulation sets them, see `Simulation.with_values`; the values of
        a dimension with `at` are a `Change` at that time.
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
            sim = simulation.with_values(values)
            if timed:
                sim.changes.extend(timed)
            simulations.append(sim)
        return indices, simulations


if __name__ == "__main__":
    from sbmlsim.simulation import Timecourse
    from sbmlsim.units import ureg

    Q_ = ureg.Quantity
    uinfo = UnitsInformation(
        udict=dict.fromkeys(["X", "[X]", "n", "Y"], "dimensionless"), ureg=ureg
    )

    scan2d = ScanSim(
        simulation=TimecourseSim(
            [
                Timecourse(
                    start=0, end=100, steps=100, changes={"X": Q_(10, "dimensionless")}
                ),
                Timecourse(
                    start=0, end=60, steps=100, changes={"[X]": Q_(10, "dimensionless")}
                ),
                Timecourse(
                    start=0, end=60, steps=100, changes={"X": Q_(10, "dimensionless")}
                ),
            ]
        ),
        dimensions=[
            Dimension(
                "dim1",
                index=np.arange(8),
                changes={
                    "n": Q_(np.linspace(start=2, stop=10, num=8), "dimensionless"),
                },
            ),
            Dimension(
                "dim2",
                index=np.arange(4),
                changes={
                    "Y": Q_(np.linspace(start=10, stop=20, num=4), "dimensionless"),
                },
            ),
        ],
    )
    indices, sims = scan2d.to_simulations()
    scan2d.normalize(uinfo=uinfo)
