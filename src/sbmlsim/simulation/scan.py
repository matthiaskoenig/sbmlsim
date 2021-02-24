"""Scan simulation.

Allows scans over other simulations.
"""

import logging
from copy import deepcopy
from typing import Dict, List

import numpy as np

from sbmlsim.simulation import AbstractSim, Dimension
from sbmlsim.units import UnitRegistry, Units


logger = logging.getLogger()


class ScanSim(AbstractSim):
    """A scan simulation over another AbstractSim.

    FIXME: probably not necessary to make this a simulation.
    """

    def __init__(
        self,
        simulation: AbstractSim,
        dimensions: List[Dimension] = None,
        mapping: Dict[str, int] = None,
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
        self.simulation = simulation
        if dimensions is None:
            # handling the simple simulation case
            dimensions = [Dimension("dim0", index=np.arange(1))]
        self.dimensions = dimensions
        dimension_keys = [dim.dimension for dim in self.dimensions]
        if len(dimension_keys) > len(set(dimension_keys)):
            raise ValueError(f"duplicate dimension keys in scan: {dimension_keys}")

        if mapping is None:
            # if no mapping is provided than the changes map on the
            # initial part of the simulation
            mapping = {dim.dimension: 0 for dim in self.dimensions}
        if len(mapping) != len(dimensions):
            raise ValueError(
                f"mapping '{mapping}' incompatible with dimensions " f"'{dimensions}'."
            )
        self.mapping = mapping

    def __repr__(self) -> str:
        """Get representation."""
        return (
            f"Scan({self.simulation.__class__.__name__}: "
            f"[{', '.join([str(d) for d in self.dimensions])}])"
        )

    def dimensions(self) -> List[Dimension]:
        """Get dimensions."""
        return self.dimensions

    def get_dimension(self, key: str) -> Dimension:
        """Get dimension by key."""
        for dim in self.dimensions:
            if dim.dimension == dim:
                return dim
        raise KeyError(f"Dimension with key '{key}' does not exist.")

    def indices(self):
        """Get indices of all combinations."""
        return Dimension.indices_from_dimensions(self.dimensions)

    def add_model_changes(self, model_changes: Dict) -> None:
        """Add model changes to first timecourse."""
        # import here to avoid circular import
        from sbmlsim.simulation import TimecourseSim

        if self.simulation and isinstance(self.simulation, TimecourseSim):
            self.simulation.add_model_changes(model_changes)

    def normalize(self, udict: Dict, ureg: UnitRegistry):
        """Normalize scan.

        Requires normalization of timecourse simulation as well
        as all dimensions in the scan.
        """
        # normalize simulation
        self.simulation.normalize(udict=udict, ureg=ureg)

        # normalize changes in all dimensions
        for scan_dim in self.dimensions:
            scan_dim.changes = Units.normalize_changes(
                scan_dim.changes, udict=udict, ureg=ureg
            )

    def to_simulations(self):
        """Flatten the scan to individual simulations.

        Here the changes are appended.
        Scan should be normalized before calling this function.
        Necessary to track the results.
        """
        # create all combinations of the scan
        indices = self.indices()
        # create respective simulations
        simulations = []
        for index_list in indices:
            sim_new = deepcopy(self.simulation)

            # TODO: support additional simulation types (currently
            #       only Timecourses assumed.
            for k_dim, k_index in enumerate(index_list):
                # add all changes for the given dimension and index
                dim = self.dimensions[k_dim]
                changes = dim.changes
                map_index = self.mapping[dim.dimension]
                # changes have to be applied to correct part of simulation
                tc = sim_new.timecourses[map_index]
                for key in changes.keys():
                    value = changes[key][k_index]
                    tc.add_change(key, value)

            simulations.append(sim_new)

        # print(simulations)
        # for sim in simulations:
        #    print("-" * 80)
        #    print(sim)

        return indices, simulations


if __name__ == "__main__":
    import warnings

    from sbmlsim.simulation import Timecourse, TimecourseSim
    from sbmlsim.units import Quantity, UnitRegistry

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Quantity([])

    ureg = UnitRegistry()
    Q_ = ureg.Quantity
    udict = {k: "dimensionless" for k in ["X", "[X]", "n", "Y"]}

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
                index=range(8),
                changes={
                    "n": Q_(np.linspace(start=2, stop=10, num=8), "dimensionless"),
                },
            ),
            Dimension(
                "dim2",
                index=range(4),
                changes={
                    "Y": Q_(np.linspace(start=10, stop=20, num=4), "dimensionless"),
                },
            ),
        ],
    )
    indices, sims = scan2d.to_simulations()
    scan2d.normalize(udict=udict, ureg=ureg)
