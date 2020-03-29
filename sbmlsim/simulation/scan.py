import logging
import numpy as np
import itertools
from pint import UnitRegistry
from copy import deepcopy
from typing import Dict, List

from sbmlsim.simulation import AbstractSim, Dimension
from sbmlsim.units import Units

logger = logging.getLogger()


class ScanSim(AbstractSim):
    """A scan simulation over another AbstractSim.

    FIXME: probably not necessary to make this a simulation.
    """

    def __init__(self, simulation: AbstractSim, dimensions: List[Dimension]):
        """Scanning a simulation.

        Parameters or initial conditions can be scanned.
        Multiple parameters will result in a multi-dimensional scan.

        :param simulation: simulation to scan over the given parameters
        :param scan: dictionary of parameters or conditions to scan
        """
        self.simulation = simulation
        self.dimensions = dimensions

    def __repr__(self):
        return f"Scan({self.simulation.__class__.__name__}: " \
               f"[{', '.join([str(d) for d in self.dimensions])}])"

    def dimensions(self):
        return self.dimensions

    def indices(self):
        """Indices of all combinations."""
        index_vecs = [dim.index for dim in self.dimensions]
        return list(itertools.product(*index_vecs))

    def to_simulations(self):
        """Flattens the scan to individual simulations.

        Necessary to track the results.
        """
        # create all combinations of the scan
        indices = self.indices()
        # create respective simulations
        simulations = []
        for index_list in indices:
            sim_new = deepcopy(self.simulation)

            # TODO: support additional simulation types (assuming Timecourse here)
            # changes are mixed in the first timecourse
            tc = sim_new.timecourses[0]
            for k_dim, k_index in enumerate(index_list):
                # add all changes for the given dimension and index
                changes = self.dimensions[k_dim].changes
                for key in changes.keys():
                    value = changes[key][k_index]
                    tc.add_change(key, value)

            simulations.append(sim_new)

        return indices, simulations

    def normalize(self, udict: Dict, ureg: UnitRegistry):
        # normalize simulation
        self.simulation.normalize(udict=udict, ureg=ureg)

        # normalize changes in all dimensions
        for scan_dim in self.dimensions:
            self.changes = Units.normalize_changes(scan_dim.changes, udict, ureg)
            self.normalized = True


if __name__ == "__main__":
    from sbmlsim.simulation import TimecourseSim, Timecourse
    import numpy as np
    import warnings
    from pint import Quantity, UnitRegistry

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Quantity([])

    ureg = UnitRegistry()
    Q_ = ureg.Quantity
    udict = {k: "dimensionless" for k in ['X', '[X]', 'n', 'Y']}

    scan2d = ScanSim(
        simulation=TimecourseSim([
            Timecourse(start=0, end=100, steps=100,
                       changes={'X': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100,
                       changes={'[X]': Q_(10, "dimensionless")}),
            Timecourse(start=0, end=60, steps=100,
                       changes={'X': Q_(10, "dimensionless")}),
        ]),
        dimensions=[
            Dimension("dim1", index=range(8), changes={
                'n': Q_(np.linspace(start=2, stop=10, num=8), "dimensionless"),
            }),
            Dimension("dim2", index=range(4), changes={
                'Y': Q_(np.linspace(start=10, stop=20, num=4), "dimensionless"),
            }),
        ]
    )
    print(scan2d)
    indices, sims = scan2d.to_simulations()
    scan2d.normalize(udict=udict, ureg=ureg)
