import logging
import numpy as np
from typing import Dict
from sbmlsim.simulation.simulation import AbstractSim

logger = logging.getLogger()


class ParameterScan(AbstractSim):
    """A parameter or initial condition scan over a AbstractSim."""

    def __init__(self, simulation: AbstractSim, scan: Dict[str, np.ndarray]):
        """Scanning a simulation.

        Parameters or initial conditions can be scanned.
        Multiple parameters will result in a multi-dimensional scan.

        :param simulation: simulation to scan over the given parameters
        :param scan: dictionary of parameters or conditions to scan
        """
        self.tcsim = simulation
        self.scan = scan

    def normalize(self, udict, ureg):
        # normalize timecourse sim
        self.tcsim.normalize(udict=udict, ureg=ureg)
        # normalize scan parameters
        logger.warning("scan parameters not normalized")
        # FIXME: implement


def scan_changeset(selector: str, values):
    """Create changeset to scan parameter.

    :param r: RoadRunner model
    :param selector: selector in model
    :param values:
    :return: changeset
    """
    return [{selector: value} for value in values]