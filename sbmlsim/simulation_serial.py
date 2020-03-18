"""
Serial simulator.
"""
import logging
from typing import List
from pathlib import Path

from sbmlsim.simulation import SimulatorAbstract, SimulatorWorker, set_integrator_settings
from sbmlsim.models.model import load_model
from sbmlsim.result import Result
from sbmlsim.timecourse import TimecourseSim
from sbmlsim.units import Units

logger = logging.getLogger(__name__)


class SimulatorSerial(SimulatorAbstract, SimulatorWorker):
    def __init__(self, path: Path, selections: List[str] = None, **kwargs):
        """

        :param path: Path to model
        :param selections: Selections to set
        :param kwargs: integrator arguments
        """
        if path:
            self.r = load_model(path=path, selections=selections)
            set_integrator_settings(self.r, **kwargs)
            self.udict, self.ureg = Units.get_units_from_sbml(model_path=path)

        else:
            self.r = None
            self.udict = None
            self.ureg = None
            logger.warning("Simulator without model instance created!")

    def timecourses(self, simulations: List[TimecourseSim]) -> Result:
        """ Run many timecourses."""
        if isinstance(simulations, TimecourseSim):
            simulations = [simulations]

        if len(simulations) > 1:
            logger.warning("Use of SimulatorSerial to run multiple timecourses. "
                           "Use SimulatorParallel instead.")
        dfs = [self.timecourse(sim) for sim in simulations]
        return Result(dfs, self.udict, self.ureg)


if __name__ == "__main__":
    from sbmlsim.tests.constants import MODEL_REPRESSILATOR

    from sbmlsim.simulation_serial import SimulatorSerial
    from sbmlsim.result import Result
    from sbmlsim.timecourse import Timecourse, TimecourseSim
    from matplotlib import pyplot as plt

    # running first simulation
    simulator = SimulatorSerial(MODEL_REPRESSILATOR)
    result = simulator.timecourse(Timecourse(0, 100, 201, ))

    # make a copy of current model with state
    r_copy = RoadrunnerSBMLModel.copy_model(simulator.r)

    # continue simulation
    result2 = simulator.timecourse(
        TimecourseSim(Timecourse(100, 200, 201), reset=False))

    plt.plot(result.time, result.X)
    plt.plot(result2.time, result2.X)
    plt.show()

    if True:
        # r = simulator.r
        simulator2 = SimulatorSerial(path=None)
        simulator2.r = r_copy
        result3 = simulator2.timecourse(TimecourseSim(Timecourse(100, 200, 201),
                                                      reset=False))
        plt.plot(result.time, result.X)
        plt.plot(result3.time, result3.X)
        plt.show()