from sbmlsim.tests.constants import MODEL_REPRESSILATOR
from sbmlsim.models import RoadrunnerSBMLModel

from sbmlsim.simulation_serial import SimulatorSerial
from sbmlsim.result import Result
from sbmlsim.timecourse import Timecourse, TimecourseSim

from matplotlib import pyplot as plt

# running first simulation
simulator = SimulatorSerial(MODEL_REPRESSILATOR)
result = simulator.timecourse(Timecourse(0, 100, 201, ))


# continue simulation
result2 = simulator.timecourse(
    TimecourseSim(Timecourse(100, 200, 201), reset=False))

plt.plot(result.time, result.X)
plt.plot(result2.time, result2.X)
plt.show()


"""

# make a copy of current model with state
r_copy = RoadrunnerSBMLModel.copy_model(simulator.r)
if True:
    # r = simulator.r
    simulator2 = SimulatorSerial(path=None)
    simulator2.r = r_copy
    result3 = simulator2.timecourse(TimecourseSim(Timecourse(100, 200, 201),
                                                  reset=False))
    plt.plot(result.time, result.X)
    plt.plot(result3.time, result3.X)
    plt.show()
"""