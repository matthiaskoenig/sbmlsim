from pathlib import Path
from pint import Quantity

from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial

# run example simulation
simulator = SimulatorSerial(Path(__file__).parent / "parameter_fitting" / "simple_reaction.xml")
Q_ = Quantity
tcsim = TimecourseSim(timecourses=[
    Timecourse(start=0, end=50, steps=11,
               changes={
                   "[A]": Q_(10, "mM"),
                   "[B]": Q_(0, "mM")
               })
    ]
)
s = simulator._timecourse(simulation=tcsim)
print(s)


# create data for fitting
# s_data =
