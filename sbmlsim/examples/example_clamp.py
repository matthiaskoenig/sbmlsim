from matplotlib import pyplot as plt

from sbmlsim import plotting_matplotlib as plotting
from sbmlsim.timecourse import TimecourseSim, Timecourse
from sbmlsim.simulation_serial import SimulatorSerial as Simulator
# from sbmlsim.simulation_ray import SimulatorParallel as Simulator
from sbmlsim.result import Result
from sbmlsim.tests.constants import MODEL_REPRESSILATOR


def run_clamp():

    def plot_result(result: Result, title: str = None) -> None:
        # create figure
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        plotting.add_line(ax=ax1, data=result,
                          xid='time', yid="X", label="X")
        plotting.add_line(ax=ax1, data=result,
                          xid='time', yid="Y", label="Y", color="darkblue")

        if title:
            ax1.set_title(title)

        ax1.legend()
        plt.show()

    # reference simulation
    simulator = Simulator(MODEL_REPRESSILATOR)
    tcsim = TimecourseSim([
        Timecourse(start=0, end=220, steps=300, changes={"X": 10}),
        # clamp simulation
        Timecourse(start=0, end=200, steps=200,
                   model_changes={'boundary_condition': {'X': True}}),
        # free simulation
        Timecourse(start=0, end=400, steps=400,
                   model_changes={'boundary_condition': {'X': False}}),
    ])
    result = simulator.timecourses(tcsim)
    assert isinstance(result, Result)
    plot_result(result, "clamp experiment (220-420)")


if __name__ == "__main__":
    run_clamp()
