from matplotlib import pyplot as plt

import sbmlsim
from sbmlsim import plotting_matlab as plotting
from sbmlsim.simulation import TimecourseSimulation, Timecourse
from sbmlsim.results import Result
from sbmlsim.model import clamp_species

from sbmlsim.tests.settings import MODEL_REPRESSILATOR


def run_clamp_sid():

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
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    tsim = TimecourseSimulation([
        Timecourse(start=0, end=400, steps=400, changes={"X": 10}),
        # clamp simulation
        Timecourse(start=0, end=200, steps=200,
                   model_changes={'boundary_condition': {'X': True}}),
        # free simulation
        Timecourse(start=0, end=400, steps=400,
                   model_changes={'boundary_condition': {'X': False}}),
    ])
    result = sbmlsim.timecourse(r, tsim)
    plot_result(result, "clamp experiment (400-600)")


if __name__ == "__main__":
    run_clamp_sid()
