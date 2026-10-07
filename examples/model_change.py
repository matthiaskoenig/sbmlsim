"""Examples for structural changes of a model.

`ModelChange.clamp_species` clamps a species of a loaded roadrunner instance
to a value or a formula, by a fast reaction which drives the species to it. It
changes the structure of the model, which a `Simulation` does not do: the
instance is changed between simulations.
"""

import pandas as pd
from matplotlib import pyplot as plt

from sbmlsim.model import ModelChange, RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML


def run_model_change_example1():
    """Manually clamping species.

    :return:
    """
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    r = model.r
    if r is None:
        raise ValueError("Model not loaded in roadrunner.")
    RoadrunnerSBMLModel.set_timecourse_selections(r)

    s1 = r.simulate(start=0, end=100, steps=500)
    s1 = pd.DataFrame(s1, columns=s1.colnames)

    ModelChange.clamp_species(r, "X", "10.0")
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    s2 = r.simulate(start=0, end=100, steps=500)
    s2 = pd.DataFrame(s2, columns=s2.colnames)
    s2.time = s2.time + 100.0

    ModelChange.clamp_species(r, "X", False)
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    s3 = r.simulate(start=0, end=100, steps=500)
    s3 = pd.DataFrame(s3, columns=s3.colnames)
    s3.time = s3.time + 200.0

    ModelChange.clamp_species(r, "X", True)
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    s4 = r.simulate(start=0, end=100, steps=500)
    s4 = pd.DataFrame(s4, columns=s4.colnames)
    s4.time = s4.time + 300.0

    ModelChange.clamp_species(r, "X", False)
    RoadrunnerSBMLModel.set_timecourse_selections(r)
    s5 = r.simulate(start=0, end=100, steps=500)
    s5 = pd.DataFrame(s5, columns=s5.colnames)
    s5.time = s5.time + 400.0

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    for s in [s1, s2, s3, s4, s5]:
        ax.plot(s.time, s.X, "o-")
    fig.savefig("model_change_clamp_manual.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_model_change_example1()
