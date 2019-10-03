import sbmlsim

from sbmlsim.simulation import Sim, TimecourseResult, parameter_sensitivity_changeset
from sbmlsim.plots import add_line
from matplotlib import pyplot as plt

from sbmlsim.tests.settings import MODEL_REPRESSILATOR, MODEL_GLCWB

if __name__ == "__main__":
    print("simulate")
    r = sbmlsim.load_model(MODEL_REPRESSILATOR)
    changeset = parameter_sensitivity_changeset(r, sensitivity=0.5)
    results = sbmlsim.timecourse(r, Sim(tstart=0, tend=400, steps=400,
                                        changeset=changeset))

    # create figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(ax=ax1, data=results,
             xid='time', yid="X", label="X")
    add_line(ax=ax1, data=results,
             xid='time', yid="Y", label="Y", color="darkblue")

    ax1.legend()
    plt.show()

    print("simulate")
    r = sbmlsim.load_model(MODEL_GLCWB)
    changeset = parameter_sensitivity_changeset(r, sensitivity=0.5)
    results = sbmlsim.timecourse(r, Sim(tstart=0, tend=400, steps=400,
                                        changeset=changeset))

    # create figure
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(ax=ax1, data=results,
             xid='time', yid="Cve_glc", label="Cve_glc")
    add_line(ax=ax1, data=results,
              xid='time', yid="Cve_cpep", label="Cve_cpep", color="darkblue")

    ax1.legend()
    plt.show()
