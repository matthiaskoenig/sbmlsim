"""
Running typical simulation experiments with PK/PB models.
"""
from sbmlsim.model import load_model
from sbmlsim.timecourse import TimecourseSim, Timecourse, ensemble
from sbmlsim.simulation_ray import SimulatorParallel as Simulator
from sbmlsim.parametrization import ChangeSet

from sbmlsim.pkpd import pkpd
from sbmlsim.plotting_matplotlib import plt, add_line
from sbmlsim.tests.constants import MODEL_GLCWB


def somatostatin_plot(result, title=''):
    """ Reference plot."""
    # create figure
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    f.subplots_adjust(wspace=0.3, hspace=0.3)

    add_line(ax=ax1, data=result,
             xid='time', yid="Cve_som", label="somatostatin blood")
    ax1.set_ylabel("concentration [mM]")

    add_line(ax=ax2, data=result,
             xid='time', yid="Aurine_som", label="somatostatin urine", color="darkblue")
    ax2.set_ylabel("amount [mmole]")

    for ax in (ax1, ax2):
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel("time [min]")

    return f


def po_bolus(simulator, r):
    """ Oral bolus dose
    oral dose (single dose, at given start with given dose).
    Examples are the oral glucose tolerance test (OGTT) or the application of a drug orally,
    e.g., codeine, midazolam, caffeine or paracetamol.

    :return:
    """
    # set initial concentration of somatostatin in all blood compartments
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]

    # oral bolus dose
    # FIXME: dosing changesets
    changes_po_bolus = {
        'PODOSE_som': 10.0E-6,  # [mg]
    }

    # simulate
    tcsims = ensemble(
        TimecourseSim([
            Timecourse(start=0, end=1200, steps=600,
                       changes={
                           **changes_init,
                           **changes_po_bolus}
                       )
            ]),
        changeset=ChangeSet.parameter_sensitivity_changeset(r, 0.1)
    )
    return simulator.timecourses(tcsims)


def iv_bolus(simulator, r):
    """ [2] bolus injection (at given time tstart, with given dose and given injection time, e.g. 5-10 min)
        - c-peptide bolus
        - ivGTT (glucose bolus)
        - insulin injection
    """
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]
    changes_iv_bolus = {
        'IVDOSE_som': 10E-6,  # [mg]
    }
    p_changeset = ChangeSet.parameter_sensitivity_changeset(r, 0.1)

    tcsims = ensemble(
        TimecourseSim([
            Timecourse(start=0, end=1200, steps=600,
                       changes={
                           **changes_init,
                           **changes_iv_bolus}
                       )
        ]), p_changeset)

    return simulator.timecourses(tcsims)


def iv_infusion(simulator, r):
    """ [3] constant infusion (for given period, tstart, tend)
        - somatostatin infusion
        - insulin infusion
        - glucose infusion
        - glucagon infusion

    :return:
    """
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]
    tcsims = ensemble(
        TimecourseSim([
            Timecourse(start=0, end=60, steps=120, changes=changes_init),
            Timecourse(start=0, end=120, steps=240, changes={'Ri_som': 10.0E-6}),  # [mg/min],
            Timecourse(start=0, end=120, steps=240, changes={'Ri_som': 0.0}),      # [mg/min],
        ]), ChangeSet.parameter_sensitivity_changeset(r, 0.1)
    )
    return simulator.timecourses(tcsims)


def clamp(simulator, r):
    """
    clamping of substance (e.g. glucose, tstart, tend)
    - glucose clamping (euglycemic clamp)
    - insulin clamping (insulin clamp)

    :return:
    """
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]

    # FIXME: some bug in the concentrations and assignments
    # tcsims = ensemble(
    tcsims = TimecourseSim([
            Timecourse(start=0, end=60, steps=120, changes={**changes_init, **{'PODOSE_som': 1E-9}}),
            Timecourse(start=0, end=60, steps=120, changes={**{'PODOSE_som': 1E-9}}),
            Timecourse(start=0, end=120, steps=240, model_changes={'boundary_condition': {"Ave_som": True}}), # clamp venous som
            Timecourse(start=0, end=120, steps=240, model_changes={'boundary_condition': {"Ave_som": False}}),   # release venous som,
        ]) #, ChangeSet.parameter_sensitivity_changeset(r, 0.1))

    return simulator.timecourses(tcsims)


def mix(simulator, r):
    """
    [5] combination experiments
    - somatostatin infusion + c-peptide bolus
    - hyperinsulinemic, euglycemic clamp
    """
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]
    tcsims = ensemble(
        TimecourseSim([
            Timecourse(start=0, end=60, steps=120, changes=changes_init),
            Timecourse(start=0, end=60, steps=120, changes={'IVDOSE_som': 10.0E-6}),
            Timecourse(start=0, end=60, steps=240, changes={'Ri_som': 10.0E-6}),  # [mg/min],
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 10.0E-6, 'PODOSE_som': 10.0E-4}),
            Timecourse(start=0, end=120, steps=240, changes={'Ri_som': 0.0}),      # [mg/min],
        ]), ChangeSet.parameter_sensitivity_changeset(r, 0.1))

    return simulator.timecourses(tcsims)


def stepped_clamp(simulator, r):

    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]
    tcsims = ensemble(
        TimecourseSim([
            Timecourse(start=0, end=60, steps=120, changes=changes_init),
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 10.0E-6}),  # [mg/min],
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 20.0E-6}),  # [mg/min],
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 40.0E-6}),  # [mg/min],
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 80.0E-6}),  # [mg/min],
        ]), ChangeSet.parameter_sensitivity_changeset(r, 0.1)
    )
    return simulator.timecourses(tcsims)


if __name__ == "__main__":
    import sys
    r = load_model(MODEL_GLCWB)
    simulator = Simulator(MODEL_GLCWB)
    results = {}
    # for f_simulate in [clamp]:
    for f_simulate in [po_bolus, iv_bolus, iv_infusion, clamp, mix, stepped_clamp]:
        f_key = f_simulate.__name__
        print(f_key)
        result = f_simulate(simulator, r)
        print(result)
        print("results:", len(result), sys.getsizeof(result))
        results[f_key] = result

    for key, result in results.items():
        somatostatin_plot(result, title=key)
        plt.show()

    import ray
    ray.timeline(filename="timeline.json")

'''
def exlude_pkdb_parameter_filter(pid):
    """ Returns True if excluded, False otherwise

    :param pid:
    :return:
    """
    # TODO: implement
    # dose parameters
    if (pid.startswith("IVDOSE_")) or (pid.startswith("PODOSE_")):
        return True

    # physical parameters
    if (pid.startswith("Mr_")) or pid in ["R_PDB"]:
        return True
    return False
'''





