"""
Running typical simulation experiments with PK/PB models.
"""
from sbmlsim.model import load_model
from sbmlsim.simulation import TimecourseSimulation, Timecourse, timecourses
from sbmlsim.parametrization import ChangeSet
from sbmlsim.pkpd import pkpd
from sbmlsim.plotting_matplotlib import plt, add_line

from sbmlsim.tests.constants import MODEL_GLCWB


def somatostatin_plot(result):
    """ Reference plot."""
    # create figure
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    f.subplots_adjust(wspace=0.3, hspace=0.3)

    ax1.set_title("Blood Somatostatin")
    add_line(ax=ax1, data=result,
             xid='time', yid="Cve_som", label="somatostatin blood")
    ax1.set_ylabel("concentration [mM]")

    ax2.set_title("Urine Somatostatin")
    add_line(ax=ax2, data=result,
             xid='time', yid="Aurine_som", label="somatostatin urine", color="darkblue")
    ax2.set_ylabel("amount [mmole]")

    for ax in (ax1, ax2):
        ax.legend()
        ax.set_xlabel("time [min]")

    return f


def po_bolus():
    """ Oral bolus dose
    oral dose (single dose, at given start with given dose).
    Examples are the oral glucose tolerance test (OGTT) or the application of a drug orally,
    e.g., codeine, midazolam, caffeine or paracetamol.

    :return:
    """
    # load whole-body glucose model
    r = load_model(MODEL_GLCWB)

    # set initial concentration of somatostatin in all blood compartments
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]

    # oral bolus dose
    # FIXME: dosing changesets
    changes_po_bolus = {
        'PODOSE_som': 10.0E-6,  # [mg]
    }

    # simulate
    tc_sims = TimecourseSimulation(
        Timecourse(start=0, end=1200, steps=600,
                   changes={
                       **changes_init,
                       **changes_po_bolus}
                   )
    )  # .ensemble(ChangeSet.parameter_sensitivity_changeset(r, 0.1))
    result = timecourses(r, tc_sims)
    f = somatostatin_plot(result)


def iv_bolus():
    """ [2] bolus injection (at given time tstart, with given dose and given injection time, e.g. 5-10 min)
        - c-peptide bolus
        - ivGTT (glucose bolus)
        - insulin injection
    """
    r = load_model(MODEL_GLCWB)
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]
    changes_iv_bolus = {
        'IVDOSE_som': 10E-6,  # [mg]
    }
    p_changeset = ChangeSet.parameter_sensitivity_changeset(r, 0.1)

    tc_sims = TimecourseSimulation(
        Timecourse(start=0, end=1200, steps=600,
                   changes={
                       **changes_init,
                       **changes_iv_bolus}
                   )
    )# .ensemble(p_changeset)
    result = timecourses(r, tc_sims)
    f = somatostatin_plot(result)
    plt.show()


def iv_infusion():
    """ [3] constant infusion (for given period, tstart, tend)
        - somatostatin infusion
        - insulin infusion
        - glucose infusion
        - glucagon infusion

    :return:
    """
    r = load_model(MODEL_GLCWB)
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]
    tc_sims = TimecourseSimulation([
            Timecourse(start=0, end=60, steps=120, changes=changes_init),
            Timecourse(start=0, end=120, steps=240, changes={'Ri_som': 10.0E-6}),  # [mg/min],
            Timecourse(start=0, end=120, steps=240, changes={'Ri_som': 0.0}),      # [mg/min],
    ])  #.ensemble(ChangeSet.parameter_sensitivity_changeset(r, 0.1))
    result = timecourses(r, tc_sims)
    somatostatin_plot(result)


def clamp():
    """
    clamping of substance (e.g. glucose, tstart, tend)
    - glucose clamping (euglycemic clamp)
    - insulin clamping (insulin clamp)

    :return:
    """
    r = load_model(MODEL_GLCWB)
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]

    # FIXME: some bug in the concentrations and assignments
    tc_sims = TimecourseSimulation([
            Timecourse(start=0, end=60, steps=120, changes={**changes_init, **{'PODOSE_som': 1E-9}}),
            Timecourse(start=0, end=120, steps=240, model_changes={'boundary_condition': {"Ave_som": True}}), # clamp venous som
            Timecourse(start=0, end=120, steps=240, model_changes={'boundary_condition': {"Ave_som": False}}),   # release venous som,
    ])  #.ensemble(ChangeSet.parameter_sensitivity_changeset(r, 0.1))
    result = timecourses(r, tc_sims)
    somatostatin_plot(result)

def mix():
    """
    [5] combination experiments
    - somatostatin infusion + c-peptide bolus
    - hyperinsulinemic, euglycemic clamp
    """
    r = load_model(MODEL_GLCWB)
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]
    tc_sims = TimecourseSimulation([
            Timecourse(start=0, end=60, steps=120, changes=changes_init),
            Timecourse(start=0, end=60, steps=120, changes={'IVDOSE_som': 10.0E-6}),
            Timecourse(start=0, end=60, steps=240, changes={'Ri_som': 10.0E-6}),  # [mg/min],
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 10.0E-6, 'PODOSE_som': 10.0E-4}),
            Timecourse(start=0, end=120, steps=240, changes={'Ri_som': 0.0}),      # [mg/min],
    ])  #.ensemble(ChangeSet.parameter_sensitivity_changeset(r, 0.1))
    result = timecourses(r, tc_sims)
    somatostatin_plot(result)

    r = load_model(MODEL_GLCWB)
    changes_init = pkpd.init_concentrations_changes(r, 'som', 0E-6)  # [0 nmol/L]
    tc_sims = TimecourseSimulation([
            Timecourse(start=0, end=60, steps=120, changes=changes_init),
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 10.0E-6}),  # [mg/min],
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 20.0E-6}),  # [mg/min],
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 40.0E-6}),  # [mg/min],
            Timecourse(start=0, end=60, steps=120, changes={'Ri_som': 80.0E-6}),  # [mg/min],
    ]).ensemble(ChangeSet.parameter_sensitivity_changeset(r, 0.1))
    result = timecourses(r, tc_sims)
    somatostatin_plot(result)


if __name__ == "__main__":
    po_bolus()
    iv_bolus()
    iv_infusion()
    clamp()
    mix()

    plt.show()
'''

# iv bolus dose
changeset_iv_bolus = {
    'PODOSE_som': 0.0,  # [mg]
    'IVDOSE_som': 10E-6,  # [mg]
    'Ri_som': 0.0,  # [mg/min]
}

# iv continious infusion
changeset_iv_infusion = {
    'IVDOSE_som': 0.0,  # [mg]
    'PODOSE_som': 0.0,  # [mg]
    'Ri_som': 10.0E-6,  # [mg/min]
}

# parameter sensitivity
# changeset = ChangeSet.parameter_sensitivity_changeset(r)

# create the changesets for the simulationns
changesets = {
    "oral bolus": {**changeset_init, **changeset_po_bolus},
    "iv bolus": {**changeset_init, **changeset_iv_bolus},
    "iv infusion": {**changeset_init, **changeset_iv_infusion},
}

for key, changeset in changesets.items():
    r.resetToOrigin()
    print("***", key, "***")
    pprint(changeset)

    # simulate
    tsim = TimecourseSimulation(tstart=0, tend=60, steps=360,
                                changeset=changeset)
    results = sbmlsim.timecourse(r, tsim)
    # plots
    plot_substance_in_tissues(results, "som", unit="nmole")
    plot_somatostatin_info(results)

    plt.show()


def clamps():
    from sbmlsim import TimecourseSimulation
    from sbmlsim.parametrization import ChangeSet
    from sbmlsim.pkpd import pkpd
    import pandas as pd
    from pyglucose.plots import plot_somatostatin_info
    from pprint import pprint

    # load whole-body glucose model
    r = sbmlsim.load_model(model_path())
    r.resetToOrigin()

    # set initial concentration of somatostatin in all blood compartments
    changeset_init = pkpd.set_initial_concentrations(r, 'som', 0E-6)  # [0 nmol/L]

    # iv continious infusion
    changeset_iv_infusion = {
        'IVDOSE_som': 0.0,  # [mg]
        'PODOSE_som': 0.0,  # [mg]
        'Ri_som': 10.0E-6,  # [mg/min]
    }

    # create the changesets for the simulationns
    changeset = {**changeset_init, **changeset_iv_infusion}
    tsim = TimecourseSimulation(tstart=0, tend=60, steps=360, changeset=changeset)
    print(tsim)

    # reference simulation
    results = sbmlsim.timecourse(r, tsim)
    plot_substance_in_tissues(results, "som", unit="nmole")

    # clamp somatostatin
    from sbmlsim.model import clamp_species
    rclamp = clamp_species(r, "Ave_som")

    results = sbmlsim.timecourse(rclamp, tsim)
    plot_substance_in_tissues(results, "som", unit="nmole")

    # free somatostatin again
    from sbmlsim.model import clamp_species
    rfree = clamp_species(rclamp, "Ave_som", boundary_condition=False)
    results = sbmlsim.timecourse(rfree, tsim)
    plot_substance_in_tissues(results, "som", unit="nmole")

    plt.show()


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





