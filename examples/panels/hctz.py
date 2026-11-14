"""The panel and the app of the HCTZ model, served by `python -m sbmlsim.serve`.

A model whose SBML carries the annotation of its panels needs no such module;
this one declares them in python and binds them to the HCTZ project with
`Project.replace`, which is what the tests of the service and the end to end
tests of sbmlsim-web serve:

    python -m sbmlsim.serve --model hctz=examples.panels.hctz:PROJECT

The time unit of the HCTZ model is minutes, which is the unit the `end` of the
simulation and the x of a result are in. A panel draws one quantity on its
axis, so each of the three pages has a panel of the plasma concentration and
one of the amount in the urine: `renal_plasma` and `renal_urine` over the
renal function, `dose_plasma` and `dose_urine` over the dose and
`patient_plasma` and `patient_urine` of a patient. The cumulative amount in the
urine of Beermann 1976 is the study data under the urine curves of
`renal_urine` and `patient_urine`.

The page `patient` exercises the extension of the declaration: its knob of the
creatinine clearance (`CRCL`) sets no entity of the model and is read by the
formula of the derived renal function, `min(crcl / 110, 1)`
(`PanelSimulation.derived`), it carries named stops, the categories of a
renal impairment, at one of which the series of Beermann 1976 was measured,
its axes are fixed (`Panel.y_min`, `y_max`, `x_max`, the time in hours) so two
patients are compared on one scale, and its scenarios (`PATIENTS`) set the
clearance and the dose of a patient at once.

`PROJECT_VARIANTS` is the variant of a declaration: the HCTZ project of two
variants, whose page `patient` is declared otherwise in the variant
`low_reference` (`Project.variant_panels`), a renal function whose reference
is a clearance of 90 ml/min and titles which say so, as the builds of one
model differ, which a request names with its `variant`:

    python -m sbmlsim.serve --model hctz=examples.panels.hctz:PROJECT_VARIANTS

A project gives every variant a file of its own, so that variant is a copy of
the SBML, written into a temporary directory when the attribute is first asked
for and removed when the process ends; importing the module for `PROJECT`
writes nothing.

That series was measured in a healthy volunteer after 5 mg orally and says so
(`PanelData.at`), so it is drawn at that state of the knobs and under no other:
the knobs which condition it, the renal function and the dose, are therefore
the choices of `EGFR_STAGES` and of `DOSE`, which is what a panel with study
data looks like. The dose of the two panels of the page `dose` is a range
(`DOSE_RANGE`), since they draw no study data.

The plasma curves are the concentrations `[Cve_hctz]` at the sampling site and
`[Cki_plasma_hctz]` in the plasma of the kidney, both `mmol/litre` in the model
and drawn in `uM`: `Cve_hctz` is a species of the compartment `Vve` which is
not in substance units, so the bare selection is its amount and no panel about
a plasma concentration draws it. The kidney is the site the knob of the panel
changes, so the two curves separate as it is turned, by 15 % of the axis at the
normal renal function and by 7 % at the impaired one, which the arterial
concentration does not do: it is the venous one to within 1.3 %.
`Aurine_hctz` is in substance units and is the amount excreted so far, `mmol`
in the model and drawn in `umol`. The labels of the curves are the site alone,
since every panel is titled with the drug, and `y_label` names the axis the
curves of a panel share.
"""

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

from examples.hctz_fitting import MODEL_PATH
from examples.hctz_fitting.project import PROJECT as HCTZ_PROJECT
from sbmlsim.panel import (
    App,
    AppPage,
    AppVariant,
    Choice,
    Knob,
    Panel,
    PanelCurve,
    PanelData,
    PanelSimulation,
    Scenario,
)
from sbmlsim.project import Project

#: one day in the time unit of the model
END = 24 * 60.0
#: the output points of the panel, fewer than a real panel so it answers quickly
POINTS = 50

#: the inverse of the molecular weight of hydrochlorothiazide, with which the
#: mass a study measured is the substance the model carries, as the
#: experiments of the HCTZ example convert it
MR_HCTZ = "1/(297.7 g/mole)"

#: the renal function as the stages of a choice knob: the study data of the
#: `renal` panel was measured at one of them, and the state a series was
#: measured under names choices (`PanelData.at`, `Panel._check_data_at`), so a
#: panel which overlays study data declares the knobs which condition it this
#: way. The reference 1.0 is its default and the state of a healthy volunteer
EGFR_STAGES = Knob(
    id="egfr",
    label="renal function",
    target="f_renal_function",
    unit="dimensionless",
    kind="choice",
    choices=(
        Choice(label="impaired", value=0.4),
        Choice(label="mildly impaired", value=0.7),
        Choice(label="normal", value=1.0),
    ),
    default=1.0,
)
#: the doses of the tablets with the 5 mg of Beermann 1976 in front of them, so
#: the panel can be put at the dose its study data was measured at; the default
#: is the 25 mg of a tablet, where that series is therefore not drawn
DOSE = Knob(
    id="dose",
    label="dose",
    target="PODOSE_hctz",
    unit="mg",
    kind="choice",
    choices=(
        Choice(label="5 mg", value=5.0),
        Choice(label="12.5 mg", value=12.5),
        Choice(label="25 mg", value=25.0),
        Choice(label="50 mg", value=50.0),
    ),
    default=25.0,
)

#: the cumulative amount of hydrochlorothiazide in the urine of the fourth
#: subject of Beermann 1976 after 5 mg orally, measured in mg and delivered in
#: the unit of the curve it belongs under, the `umol` of `Aurine_hctz`. The
#: subject was a healthy volunteer and took 5 mg, which `at` states, so the
#: series is drawn at that state of the knobs and never under the curve of
#: another dose or of an impaired kidney
URINE = PanelData(
    id="beermann1976_urine",
    label="Beermann 1976 (5 mg, oral)",
    curve="Aurine_hctz",
    dataset="Beermann1976_Tab1A",
    x="time",
    y="value",
    sel={"label": "amount_urine_hctz5po_4"},
    at={"egfr": 1.0, "dose": 5.0},
    conversion=MR_HCTZ,
)


#: the dose as a range, the knob of the panels of the page which varies it:
#: they draw no study data, so the dose is a continuum there and not the
#: choices of the `renal` panel
DOSE_RANGE = Knob(
    id="dose",
    label="dose",
    target="PODOSE_hctz",
    unit="mg",
    kind="range",
    min=12.5,
    max=50.0,
    steps=4,
    default=25.0,
)
#: the pre-initialization the panels share, the dose being a knob
SIMULATION = PanelSimulation(
    end=END, points=POINTS, preinit={"PODOSE_hctz": 0.0, "BW": 75.0}
)

#: the creatinine clearance of a patient, the input of the renal function of
#: the page `patient`: a knob without a target, whose number the formula of
#: `PATIENT_SIMULATION` reads. Its stops are the categories of the kidney
#: function of the FDA guidance on the pharmacokinetics in patients with
#: impaired renal function (2020), which follow the KDIGO stages of the GFR:
#: normal at 90 ml/min and above, a mild impairment from 60 to 89, a moderate
#: one from 30 to 59, a severe one from 15 to 29 and kidney failure below 15;
#: each stop is a point inside its category, and the normal one is the 110
#: ml/min the renal function of the model is its reference at
CRCL = Knob(
    id="crcl",
    label="creatinine clearance",
    unit="ml/min",
    kind="range",
    min=10.0,
    max=150.0,
    steps=8,
    choices=(
        Choice(label="normal", value=110.0),
        Choice(label="mild impairment", value=75.0),
        Choice(label="moderate impairment", value=45.0),
        Choice(label="severe impairment", value=20.0),
        Choice(label="kidney failure", value=10.0),
    ),
    default=110.0,
)
#: the reference creatinine clearance of the renal function of the model
REFERENCE_CRCL = 110.0


def renal_function(reference: float) -> dict[str, str]:
    """The renal function of the model computed from the creatinine clearance.

    The formula reads the number of the knob in its unit, ml/min, and its
    value is the dimensionless `f_renal_function` of the model: the clearance
    over the one of the reference, at most 1, so a clearance above the
    reference is a healthy kidney and no better one.

    Args:
        reference: the creatinine clearance of a healthy kidney, in ml/min.

    Returns:
        The derived target of the renal function and its formula.
    """
    return {"f_renal_function": f"min(crcl / {reference:g}, 1)"}


#: the simulation of the page `patient`, whose renal function is derived and
#: whose time axis is delivered in hours, the day of `END` minutes
PATIENT_SIMULATION = SIMULATION.model_copy(
    update={"derived": renal_function(REFERENCE_CRCL), "x_unit": "hr"}
)
#: the cumulative amount of Beermann 1976 on the page `patient`: the subject
#: was a healthy volunteer, the stop `normal` of the creatinine clearance, and
#: took 5 mg, so the series is drawn at that stop and dose and under no other
PATIENT_URINE = URINE.model_copy(update={"at": {"crcl": 110.0, "dose": 5.0}})

#: the fixed value axes of the page `patient`, so two patients are compared
#: on one scale. The highest curve the knobs reach was measured with
#: `sbmlsim.serve.runner.PanelRunner.run` over a grid of the 8 points of the
#: creatinine clearance and the 4 doses: 1.63 uM in the venous plasma at 10
#: ml/min and 50 mg, the end of the range of either knob, and 94.5 umol in the
#: urine at 50 mg and a clearance of 110 ml/min or more, where the renal
#: function is the reference; the axes end at the round numbers above them
PATIENT_PLASMA_MAX = 2.0
PATIENT_URINE_MAX = 100.0
#: the end of the time axis, in the hours it is delivered in: the whole day,
#: since the plasma of a failing kidney is at two thirds of its peak after
#: 12 hours, so a shorter axis would cut the curves the page is about
PATIENT_X_MAX = 24.0

HCTZ_PATIENT_PLASMA = Panel(
    id="patient_plasma",
    title="Plasma concentration of a patient",
    subtitle="hydrochlorothiazide over the creatinine clearance of a patient",
    x_label="time (hr)",
    y_label="plasma concentration",
    y_min=0.0,
    y_max=PATIENT_PLASMA_MAX,
    x_max=PATIENT_X_MAX,
    knobs=(CRCL, DOSE),
    curves=(
        PanelCurve(id="[Cve_hctz]", label="plasma", unit="uM"),
        PanelCurve(id="[Cki_plasma_hctz]", label="kidney", unit="uM"),
    ),
    simulation=PATIENT_SIMULATION,
)

HCTZ_PATIENT_URINE = Panel(
    id="patient_urine",
    title="Urinary excretion of a patient",
    subtitle="the cumulative amount over the creatinine clearance of a patient",
    x_label="time (hr)",
    y_label="amount excreted",
    y_min=0.0,
    y_max=PATIENT_URINE_MAX,
    x_max=PATIENT_X_MAX,
    knobs=(CRCL, DOSE),
    curves=(PanelCurve(id="Aurine_hctz", label="urine", unit="umol"),),
    data=(PATIENT_URINE,),
    simulation=PATIENT_SIMULATION,
)

#: the patients a reader of the page `patient` picks, each of a creatinine
#: clearance and a dose
PATIENTS = (
    Scenario(
        id="healthy", label="healthy, 25 mg", values={"crcl": 110.0, "dose": 25.0}
    ),
    Scenario(
        id="moderate",
        label="moderate impairment, 25 mg",
        values={"crcl": 45.0, "dose": 25.0},
    ),
    Scenario(
        id="severe",
        label="severe impairment, 12.5 mg",
        values={"crcl": 20.0, "dose": 12.5},
    ),
)

#: the page of a patient, of the renal function derived from the creatinine
#: clearance, the axes fixed and the patients as scenarios
PATIENT_PAGE = AppPage(
    id="patient",
    title="Patient",
    panels=("patient_plasma", "patient_urine"),
    scenarios=PATIENTS,
)

#: the app of the model of the tests: the two panels of the renal function on
#: one page, the two panels of the dose on another and the two panels of a
#: patient on a third, which is what a switch of the page of an app has
#: something real to show; the project has one SBML file, so the app declares
#: no variants
HCTZ_APP = App(
    title="Hydrochlorothiazide",
    subtitle="the model of hydrochlorothiazide in renal impairment",
    pages=(
        AppPage(
            id="renal",
            title="Renal impairment",
            panels=("renal_plasma", "renal_urine"),
        ),
        AppPage(id="dose", title="Dose", panels=("dose_plasma", "dose_urine")),
        PATIENT_PAGE,
    ),
    citation="Beermann et al. (1976)",
    doi="10.5281/zenodo.0000000",
    licenses=("CC BY-SA 4.0",),
)

HCTZ_RENAL_PLASMA = Panel(
    id="renal_plasma",
    title="Plasma concentration over the renal function",
    subtitle="hydrochlorothiazide in the venous plasma and in the kidney",
    x_label="time (min)",
    y_label="plasma concentration",
    knobs=(EGFR_STAGES, DOSE),
    curves=(
        PanelCurve(id="[Cve_hctz]", label="plasma", unit="uM"),
        PanelCurve(id="[Cki_plasma_hctz]", label="kidney", unit="uM"),
    ),
    simulation=SIMULATION,
)

HCTZ_RENAL_URINE = Panel(
    id="renal_urine",
    title="Urinary excretion over the renal function",
    subtitle="the cumulative amount of hydrochlorothiazide in the urine",
    x_label="time (min)",
    y_label="amount excreted",
    knobs=(EGFR_STAGES, DOSE),
    curves=(PanelCurve(id="Aurine_hctz", label="urine", unit="umol"),),
    data=(URINE,),
    simulation=SIMULATION,
)

HCTZ_DOSE_PLASMA = Panel(
    id="dose_plasma",
    title="Plasma concentration over the dose",
    subtitle="hydrochlorothiazide in plasma",
    x_label="time (min)",
    y_label="plasma concentration",
    knobs=(DOSE_RANGE, EGFR_STAGES),
    curves=(PanelCurve(id="[Cve_hctz]", label="plasma", unit="uM"),),
    simulation=SIMULATION,
)

HCTZ_DOSE_URINE = Panel(
    id="dose_urine",
    title="Urinary excretion over the dose",
    subtitle="the cumulative amount of hydrochlorothiazide in the urine",
    x_label="time (min)",
    y_label="amount excreted",
    knobs=(DOSE_RANGE, EGFR_STAGES),
    curves=(PanelCurve(id="Aurine_hctz", label="urine", unit="umol"),),
    simulation=SIMULATION,
)

#: the HCTZ project with the panels and the app, which the service serves
PROJECT: Project = HCTZ_PROJECT.replace(
    variants={"default": MODEL_PATH},
    panels={
        "renal_plasma": HCTZ_RENAL_PLASMA,
        "renal_urine": HCTZ_RENAL_URINE,
        "dose_plasma": HCTZ_DOSE_PLASMA,
        "dose_urine": HCTZ_DOSE_URINE,
        "patient_plasma": HCTZ_PATIENT_PLASMA,
        "patient_urine": HCTZ_PATIENT_URINE,
    },
    app=HCTZ_APP,
)

#: the variant of `PROJECT_VARIANTS` whose renal function is its reference at
#: a lower creatinine clearance, as two builds of a model differ, e.g. the
#: male and the female build of a model of other reference GFRs
LOW_REFERENCE = "low_reference"
#: the creatinine clearance at which the renal function of that variant is its
#: reference, in ml/min
LOW_REFERENCE_CRCL = 90.0


def _low_reference(panel: Panel) -> Panel:
    """The declaration of a panel of the page `patient` in `LOW_REFERENCE`.

    Args:
        panel: the declaration of the default variant.

    Returns:
        The panel with the renal function of the lower reference and a title
        which names it.
    """
    return panel.model_copy(
        update={
            "title": f"{panel.title} (reference {LOW_REFERENCE_CRCL:g} ml/min)",
            "simulation": panel.simulation.model_copy(
                update={"derived": renal_function(LOW_REFERENCE_CRCL)}
            ),
        }
    )


#: the app of `PROJECT_VARIANTS`: the page of a patient in two variants
VARIANTS_APP = App(
    title="Hydrochlorothiazide",
    subtitle="a patient of two references of the renal function",
    pages=(PATIENT_PAGE,),
    variants=(
        AppVariant(id="default", label=f"reference {REFERENCE_CRCL:g} ml/min"),
        AppVariant(id=LOW_REFERENCE, label=f"reference {LOW_REFERENCE_CRCL:g} ml/min"),
    ),
    citation=HCTZ_APP.citation,
    doi=HCTZ_APP.doi,
    licenses=HCTZ_APP.licenses,
)


@functools.cache
def _variants_project() -> Project:
    """Get the HCTZ project of two variants whose panels are declared otherwise.

    A project gives every variant a file of its own, so the variant
    `LOW_REFERENCE` is a copy of the SBML of the default one, written into a
    temporary directory of the process when the project is first asked for
    and removed when the process ends: the two variants simulate one model
    and differ by the declaration of their panels alone
    (`Project.variant_panels`), which is what the variants of a declaration
    are about.

    Returns:
        The project.
    """
    directory = Path(tempfile.mkdtemp(prefix="sbmlsim-panels-"))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    copy = directory / f"{LOW_REFERENCE}.xml"
    shutil.copyfile(MODEL_PATH, copy)
    return HCTZ_PROJECT.replace(
        variants={"default": MODEL_PATH, LOW_REFERENCE: copy},
        panels={
            "patient_plasma": HCTZ_PATIENT_PLASMA,
            "patient_urine": HCTZ_PATIENT_URINE,
        },
        variant_panels={
            LOW_REFERENCE: {
                "patient_plasma": _low_reference(HCTZ_PATIENT_PLASMA),
                "patient_urine": _low_reference(HCTZ_PATIENT_URINE),
            }
        },
        app=VARIANTS_APP,
    )


def __getattr__(name: str) -> Project:
    """Get `PROJECT_VARIANTS`, the project of two variants, on its first use.

    It is an attribute of the module which is created when it is asked for,
    by `python -m sbmlsim.serve --model hctz=examples.panels.hctz:PROJECT_VARIANTS`
    or an import of it, so importing the module for `PROJECT` writes no copy
    of the model, see `_variants_project`.

    Args:
        name: the name of the attribute.

    Returns:
        The project of `PROJECT_VARIANTS`.

    Raises:
        AttributeError: for any other name.
    """
    if name == "PROJECT_VARIANTS":
        return _variants_project()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
