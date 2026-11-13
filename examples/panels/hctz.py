"""The panel and the app of the HCTZ model, served by `python -m sbmlsim.serve`.

A model whose SBML carries the annotation of its panels needs no such module;
this one declares them in python and binds them to the HCTZ project with
`Project.replace`, which is what the tests of the service and the end to end
tests of sbmlsim-web serve:

    python -m sbmlsim.serve --model hctz=examples.panels.hctz:PROJECT

The time unit of the HCTZ model is minutes, which is the unit the `end` of the
simulation and the x of a result are in. A panel draws one quantity on its
axis, so each of the two pages has a panel of the plasma concentration and one
of the amount in the urine: `renal_plasma` and `renal_urine` over the renal
function, `dose_plasma` and `dose_urine` over the dose. The cumulative amount
in the urine of Beermann 1976 is the study data under the urine curve of
`renal_urine`.

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

from examples.hctz_fitting import MODEL_PATH
from examples.hctz_fitting.project import PROJECT as HCTZ_PROJECT
from sbmlsim.panel import (
    App,
    AppPage,
    Choice,
    Knob,
    Panel,
    PanelCurve,
    PanelData,
    PanelSimulation,
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

#: the app of the model of the tests: the two panels of the renal function on
#: one page and the two panels of the dose on another, which is what a switch
#: of the page of an app has something real to show; the project has one SBML
#: file, so the app declares no variants
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
    },
    app=HCTZ_APP,
)
