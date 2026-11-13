"""The panel and the app of the HCTZ model, served by `python -m sbmlsim.serve`.

A model whose SBML carries the annotation of its panels needs no such module;
this one declares them in python and binds them to the HCTZ project with
`Project.replace`, which is what the tests of the service and the end to end
tests of sbmlsim-web serve:

    python -m sbmlsim.serve --model hctz=examples.panels.hctz:PROJECT

The time unit of the HCTZ model is minutes, which is the unit the `end` of the
simulation and the x of a result are in. The panel draws the plasma and the
urine of hydrochlorothiazide and declares the cumulative amount in the urine of
Beermann 1976 as the study data under the curve of the urine.
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

EGFR = Knob(
    id="egfr",
    label="renal function",
    target="f_renal_function",
    unit="dimensionless",
    kind="range",
    min=0.2,
    max=1.0,
    steps=5,
    default=1.0,
)
DOSE = Knob(
    id="dose",
    label="dose",
    target="PODOSE_hctz",
    unit="mg",
    kind="choice",
    choices=(
        Choice(label="12.5 mg", value=12.5),
        Choice(label="25 mg", value=25.0),
        Choice(label="50 mg", value=50.0),
    ),
    default=25.0,
)

#: the cumulative amount of hydrochlorothiazide in the urine of the fourth
#: subject of Beermann 1976 after 5 mg orally, measured in mg and delivered in
#: the unit of the curve it belongs under, the `umol` of `Aurine_hctz`
URINE = PanelData(
    id="beermann1976_urine",
    label="Beermann 1976 (5 mg, oral)",
    curve="Aurine_hctz",
    dataset="Beermann1976_Tab1A",
    x="time",
    y="value",
    sel={"label": "amount_urine_hctz5po_4"},
    conversion=MR_HCTZ,
)


#: the app of the model of the tests: one page of the one panel
HCTZ_APP = App(
    title="Hydrochlorothiazide",
    subtitle="the model of hydrochlorothiazide in renal impairment",
    pages=(AppPage(id="renal", title="Renal impairment", panels=("renal",)),),
    citation="Beermann et al. (1976)",
    doi="10.5281/zenodo.0000000",
    licenses=("CC BY-SA 4.0",),
)

HCTZ_PANEL = Panel(
    id="renal",
    title="Hydrochlorothiazide in renal impairment",
    subtitle="the plasma concentration over the renal function",
    x_label="time (min)",
    knobs=(EGFR, DOSE),
    curves=(
        PanelCurve(id="Cve_hctz", label="hydrochlorothiazide plasma", unit="umol"),
        PanelCurve(id="Aurine_hctz", label="hydrochlorothiazide urine", unit="umol"),
    ),
    data=(URINE,),
    # the panel pre-initializes the dose to zero, which the knob overrides
    simulation=PanelSimulation(
        end=END, points=POINTS, preinit={"PODOSE_hctz": 0.0, "BW": 75.0}
    ),
)

#: the HCTZ project with the panel and the app, which the service serves
PROJECT: Project = HCTZ_PROJECT.replace(
    variants={"default": MODEL_PATH},
    panels={"renal": HCTZ_PANEL},
    app=HCTZ_APP,
)
