"""The HCTZ demo of two variants, which the tests and the end to end tests serve.

A project declares the panel of a variant otherwise with
`Project.variant_panels`, laid over `Project.panels` for that variant, see
`docs/serve.md`. A real model has one SBML file per build, e.g. a male and a
female build of other reference GFRs, and each variant is its own file. The
HCTZ example has one SBML file only, and `Project` refuses two variants of one
path, so this module is a device of the demo and of its tests and no pattern
to follow: its second variant is a byte copy of the HCTZ SBML, written into a
temporary directory of the process when `PROJECT_VARIANTS` is first asked for
and removed when the process ends (`atexit`). Importing the module writes
nothing, since the project is made by the `__getattr__` of the module:

    python -m sbmlsim.serve --model hctz=examples.panels.hctz_variants:PROJECT_VARIANTS

The two variants simulate one model and differ by the declaration of the page
`patient` of `examples.panels.hctz` alone: the variant `low_reference`
derives the renal function with a reference creatinine clearance of 90 ml/min
instead of 110 and says so in the titles of its panels, which changes the
declaration and the answer of a simulation which names the variant.
"""

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

from examples.hctz_fitting import MODEL_PATH
from examples.hctz_fitting.project import PROJECT as HCTZ_PROJECT
from examples.panels.hctz import (
    HCTZ_APP,
    HCTZ_PATIENT_PLASMA,
    HCTZ_PATIENT_URINE,
    PATIENT_PAGE,
    REFERENCE_CRCL,
    renal_function,
)
from sbmlsim.panel import App, AppVariant, Panel
from sbmlsim.project import Project

#: the prefix of the temporary directory of the copy of the SBML
DIRECTORY_PREFIX = "sbmlsim-panels-"

#: the variant whose renal function is its reference at a lower creatinine
#: clearance, as two builds of a model differ
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

    The variant `LOW_REFERENCE` is a copy of the SBML of the default one, see
    the module; it is written into a temporary directory of the process, which
    is removed when the process ends.

    Returns:
        The project.
    """
    directory = Path(tempfile.mkdtemp(prefix=DIRECTORY_PREFIX))
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

    The project is made when it is asked for, by `python -m sbmlsim.serve` or
    an import of it, so importing the module writes no copy of the model, see
    `_variants_project`.

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
