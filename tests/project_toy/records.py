"""The adopted fit of the toy project and the models with its uncertainty, for the tests.

`adopt_toy_fit` runs the fit `DECAY` of the toy project with what an adopted
fit needs (the profiles, the Fisher information, the posterior and the report)
and adopts it into its out; `toy_model_file` writes a variant of the toy model whose rate is
the value of the adopted fit with its uncertainty, as the factory of a model
package takes it from the records.
"""

import dataclasses
import json
from pathlib import Path

from sbmlutils.factory import Parameter, create_model

from sbmlsim import run
from sbmlsim.fit.records import AdoptedRecords
from tests.project_toy.factory import U, toy_model

#: the fit of the toy project with what an adopted fit needs
TOY_FIT = {
    "project": "tests.project_toy:PROJECT",
    "subset": "DECAY",
    "repeats": 2,
    "seed": 1,
    "identifiability": True,
    "profile": {"max_points": 5},
    "posterior": True,
    "report": True,
    "mapping_figures": False,
    "adopt": True,
}


def adopt_toy_fit(out: Path) -> Path:
    """Run the fit of the toy project and adopt it into its out.

    Args:
        out: the out of the run, with the records of the fit, `fits/` and
            `adopted.json`.

    Returns:
        The records of the fit in the out, `<out>/<fit id>`.
    """
    run.execute("fit", TOY_FIT, out, workers=1)
    (fit_id,) = json.loads((out / "adopted.json").read_text(encoding="utf-8"))[
        "fits"
    ].values()
    return out / fit_id


def toy_model_file(
    path: Path,
    records: AdoptedRecords | None,
    *,
    sid: str = "toy",
    initial: float = 1.0,
    upper: float | None = None,
) -> Path:
    """Write a variant of the toy model, its rate the one of the adopted fit.

    Args:
        path: the SBML file.
        records: the records of the adopted fits; the rate of the model (0.2)
            without its uncertainty without.
        sid: the id of the model.
        initial: the initial concentration of A [mM].
        upper: another upper bound of the distribution of the rate.

    Returns:
        The SBML file.
    """
    model = toy_model(0.2)
    model.sid = sid
    model.species[0].initialConcentration = initial
    if records is not None:
        kwargs = records.parameter("k", unit=U.per_min)
        if upper is not None:
            uncertainty = dataclasses.replace(records["k"], upper_bound=upper)
            kwargs["uncertainties"] = [uncertainty.uncertainty(U.per_min)]
        model.parameters = [Parameter("k", **kwargs, name="rate")]
    path.parent.mkdir(parents=True, exist_ok=True)
    create_model(
        model=model,
        filepath=path,
        sbml_level=3,
        sbml_version=2,
        validate=False,
        create_markdown=False,
        create_antimony=False,
    )
    return path
