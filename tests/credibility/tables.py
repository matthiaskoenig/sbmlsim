"""Hand-built tables of the data points of the toy credibility package."""

from collections.abc import Mapping

import pandas as pd

from sbmlsim.credibility import FitTable
from sbmlsim.fit import FitMapping
from sbmlsim.fit.helpers import FitMappings
from sbmlsim.fit.metrics import metadata_dict
from tests.credibility_toy import CREDIBILITY_PATH
from tests.credibility_toy.experiments import CredA, CredB

#: the kinds of the adopted fit of the toy package
KINDS = {
    ("CredA", "fm_train"): "training",
    ("CredA", "fm_dose"): "validation",
    ("CredA", "fm_renal"): "validation",
    ("CredA", "fm_limit"): "validation",
    ("CredA", "fm_subject"): "validation",
    ("CredA", "fm_outlier"): "outlier",
    ("CredB", "fm_dose"): "training",
    ("CredB", "fm_zero"): "validation",
}
#: the number of data points of a mapping of the hand-built table, else 5
POINTS = {("CredA", "fm_limit"): 2, ("CredA", "fm_outlier"): 3}
#: the value of the data of a mapping, else 1: three times the decay and zeros
VALUES = {("CredA", "fm_limit"): 3.0, ("CredB", "fm_zero"): 0.0}


def fit_table(
    factor: float = 1.0,
    factors: Mapping[tuple[str, str], float] | None = None,
    *,
    times: bool = False,
) -> FitTable:
    """Get a table of the data points of the kinds of the toy package.

    Every prediction is 1, the decay the model predicts, times a factor.

    Args:
        factor: the factor of the predictions.
        factors: the factor of the predictions of a mapping instead of `factor`.
        times: give the points of a mapping the times `x` 0, 1, 2, ...

    Returns:
        The table with the columns `DV`, `IPRED` and `DV_SD`.
    """
    factors = factors or {}
    mappings = FitMappings(
        experiment_classes=[CredA, CredB],
        base_path=CREDIBILITY_PATH,
        data_path=CREDIBILITY_PATH,
    )

    def lookup(experiment: str, mapping: str) -> FitMapping:
        return mappings.experiments[experiment]._fit_mappings[mapping]

    rows = [
        {
            "fit": "DECAY",
            "experiment": experiment,
            "mapping": mapping,
            "kind": kind,
            "observable": "[A]",
            **metadata_dict(lookup(experiment, mapping).metadata, mapping),
            "DV": VALUES.get((experiment, mapping), 1.0),
            "IPRED": factors.get((experiment, mapping), factor),
            "DV_SD": 0.1,
            **({"x": float(k)} if times else {}),
        }
        for (experiment, mapping), kind in KINDS.items()
        for k in range(POINTS.get((experiment, mapping), 5))
    ]
    return FitTable(
        table=pd.DataFrame(rows),
        lookup=lookup,
        recorded_kinds={("DECAY", e, m): k for (e, m), k in KINDS.items()},
    )
