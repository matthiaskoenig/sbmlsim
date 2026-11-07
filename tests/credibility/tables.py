"""Hand-built tables of the data points of the toy credibility package."""

from collections.abc import Mapping
from types import MappingProxyType

import pandas as pd

from sbmlsim.credibility import (
    BATTERY_CHECKS,
    Battery,
    BatteryResult,
    BatteryStatus,
    Check,
    FitTable,
    Provenance,
    Verification,
    VerificationSummary,
)
from sbmlsim.credibility.verification import (
    CROSS_SIMULATOR,
    INTEGRATOR,
    LIMIT,
    SCALE_FLOOR,
    TIGHTENING,
    TOLERANCES,
)
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


#: the provenance of the assessments of hand-built tables
PROVENANCE = Provenance(
    model="credibility",
    title="Toy model with every kind of evidence",
    fits=["DECAY_fit"],
    versions={"sbmlsim": "0.0.0"},
    commit=None,
    dirty=None,
    date="2026-11-03T12:00:00+01:00",
)


def verification_of(
    fits: FitTable,
    factor: float = 1.0,
    *,
    factors: dict[Check, float] | None = None,
    failures: dict[Check, dict[str, str]] | None = None,
    problems: tuple[str, ...] = (),
    limit: float = LIMIT,
    battery: Battery | None = None,
) -> Verification:
    """Get a verification whose alternatives are the reference times a factor."""
    factors = factors or {}
    failures = failures or {}
    ipred = fits.table["IPRED"].astype(float)
    return Verification(
        predictions=MappingProxyType(
            {
                check: ipred * factors.get(check, factor)
                for check in (TOLERANCES, INTEGRATOR)
            }
        ),
        failures=MappingProxyType(
            {
                check: MappingProxyType(failures.get(check, {}))
                for check in (TOLERANCES, INTEGRATOR)
            }
        ),
        summary=VerificationSummary(
            tightening=TIGHTENING,
            limit=limit,
            scale_floor=SCALE_FLOOR,
            fits=[
                {
                    "fit": "DECAY",
                    "reference": "CVODE BDF, rtol 1e-06",
                    "tolerances": "CVODE BDF, rtol 1e-08",
                    "integrator": "CVODE BDF of order at most 2, rtol 1e-08",
                }
            ],
            problems=list(problems),
            battery=battery,
        ),
    )


def battery_of(failed: Check | None = None) -> Battery:
    """Get a battery without a simulation: the second simulator is not applicable.

    Args:
        failed: a check which fails with the problem `broken`, every other
            check passes.
    """
    results = []
    for check in BATTERY_CHECKS:
        if check == CROSS_SIMULATOR:
            status = BatteryStatus.NOT_APPLICABLE
        elif check == failed:
            status = BatteryStatus.FAILED
        else:
            status = BatteryStatus.PASSED
        applicable = status is not BatteryStatus.NOT_APPLICABLE
        results.append(
            BatteryResult(
                check=check,
                status=status,
                cases=4 if applicable else 0,
                largest=1e-9 if applicable else float("nan"),
                at="default: A" if applicable else "",
                problems=("broken",) if status is BatteryStatus.FAILED else (),
                statement=f"The check {check.id}.",
            )
        )
    return Battery(
        results=tuple(results),
        sections={"doses": [{"variant": "default", "case": "dose", "end": 0.1}]},
        settings="roadrunner CVODE BDF.",
    )
