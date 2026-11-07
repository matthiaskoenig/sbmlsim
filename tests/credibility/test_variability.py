"""The predictive check of the decisive points (`sbmlsim.credibility.variability`).

The record of a context and its statement run on the hand-built table of the
toy package with hand-built bands; `predictive_checks` runs on its adopted fit
with a variability of the fitted rate.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.credibility import (
    PredictiveChecks,
    VariabilitySettings,
    evaluate_fits,
    predictive_checks,
    select,
)
from sbmlsim.credibility.variability import (
    VPC_COLUMNS,
    VariabilityRecord,
    _wilson,
    variability_record,
    variability_record_from_dict,
    variability_statement,
)
from sbmlsim.fit.variability import Variability
from sbmlsim.project import Project
from sbmlsim.simulation.sampling import LogNormal
from tests.credibility.tables import fit_table
from tests.credibility_toy.credibility import CREDIBILITY

DATA = Path(__file__).parents[1] / "data" / "credibility"
VARIABILITY = Variability({"k": LogNormal(cv=0.2)})


def _checks(width: float) -> PredictiveChecks:
    """Get bands of the points of the table, `width` around the observations.

    A width below 1 puts every observation outside its band.
    """
    table = fit_table().table
    dv = table["DV"].to_numpy(dtype=float)
    low, high = dv / (1 + width), dv * (1 + width)
    if width < 1:
        low, high = dv * 2.0, dv * 3.0
    points = pd.DataFrame(
        {
            "N": 4,
            "statistic": "mean",
            "PRED": table["IPRED"].to_numpy(dtype=float),
            "low": low,
            "median": dv,
            "high": high,
            "inside": [
                bool(lo <= v <= hi) for v, lo, hi in zip(dv, low, high, strict=True)
            ],
        },
        index=table.index,
    )
    return PredictiveChecks(
        settings=VariabilitySettings(subjects=16, groups=10),
        variability=VARIABILITY,
        points=points,
        vectors={"DECAY": 1},
        studies={"DECAY": 1},
        failed={"DECAY": 0},
    )


def _record(checks: PredictiveChecks, context: str = "dose") -> VariabilityRecord:
    """Get the record of the decisive points of a context of the declaration."""
    fits = fit_table()
    cou = CREDIBILITY.context(context)
    rows = pd.Index([])
    for selection in select(CREDIBILITY, fits).evidence:
        if (
            selection.context == context
            and selection.evidence.role is cou.decisive_role
        ):
            rows = rows.union(selection.rows)
    decisive = fits.table.loc[rows]
    evaluable = decisive.index[decisive["DV"].to_numpy(dtype=float) > 0]
    return variability_record(checks, decisive, evaluable, cou.quantities)


def test_the_settings() -> None:
    assert VariabilitySettings().subjects == 64
    with pytest.raises(ValueError, match="fewer than two subjects; no virtual group"):
        VariabilitySettings(subjects=1, groups=0)
    with pytest.raises(ValueError, match="not increasing"):
        VariabilitySettings(quantiles=(0.9, 0.1))
    with pytest.raises(ValueError, match="no study"):
        VariabilitySettings(studies=0)


def test_the_wilson_interval() -> None:
    low, high = _wilson(9, 10, 0.95)
    assert low == pytest.approx(0.5958, abs=1e-4)
    assert high == pytest.approx(0.9821, abs=1e-4)
    assert _wilson(10, 10, 0.95)[1] == pytest.approx(1.0)


def test_the_record_of_a_context() -> None:
    record = _record(_checks(1.0))
    assert record["n_points"] == 10
    assert record["coverage"] == 1.0
    # every observation inside: wide unless the interval of few points holds 90 %
    low = record["low"]
    assert low is not None
    assert record["agreement"] == ("wide" if low > record["expected"] else "consistent")
    assert record["targets"] == {"k": "LogNormal(cv=0.2)"}
    assert record["n_groups"] == record["n_points"]
    assert [q["quantity"] for q in record["quantities"]] == ["[A]"]
    statement = variability_statement(record)
    assert "16 subjects at the adopted parameters" in statement
    assert "hold 100 %" in statement
    # every observation outside: the bands are narrow
    assert _record(_checks(0.5))["agreement"] == "narrow"
    # a ratio quantity has no coverage of its own
    renal = _record(_checks(1.0), "renal")
    assert [q["quantity"] for q in renal["quantities"]] == ["[A]"]


def test_the_statement_without_a_point() -> None:
    record = _record(_checks(1.0))
    record.update(coverage=None, vectors=8, failed=2)
    statement = variability_statement(record)
    assert "nested in 8 draws of the parameters (2 failed)" in statement
    assert statement.endswith("No evaluable decisive point has a band.")


def test_the_predictive_checks_of_the_adopted_fit(
    credibility_project: Project,
) -> None:
    fits = evaluate_fits(credibility_project)
    settings = VariabilitySettings(subjects=8, groups=50, seed=2)
    checks = predictive_checks(credibility_project, fits, VARIABILITY, settings)
    points = checks.points
    assert list(points.index) == list(fits.table.index)
    assert set(points.columns) <= set(VPC_COLUMNS)
    assert set(points["statistic"]) == {"mean"}
    assert (points["N"] == 4).all()
    np.testing.assert_allclose(
        points["PRED"], fits.table["IPRED"].to_numpy(dtype=float), rtol=1e-6
    )
    banded = points.dropna(subset=["low"])
    assert (banded["low"] <= banded["high"]).all()
    assert checks.vectors == {"DECAY": 1}


def test_the_variability_between_the_studies(credibility_project: Project) -> None:
    fits = evaluate_fits(credibility_project)
    variability = Variability(
        {"k": LogNormal(cv=0.1)}, between=Variability({"k": LogNormal(cv=0.2)})
    )
    settings = VariabilitySettings(subjects=4, studies=3, groups=20, seed=2)
    checks = predictive_checks(credibility_project, fits, variability, settings)
    assert checks.studies == {"DECAY": 3}
    cou = CREDIBILITY.context("describe")
    decisive = fits.table[fits.table["kind"] == "training"]
    record = variability_record(checks, decisive, decisive.index, cou.quantities)
    assert (record["studies"], record["between"]) == (3, {"k": "LogNormal(cv=0.2)"})
    statement = variability_statement(record)
    assert (
        "The variability of the subjects (k LogNormal(cv=0.1)) and between the "
        "studies (k LogNormal(cv=0.2)): 3 studies of 4 subjects at the adopted "
        "parameters."
    ) in statement


def test_the_record_of_pkpdsim() -> None:
    """The predictive check of pkpdsim 0.19 is read, a record of 0.18 has none."""
    data = json.loads(
        (DATA / "context-0.19-variability.json").read_text(encoding="utf-8")
    )
    record = variability_record_from_dict(data["variability"])
    assert record == data["variability"]
    old = json.loads((DATA / "context-0.18.json").read_text(encoding="utf-8"))
    assert variability_record_from_dict(old.get("variability")) is None
