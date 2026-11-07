"""The analyses of an assessment of a context of use beyond the acceptance.

The parameter uncertainty, the predictive check, the lock and the
cross-validation as `assess_table` and `assess` record them; on the hand-built
table of the toy package and on the adopted fit of the session fixture
`credibility_project`.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.credibility import (
    ContextResult,
    CredibilityAssessment,
    CrossValidation,
    FitPropagation,
    HeldOutStudy,
    LockMismatchError,
    PropagationSettings,
    VariabilitySettings,
    assess,
    assess_table,
    check_declaration,
    compare,
    evaluate_fits,
    predictive_checks,
    scorecard,
    write_outputs,
)
from sbmlsim.credibility.lock import LOCK_FILE, LOCK_REMEDY
from sbmlsim.credibility.report import variability_table
from sbmlsim.credibility.uncertainty import FitDraws
from sbmlsim.credibility.variability import variability_statement
from sbmlsim.fit.variability import Variability
from sbmlsim.project import Project
from sbmlsim.simulation.sampling import LogNormal
from tests.credibility.tables import PROVENANCE, fit_table
from tests.credibility_toy.credibility import CREDIBILITY

SETTINGS = PropagationSettings(draws=256, seed=1)


def _propagation(spread: float, failed: int = 0) -> FitPropagation:
    """Get predictions per draw, the reference times a factor of the spread."""
    table = fit_table().table
    rng = np.random.default_rng(1)
    factors = np.exp(rng.normal(0.0, spread, SETTINGS.draws))
    values = np.outer(table["IPRED"].to_numpy(dtype=float), factors)
    values[:, :failed] = np.nan
    draws = FitDraws(
        fit_id="DECAY_20261001_120000__abcde",
        pids=("k",),
        values=factors[:, None],
        units={"k": "1/min"},
        correlated=False,
        note="nothing to correlate",
    )
    return FitPropagation(
        settings=SETTINGS,
        draws={"DECAY": draws},
        predictions=pd.DataFrame(values, index=table.index),
        failed={"DECAY": failed},
    )


def _factor(context: ContextResult, factor_id: str) -> dict[str, object]:
    return dict(next(f for f in context.factors if f["id"] == factor_id))


def test_the_uncertainty_of_a_context() -> None:
    assessment = assess_table(
        CREDIBILITY, fit_table(), PROVENANCE, propagation=_propagation(0.01)
    )
    dose = assessment.context("dose")
    record = dose.uncertainty
    assert record is not None
    assert (record["n_draws"], record["seed"]) == (256, 1)
    assert [f["fit"] for f in record["fits"]] == ["DECAY"]
    assert [r["name"] for r in record["rules"]] == [a["name"] for a in dose.acceptance]
    assert all(r["decision"] == "accepted" for r in record["rules"])
    assert record["n_points"] == dose.n_evaluable
    (quantity,) = record["quantities"]
    assert quantity["band_fold"] == pytest.approx(math.exp(0.01 * 2 * 1.645), rel=0.02)
    factor = _factor(dose, "model_inputs")
    assert factor["status"] == "assessed"
    statement = str(factor["statement"])
    assert "propagated with 256 draws" in statement
    assert "carried most by" in statement
    # the one parameter carries the variance of the quantity and of the rules
    by_target = {i["target"]: i for i in record["sensitivity"]}
    assert set(by_target) == {
        quantity["quantity"],
        *(r["name"] for r in record["rules"]),
    }
    assert (by_target[quantity["quantity"]]["index"] or 0.0) > 0.8
    # the verdicts do not change, the factor is not assessed without
    plain = assess_table(CREDIBILITY, fit_table(), PROVENANCE)
    assert [c.verdict for c in assessment.contexts] == [
        c.verdict for c in plain.contexts
    ]
    assert _factor(plain.context("dose"), "model_inputs")["status"] == "not_assessed"
    assert all(c.uncertainty is None for c in plain.contexts)


def test_a_wide_uncertainty_and_failed_draws() -> None:
    assessment = assess_table(
        CREDIBILITY, fit_table(), PROVENANCE, propagation=_propagation(0.5, failed=2)
    )
    record = assessment.context("dose").uncertainty
    assert record is not None
    assert any(r["decision"] == "rejected" for r in record["rules"])
    assert all(r["n_failed"] == 2 for r in record["rules"])
    assert record["fits"][0]["n_failed"] == 2
    fold = record["quantities"][0]["band_fold"]
    assert fold is not None
    assert fold > 3


def test_the_uncertainty_in_the_record_and_the_outputs(tmp_path: Path) -> None:
    assessment = assess_table(
        CREDIBILITY, fit_table(), PROVENANCE, propagation=_propagation(0.01)
    )
    path = tmp_path / "credibility.json"
    assessment.to_json(path)
    assert CredibilityAssessment.from_json(path).to_dict() == assessment.to_dict()
    output = write_outputs(assessment, tmp_path / "out")
    rules = pd.read_csv(output / "uncertainty.tsv", sep="\t")
    assert set(rules["decision"]) == {"accepted"}
    quantities = pd.read_csv(output / "uncertainty_quantities.tsv", sep="\t")
    assert list(quantities.columns) == [
        "context",
        "quantity",
        "n_points",
        "band_fold",
        "coverage",
    ]
    sensitivity = pd.read_csv(output / "uncertainty_sensitivity.tsv", sep="\t")
    assert set(sensitivity["parameter"]) == {"k"}
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "leading parameters" in html
    assert "Parameter uncertainty" in html
    assert "P(met)" in html


def test_the_uncertainty_in_the_scorecard() -> None:
    before = scorecard(
        {
            "toy": assess_table(
                CREDIBILITY, fit_table(), PROVENANCE, propagation=_propagation(0.01)
            )
        }
    )
    assert before.set_index("context").loc["dose", "uncertainty_robust"] == True  # noqa: E712
    after = scorecard(
        {
            "toy": assess_table(
                CREDIBILITY, fit_table(), PROVENANCE, propagation=_propagation(0.5)
            )
        }
    )
    regressions = [r for r in compare(before, after) if r.what == "uncertainty_robust"]
    assert "dose" in {r.context for r in regressions}
    old = before.drop(columns=["uncertainty_robust", "band_fold"])
    assert not [r for r in compare(old, after) if r.what == "uncertainty_robust"]


def test_the_propagation_in_the_assessment(credibility_project: Project) -> None:
    assessment = assess(
        credibility_project,
        verification=False,
        uncertainty=PropagationSettings(draws=8, seed=3),
    )
    for context in assessment.contexts:
        record = context.uncertainty
        assert record is not None
        (fit,) = record["fits"]
        assert (fit["n_parameters"], fit["n_failed"]) == (1, 0)
        assert record["n_points"] == context.n_evaluable
        assert all(r["n"] == 8 for r in record["rules"])


def test_the_variability_in_the_assessment(credibility_project: Project) -> None:
    settings = VariabilitySettings(subjects=8, groups=50, seed=2)
    with pytest.raises(ValueError, match="declares no variability"):
        assess(credibility_project, verification=False, variability=settings)
    project = credibility_project.replace(
        credibility=".credibility:CREDIBILITY_VARIABILITY"
    )
    assessment = assess(project, verification=False, variability=settings)
    for context in assessment.contexts:
        record = context.variability
        assert record is not None
        assert record["subjects"] == 8
        assert record["n_points"] <= context.n_evaluable
        assert _factor(context, "equivalency_of_inputs")["status"] == "assessed"
    assert len(assessment.vpc) == len(evaluate_fits(project).table)


def test_the_variability_between_the_studies(credibility_project: Project) -> None:
    fits = evaluate_fits(credibility_project)
    variability = Variability(
        {"k": LogNormal(cv=0.1)}, between=Variability({"k": LogNormal(cv=0.2)})
    )
    settings = VariabilitySettings(subjects=4, studies=3, groups=20, seed=2)
    checks = predictive_checks(credibility_project, fits, variability, settings)
    assessment = assess_table(CREDIBILITY, fits, PROVENANCE, variability=checks)
    record = assessment.contexts[0].variability
    assert record is not None
    assert (record["studies"], record["between"]) == (3, {"k": "LogNormal(cv=0.2)"})
    assert "3 studies of 4 subjects at the adopted parameters" in (
        variability_statement(record)
    )
    assert variability_table(assessment).loc[0, "between"] == "k LogNormal(cv=0.2)"


@pytest.fixture
def rooted(credibility_project: Project, tmp_path: Path) -> Project:
    """Get the toy project with its root, where its lock lies, in a temporary directory."""
    root = tmp_path / "root"
    root.mkdir()
    return credibility_project.replace(root=root)


def test_the_assessment_writes_the_lock_into_the_out(
    rooted: Project, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    written = assess(rooted, verification=False, lock=out)
    assert written.provenance.lock == "written"
    assert (out / LOCK_FILE).is_file()
    # the root of the project is never written
    assert not (rooted.root / LOCK_FILE).exists()
    assert assess(rooted, verification=False).provenance.lock is None
    shutil.copy(out / LOCK_FILE, rooted.root / LOCK_FILE)
    assert assess(rooted, verification=False).provenance.lock == "matched"
    assert check_declaration(rooted) == []
    data = json.loads((out / LOCK_FILE).read_text(encoding="utf-8"))
    data["evidence"]["renal/renal impairment"] = []
    (rooted.root / LOCK_FILE).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(LockMismatchError, match="renal/renal impairment"):
        assess(rooted, verification=False)
    problems = check_declaration(rooted)
    assert problems[-1] == LOCK_REMEDY


def test_the_cross_validations_assess_the_model_form() -> None:
    validation = CrossValidation(
        fit="DECAY_20261001_120000__abcde",
        subset="DECAY",
        gmfe_held_out=1.1,
        gmfe_fitted=1.05,
        seed=1,
        date="2026-11-01T12:00:00+01:00",
        studies=(
            HeldOutStudy(
                study="CredA",
                n_points=5,
                gmfe_held_out=1.1,
                gmfe_fitted=1.05,
                within_2fold_held_out=100.0,
                within_2fold_fitted=100.0,
                cost=0.1,
            ),
        ),
    )
    assessment = assess_table(
        CREDIBILITY, fit_table(), PROVENANCE, cross_validations=(validation,)
    )
    for context in assessment.contexts:
        factor = _factor(context, "model_form")
        if context.n_decisive:
            assert factor["status"] == "assessed"
            assert factor["statement"] == validation.statement()
    assert assessment.cross_validations == [validation]
    plain = assess_table(CREDIBILITY, fit_table(), PROVENANCE)
    assert _factor(plain.context("dose"), "model_form")["status"] == "not_assessed"
