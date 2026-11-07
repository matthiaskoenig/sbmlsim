"""Tests of the assessment of the contexts of use, on the toy credibility package."""

import dataclasses
import json
import math
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from sbmlsim.credibility import (
    UNCLAIMED,
    AcceptanceResult,
    ContextOfUse,
    ContextResult,
    CouLevel,
    Coverage,
    CoverageEntry,
    Credibility,
    CredibilityAssessment,
    Domain,
    Evidence,
    EvidenceRole,
    FactorStatus,
    FitTable,
    QuantityOfInterest,
    QuantityRecord,
    Unclaimed,
    Verdict,
    assess,
    assess_table,
    describe_selection,
)
from sbmlsim.credibility.assessment import provenance, verdict
from sbmlsim.credibility.factors import FACTORS
from sbmlsim.fit.filters import NamedMappings, metadata_filter, named_filter
from sbmlsim.project import Project
from tests.credibility.tables import PROVENANCE, fit_table
from tests.credibility_toy import PROJECT
from tests.credibility_toy.credibility import CREDIBILITY
from tests.credibility_toy.fitting import DOSE, ZERO
from tests.credibility_toy.metadata import Health, Route, ToyMetaData

DATA = Path(__file__).parents[1] / "data" / "credibility"


def _assess(
    fits: FitTable, credibility: Credibility = CREDIBILITY
) -> CredibilityAssessment:
    """Assess a declaration on a table."""
    return assess_table(credibility, fits, PROVENANCE)


def _with(*contexts: ContextOfUse) -> Credibility:
    """Get the declaration of the toy package with other contexts."""
    return dataclasses.replace(CREDIBILITY, contexts=contexts)


def _context(assessment: CredibilityAssessment, id: str) -> ContextResult:
    """Get the result of a context of use."""
    (result,) = (c for c in assessment.contexts if c.id == id)
    return result


def _coverage(
    assessment: CredibilityAssessment, context: str, field: str, value: str
) -> CoverageEntry:
    """Get the coverage of a value of a context of use."""
    (entry,) = (
        e
        for e in assessment.coverage
        if (e.context, e.field, e.value) == (context, field, value)
    )
    return entry


def test_a_supported_context() -> None:
    dose = _context(_assess(fit_table()), "dose")
    assert dose.verdict is Verdict.SUPPORTED
    assert dose.verdict.label == "supported"
    assert dose.acceptance
    assert all(result["met"] is True for result in dose.acceptance)
    # fm_dose and fm_subject, 5 points each
    assert dose.n_decisive == 10


def test_a_not_supported_context() -> None:
    dose = _context(_assess(fit_table(3.0)), "dose")
    assert dose.verdict is Verdict.NOT_SUPPORTED
    assert dose.verdict.label == "not supported"
    assert any(result["met"] is False for result in dose.acceptance)


def test_an_uncovered_value_without_extrapolation() -> None:
    fits = fit_table()
    table = fits.table.assign(
        health=fits.table.health.replace("RENAL_IMPAIRMENT", "HEALTHY")
    )
    fits = dataclasses.replace(fits, table=table)
    assessment = _assess(fits)
    renal = _context(assessment, "renal")
    assert renal.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert all(result["met"] is True for result in renal.acceptance)
    uncovered = _coverage(assessment, "renal", "health", "RENAL_IMPAIRMENT")
    assert uncovered.coverage is Coverage.UNCOVERED
    assert uncovered.claimed
    assert uncovered.n_points == 0
    healthy = _coverage(assessment, "renal", "health", "HEALTHY")
    assert healthy.coverage is Coverage.VALIDATED
    assert healthy.n_points == 5

    # the 5 points of fm_renal are below the minimum of 20 of a high risk
    cou = dataclasses.replace(
        CREDIBILITY.context("renal"),
        extrapolation="the kidney is a clearance",
        min_points=5,
    )
    renal = _context(_assess(fits, _with(cou)), "renal")
    assert renal.verdict is Verdict.SUPPORTED_WITH_EXTRAPOLATION
    assert renal.verdict.label == "supported with extrapolation"


def test_a_c4_context_is_never_plainly_supported() -> None:
    """A C4 context extrapolates by definition, even on validation evidence."""
    cou = dataclasses.replace(
        CREDIBILITY.context("dose"),
        level=CouLevel.C4,
        extrapolation="the qualification of the platform",
    )
    assessment = _assess(fit_table(), _with(cou))
    dose = _context(assessment, "dose")
    assert all(result["met"] is True for result in dose.acceptance)
    assert all(
        e.coverage is Coverage.VALIDATED
        for e in assessment.coverage
        if e.context == "dose"
    )
    assert dose.verdict is Verdict.SUPPORTED_WITH_EXTRAPOLATION

    failed = _context(_assess(fit_table(3.0), _with(cou)), "dose")
    assert failed.verdict is Verdict.NOT_SUPPORTED


def test_the_verdict_of_a_level() -> None:
    met = [AcceptanceResult("within_2fold", 100.0, ">= 70", True)]
    for level in CouLevel:
        expected = (
            Verdict.SUPPORTED_WITH_EXTRAPOLATION
            if level is CouLevel.C4
            else Verdict.SUPPORTED
        )
        assert (
            verdict(10, met, True, "argued", level=level, n_evaluable=10, minimum=10)
            is expected
        )
    assert verdict(
        10, met, False, "", level=CouLevel.C2, n_evaluable=10, minimum=10
    ) is (Verdict.INSUFFICIENT_EVIDENCE)


def test_the_order_of_the_verdict() -> None:
    """No point, a failed rule, a rule not evaluable, too few points, the coverage."""
    met = [AcceptanceResult("within_2fold", 100.0, ">= 70", True)]
    failed = [AcceptanceResult("within_2fold", 10.0, ">= 70", False)]
    unknown = [AcceptanceResult("GMFE", math.nan, "<= 1.5", None)]
    c2 = CouLevel.C2

    def decide(
        n: int, rules: list[AcceptanceResult], covered: bool, argued: str = ""
    ) -> Verdict:
        return verdict(n, rules, covered, argued, level=c2, n_evaluable=n, minimum=10)

    assert decide(0, met, True, "argued") is Verdict.INSUFFICIENT_EVIDENCE
    # a failure on few points is still a failure
    assert decide(2, [*met, *failed, *unknown], True) is Verdict.NOT_SUPPORTED
    assert decide(20, [*met, *unknown], True) is Verdict.INSUFFICIENT_EVIDENCE
    assert decide(9, met, True, "argued") is Verdict.INSUFFICIENT_EVIDENCE
    assert decide(10, met, True) is Verdict.SUPPORTED
    assert decide(10, met, False, "argued") is Verdict.SUPPORTED_WITH_EXTRAPOLATION
    assert decide(10, met, False) is Verdict.INSUFFICIENT_EVIDENCE
    # the evaluable points decide the minimum, not the decisive ones
    assert (
        verdict(12, met, True, "", level=c2, n_evaluable=9, minimum=10)
        is Verdict.INSUFFICIENT_EVIDENCE
    )


def test_a_ratio_quantity_is_recorded() -> None:
    ratio = QuantityOfInterest("A_ratio", "impaired over healthy", ratio=True)
    cou = dataclasses.replace(
        CREDIBILITY.context("dose"),
        quantities=(*CREDIBILITY.context("dose").quantities, ratio),
    )
    assessment = _assess(fit_table(), _with(cou))
    dose = _context(assessment, "dose")
    # a ratio has no coverage, it is assessed in step 3
    assert dose.quantities[-1] == {
        "id": "A_ratio",
        "description": "impaired over healthy",
        "ratio": True,
        "coverage": None,
        "n_points": 0,
    }
    assert dose.quantities[0]["coverage"] == "covered"
    assert dose.uncovered_quantities == []
    assert dose.verdict is Verdict.SUPPORTED
    assert dose.n_decisive == _context(_assess(fit_table()), "dose").n_decisive


def test_a_point_without_the_field_is_no_value_outside_the_claim() -> None:
    fits = fit_table()
    table = fits.table.copy()
    subject = (table.experiment == "CredA") & (table.mapping == "fm_subject")
    table["health"] = table["health"].astype(object)
    table.loc[subject, "health"] = None
    assessment = _assess(dataclasses.replace(fits, table=table))
    assert not [e for e in assessment.coverage if e.value == ""]
    assert _coverage(assessment, "dose", "health", "HEALTHY").n_points == 5


@pytest.mark.parametrize(
    ("selected", "factors"),
    [
        pytest.param(ZERO, {}, id="zeros"),
        pytest.param(DOSE, {("CredA", "fm_dose"): math.nan}, id="failed simulations"),
    ],
)
def test_a_criterion_without_applicable_points(
    selected: NamedMappings, factors: dict[tuple[str, str], float]
) -> None:
    cou = dataclasses.replace(
        CREDIBILITY.context("dose"),
        evidence=(
            Evidence("no ratios", EvidenceRole.VALIDATION, named_filter(selected)),
        ),
    )
    result = _context(_assess(fit_table(factors=factors), _with(cou)), "dose")
    (within,) = (r for r in result.acceptance if r["name"] == "within_2fold")
    assert within["met"] is None
    assert math.isnan(within["value"])
    assert result.n_decisive == 5
    assert result.n_evaluable == 0
    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert result.verdict.label == "insufficient evidence"


def _with_value(
    fits: FitTable, mapping: str, column: str, value: float | str
) -> FitTable:
    """Get the table with a value of a column for the rows of a mapping of CredA."""
    table = fits.table.copy()
    table.loc[(table.experiment == "CredA") & (table.mapping == mapping), column] = (
        value
    )
    return dataclasses.replace(fits, table=table)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("IPRED", math.nan, id="failed simulations"),
        pytest.param("DV", 0.0, id="zeros"),
    ],
)
def test_points_without_a_ratio_cover_nothing(column: str, value: float) -> None:
    cou = dataclasses.replace(
        CREDIBILITY.context("renal"),
        evidence=(
            Evidence(
                "doses and renal",
                EvidenceRole.VALIDATION,
                named_filter({"CredA": {"fm_dose", "fm_renal"}}),
            ),
        ),
    )
    fits = _with_value(fit_table(), "fm_renal", column, value)
    assessment = _assess(fits, _with(cou))
    renal = _context(assessment, "renal")
    impaired = _coverage(assessment, "renal", "health", "RENAL_IMPAIRMENT")
    assert impaired.coverage is Coverage.UNCOVERED
    assert impaired.n_points == 0
    assert _coverage(assessment, "renal", "health", "HEALTHY").n_points == 5
    assert (renal.n_decisive, renal.n_evaluable) == (10, 5)
    assert all(result["met"] is True for result in renal.acceptance)
    assert renal.verdict is Verdict.INSUFFICIENT_EVIDENCE

    # the 10 points are below the minimum of 20 of a high risk
    covered = _assess(fit_table(), _with(dataclasses.replace(cou, min_points=10)))
    assert _context(covered, "renal").verdict is Verdict.SUPPORTED


def _quantity(result: ContextResult, id: str) -> QuantityRecord:
    """Get the record of a quantity of interest of a context of use."""
    (quantity,) = (q for q in result.quantities if q["id"] == id)
    return quantity


#: the quantity of the renal data in the tables of the quantity coverage
_B = QuantityOfInterest("B", "the concentration of B")


def _two_quantities(**kw: object) -> ContextOfUse:
    """Get the dose context with A and B, validated by the doses and the renal data."""
    return dataclasses.replace(
        CREDIBILITY.context("dose"),
        quantities=(*CREDIBILITY.context("dose").quantities, _B),
        evidence=(
            Evidence(
                "doses and renal",
                EvidenceRole.VALIDATION,
                named_filter({"CredA": {"fm_dose", "fm_renal"}}),
            ),
        ),
        **kw,
    )


def test_a_quantity_outside_the_claimed_domain_is_uncovered() -> None:
    """B is validated only in renal impairment, the domain claims healthy subjects."""
    fits = _with_value(fit_table(), "fm_renal", "observable", "[B]")
    assessment = _assess(fits, _with(_two_quantities()))
    dose = _context(assessment, "dose")
    assert _quantity(dose, "B")["coverage"] == "uncovered"
    assert _quantity(dose, "B")["n_points"] == 0
    assert _quantity(dose, "[A]")["coverage"] == "covered"
    assert _quantity(dose, "[A]")["n_points"] == 5
    assert dose.uncovered_quantities == ["B"]
    # every claimed value is validated, the rules are met on enough points
    assert all(
        e.coverage is Coverage.VALIDATED
        for e in assessment.coverage
        if e.context == "dose" and e.claimed
    )
    assert all(result["met"] is True for result in dose.acceptance)
    assert (dose.n_evaluable, dose.minimum_points) == (10, 10)
    assert dose.verdict is Verdict.INSUFFICIENT_EVIDENCE

    argued = _two_quantities(extrapolation="B follows A")
    dose = _context(_assess(fits, _with(argued)), "dose")
    assert dose.verdict is Verdict.SUPPORTED_WITH_EXTRAPOLATION

    # with the renal impairment claimed, B is covered
    domain = Domain(
        metadata={
            "route": (Route.IV,),
            "health": (Health.HEALTHY, Health.RENAL_IMPAIRMENT),
        }
    )
    dose = _context(_assess(fits, _with(_two_quantities(domain=domain))), "dose")
    assert (_quantity(dose, "B")["coverage"], _quantity(dose, "B")["n_points"]) == (
        "covered",
        5,
    )
    assert dose.verdict is Verdict.SUPPORTED


def test_a_quantity_without_applicable_points_is_uncovered() -> None:
    fits = _with_value(fit_table(), "fm_renal", "observable", "[B]")
    fits = _with_value(fits, "fm_renal", "health", "HEALTHY")
    covered = _context(_assess(fits, _with(_two_quantities())), "dose")
    assert _quantity(covered, "B")["coverage"] == "covered"
    assert covered.verdict is Verdict.SUPPORTED

    failed = _with_value(fits, "fm_renal", "IPRED", math.nan)
    cou = _two_quantities(min_points=5)
    dose = _context(_assess(failed, _with(cou)), "dose")
    assert (_quantity(dose, "B")["coverage"], _quantity(dose, "B")["n_points"]) == (
        "uncovered",
        0,
    )
    assert (dose.n_decisive, dose.n_evaluable) == (10, 5)
    assert all(result["met"] is True for result in dose.acceptance)
    assert dose.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_the_minimum_of_evaluable_points() -> None:
    """The doses alone are 5 points, the minimum of a medium risk is 10."""
    cou = dataclasses.replace(
        CREDIBILITY.context("dose"),
        evidence=(Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE)),),
    )
    dose = _context(_assess(fit_table(), _with(cou)), "dose")
    assert (dose.n_evaluable, dose.minimum_points) == (5, 10)
    assert all(result["met"] is True for result in dose.acceptance)
    assert dose.verdict is Verdict.INSUFFICIENT_EVIDENCE

    # a failure on few points is still a failure
    failed = _context(_assess(fit_table(3.0), _with(cou)), "dose")
    assert failed.verdict is Verdict.NOT_SUPPORTED

    fewer = dataclasses.replace(cou, min_points=5)
    dose = _context(_assess(fit_table(), _with(fewer)), "dose")
    assert dose.minimum_points == 5
    assert dose.verdict is Verdict.SUPPORTED


def test_the_minimum_of_the_toy_package() -> None:
    assessment = _assess(fit_table())
    assert {c.id: (c.n_evaluable, c.minimum_points) for c in assessment.contexts} == {
        "describe": (10, 5),
        "dose": (10, 10),
        "renal": (5, 20),
    }


def test_a_record_of_0_17_is_read(tmp_path: Path) -> None:
    """A record without the minimum and the coverage of the quantities."""
    data = _assess(fit_table()).to_dict()
    for context in cast("list[dict[str, Any]]", data["contexts"]):
        del context["minimum_points"]
        for quantity in context["quantities"]:
            del quantity["coverage"]
            del quantity["n_points"]
    path = tmp_path / "credibility.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    read = CredibilityAssessment.from_json(path)
    for context in read.contexts:
        assert context.minimum_points == 0
        assert {(q["coverage"], q["n_points"]) for q in context.quantities} == {
            (None, 0)
        }
        assert context.uncovered_quantities == []


def test_no_decisive_evidence_is_insufficient() -> None:
    cou = dataclasses.replace(
        CREDIBILITY.context("dose"),
        evidence=(
            Evidence(
                "fitted",
                EvidenceRole.CALIBRATION,
                metadata_filter(ToyMetaData, route={Route.IV}),
            ),
        ),
        extrapolation="the decay is linear",
    )
    result = _context(_assess(fit_table(), _with(cou)), "dose")
    assert result.n_decisive == 0
    # the calibration points of A lie in the claimed domain, a C2 context does
    # not decide on them
    assert result.quantities[0]["coverage"] == "uncovered"
    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_c1_decides_on_calibration() -> None:
    cou = dataclasses.replace(
        CREDIBILITY.context("describe"),
        domain=Domain(metadata={"route": (Route.IV,)}),
    )
    assessment = _assess(fit_table(), _with(cou))
    describe = _context(assessment, "describe")
    assert describe.verdict is Verdict.SUPPORTED
    # CredA.fm_train and CredB.fm_dose
    assert describe.n_decisive == 10
    iv = _coverage(assessment, "describe", "route", "IV")
    assert iv.coverage is Coverage.CALIBRATED
    assert iv.n_points == 10

    # a C2 context on the same evidence is not decided on calibration
    interpolate = dataclasses.replace(
        cou, level=CouLevel.C2, extrapolation="the decay is linear"
    )
    result = _context(_assess(fit_table(), _with(interpolate)), "describe")
    assert result.n_decisive == 0
    assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_the_limitations_are_reported_apart() -> None:
    assessment = _assess(fit_table())
    (limitation,) = assessment.limitations
    assert limitation.context == ""
    assert limitation.name == "scaled data"
    assert limitation.role is EvidenceRole.LIMITATION
    assert limitation.n_points == 2
    assert limitation.n_mappings == 1
    assert limitation.metrics["within_2fold"] == 0.0
    for factor in (0.01, 1.0, 3.0, 100.0):
        other = _assess(fit_table(factors={("CredA", "fm_limit"): factor}))
        assert [c.acceptance for c in other.contexts] == [
            c.acceptance for c in assessment.contexts
        ]


def test_the_limitations_of_a_context() -> None:
    cou = dataclasses.replace(
        CREDIBILITY.context("dose"),
        limitations=(
            Evidence(
                "scaled",
                EvidenceRole.LIMITATION,
                named_filter({"CredA": {"fm_limit"}}),
            ),
        ),
    )
    assessment = _assess(fit_table(), _with(cou))
    assert [(e.context, e.name) for e in assessment.limitations] == [
        ("", "scaled data"),
        ("dose", "scaled"),
    ]


def test_the_evidence() -> None:
    assessment = _assess(fit_table())
    (doses,) = (e for e in assessment.evidence if e.name == "doses")
    assert doses.context == "dose"
    assert doses.role is EvidenceRole.VALIDATION
    assert doses.selection == "CredA: fm_dose"
    assert (doses.n_points, doses.n_mappings, doses.n_studies) == (5, 1, 1)
    assert doses.metrics["GMFE"] == 1.0
    assert doses.metrics["within_2fold"] == 100.0
    (other,) = (e for e in assessment.evidence if e.name == "other healthy data")
    # the limitation is healthy and intravenous as well
    assert other.limitation_subtracted == 2
    (fitted,) = (e for e in assessment.evidence if e.context == "describe")
    assert (fitted.n_mappings, fitted.n_studies) == (2, 2)
    assert assessment.unclaimed == [
        {"fit": "DECAY", "experiment": "CredB", "mapping": "fm_zero", "n": 5}
    ]


def test_the_values_outside_the_claim() -> None:
    cou = dataclasses.replace(
        CREDIBILITY.context("dose"),
        evidence=(
            Evidence(
                "doses",
                EvidenceRole.VALIDATION,
                named_filter({"CredA": {"fm_dose", "fm_renal"}}),
            ),
        ),
    )
    assessment = _assess(fit_table(), _with(cou))
    outside = _coverage(assessment, "dose", "health", "RENAL_IMPAIRMENT")
    assert outside == CoverageEntry(
        context="dose",
        field="health",
        value="RENAL_IMPAIRMENT",
        coverage=None,
        claimed=False,
        n_points=5,
    )
    assert _coverage(assessment, "dose", "health", "HEALTHY").claimed


def test_the_factors() -> None:
    assessment = _assess(fit_table())
    assert len(FACTORS) == 13
    for context in assessment.contexts:
        assert [f["id"] for f in context.factors] == [f.id for f in FACTORS]
        assert [f["status"] for f in context.factors] == [
            f.status.value for f in FACTORS
        ]
    statuses = {f.id: (f.status, f.step) for f in FACTORS}
    assert statuses["numerical_code_verification"] == (FactorStatus.PLATFORM, None)
    assert statuses["model_inputs"] == (FactorStatus.NOT_ASSESSED, 4)
    assert statuses["output_comparison"] == (FactorStatus.ASSESSED, None)


def test_the_json_round_trip(tmp_path: Path) -> None:
    cou = dataclasses.replace(
        CREDIBILITY.context("dose"),
        evidence=(Evidence("zeros", EvidenceRole.VALIDATION, named_filter(ZERO)),),
    )
    assessment = _assess(fit_table(), _with(cou, CREDIBILITY.context("describe")))
    (zeros,) = (e for e in assessment.evidence if e.name == "zeros")
    assert math.isnan(zeros.metrics["GMFE"])
    path = tmp_path / "credibility.json"
    assessment.to_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    (written,) = (e for e in data["evidence"] if e["name"] == "zeros")
    assert written["metrics"]["GMFE"] is None
    read = CredibilityAssessment.from_json(path)
    assert read.to_dict() == assessment.to_dict()
    (zeros,) = (e for e in read.evidence if e.name == "zeros")
    assert math.isnan(zeros.metrics["GMFE"])
    assert read.contexts[0].verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert read.provenance == PROVENANCE


def test_describe_selection() -> None:
    assert (
        describe_selection(
            named_filter({"CredB": {"fm_zero", "fm_dose"}, "CredA": {"fm_dose"}})
        )
        == "CredA: fm_dose; CredB: fm_dose, fm_zero"
    )
    assert (
        describe_selection(
            metadata_filter(
                ToyMetaData,
                route={Route.PO, Route.IV},
                excluded={"health": {Health.HEALTHY}},
            )
        )
        == "route in [IV, PO] and health not in [HEALTHY]"
    )
    assert describe_selection(UNCLAIMED) == "unclaimed validation data"
    assert (
        describe_selection(Unclaimed(within=named_filter(DOSE)))
        == "unclaimed validation data within CredA: fm_dose"
    )

    def by_key(key: str, fit_mapping: object) -> bool:
        return key == "fm_dose"

    assert describe_selection(by_key) == by_key.__qualname__


def test_assess_the_toy_package(credibility_project: Project) -> None:
    assessment = assess(credibility_project, battery=False)
    assert [c.id for c in assessment.contexts] == ["describe", "dose", "renal"]
    assert assessment.provenance.fits == list(
        credibility_project.adopted_fits().values()
    )
    assert assessment.provenance.model == credibility_project.name
    # the extensions and the battery hook of the toy package are of no
    # distribution
    assert set(assessment.provenance.versions) == {
        "sbmlsim",
        "sbmlutils",
        "libroadrunner",
    }
    # renal claims healthy subjects, its evidence is of renal impairment only,
    # and it has 5 evaluable points of the 20 of its high risk
    assert {c.id: c.verdict for c in assessment.contexts} == {
        "describe": Verdict.SUPPORTED,
        "dose": Verdict.SUPPORTED,
        "renal": Verdict.INSUFFICIENT_EVIDENCE,
    }
    renal = _context(assessment, "renal")
    assert all(result["met"] is True for result in renal.acceptance)
    assert _coverage(assessment, "renal", "health", "HEALTHY").coverage is (
        Coverage.UNCOVERED
    )


@pytest.mark.parametrize("verdict", list(Verdict))
def test_the_labels(verdict: Verdict) -> None:
    assert verdict.label == verdict.value.replace("_", " ")


def test_the_provenance_of_a_release() -> None:
    """In a release the source commit is given, whether its tree was clean is unknown."""
    result = provenance(PROJECT, source_commit="0123abc")
    assert (result.commit, result.dirty) == ("0123abc", None)
    assert result.model == PROJECT.name


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def test_the_provenance_is_of_the_root_of_the_project(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    package = root / "model"
    package.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.org")
    _git(root, "config", "user.name", "t")
    (package / "credibility.py").write_text("CREDIBILITY = None\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    project = PROJECT.replace(root=package)
    (root / "other.txt").write_text("a change outside the model\n", encoding="utf-8")
    result = provenance(project)
    assert result.commit == _git(root, "rev-parse", "HEAD").strip()
    assert result.dirty is False
    (package / "notes.txt").write_text("a change of the model\n", encoding="utf-8")
    assert provenance(project).dirty is True
    assert provenance(PROJECT.replace(root=tmp_path)).commit is None


def _record(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("version", ["0.17", "0.18", "0.19"])
def test_the_records_of_pkpdsim(version: str) -> None:
    """The records pkpdsim 0.17 to 0.19 wrote are read and written again as they read."""
    data = _record(f"assessment-{version}.json")
    assessment = CredibilityAssessment.from_dict(data)
    assert [c.id for c in assessment.contexts] == ["describe", "dose", "renal"]
    assert [c.verdict for c in assessment.contexts] == [
        Verdict(c["verdict"]) for c in data["contexts"]
    ]
    again = CredibilityAssessment.from_dict(assessment.to_dict())
    assert again.to_dict() == assessment.to_dict()
    dose = assessment.context("dose")
    if version == "0.17":
        assert dose.minimum_points == 0
        assert {(q["coverage"], q["n_points"]) for q in dose.quantities} == {(None, 0)}
    else:
        assert dose.minimum_points == 10
    if version != "0.19":
        assert assessment.verification is None
        assert assessment.extensions == {}
        assert all(
            c.extensions == {} and not c.verification for c in assessment.contexts
        )
        return
    # the evidence of the extensions of pkpdsim 0.19 is read as their sections
    assert set(assessment.extensions) == {"studies", "inputs"}
    assert assessment.extensions["studies"]["studies"][0]["study"] == "CredA"
    assert set(dose.extensions) == {"ratios", "temporal", "inputs"}
    assert dose.section("temporal")["temporal"]["undated"] == ["CredA"]
    assert dose.verified is True
    # the checks of the PK battery of pkpdsim are read without their hook
    assert assessment.verification is not None
    battery = assessment.verification.battery
    assert battery is not None
    mass = battery.result("mass_balance")
    assert (mass.check.label, mass.check.factor) == (
        "mass balance",
        "numerical_code_verification",
    )
    assert battery.sections == {"fates": []}


def test_the_rule_of_the_ratio_pairs_of_pkpdsim_0_19() -> None:
    """The rule of the ratio pairs of a record of 0.19 is a row of the extension `ratios`."""
    data = _record("assessment-0.19.json")
    data["contexts"] = [_record("context-0.19-ratios.json")]
    (context,) = CredibilityAssessment.from_dict(data).contexts
    (row,) = (a for a in context.acceptance if a.get("extension"))
    assert row["extension"] == "ratios"
    assert row["name"] == data["contexts"][0]["ratio_acceptance"]["name"]
    assert set(context.extensions) >= {"ratios"}
    written = json.loads(json.dumps(context.to_dict()))
    assert "ratio_acceptance" not in written
    assert written["acceptance"][-1]["extension"] == "ratios"


def _with_arms() -> Credibility:
    """Get the declaration whose renal context pairs the renal with the dose arm."""
    from tests.credibility_toy.credibility import WITH_ARMS

    renal = dataclasses.replace(
        WITH_ARMS.context("renal"),
        evidence=(
            Evidence(
                "arms",
                EvidenceRole.VALIDATION,
                named_filter({"CredA": {"fm_dose", "fm_renal"}}),
            ),
        ),
        min_points=5,
    )
    return dataclasses.replace(
        WITH_ARMS,
        contexts=tuple(renal if c.id == "renal" else c for c in WITH_ARMS.contexts),
    )


def test_an_extension_contributes_to_a_context() -> None:
    from tests.credibility_toy.extensions import ARMS

    assessment = assess_table(
        _with_arms(),
        fit_table(),
        PROVENANCE,
        extensions=[ARMS],
        prepared={"arms": ["CredA", "CredB"]},
    )
    renal = assessment.context("renal")
    assert renal.verdict is Verdict.SUPPORTED
    *own, arms = renal.acceptance
    assert all("extension" not in a for a in own)
    assert arms == {
        "name": "arms_within",
        "value": 100.0,
        "limit": ">= 50",
        "met": True,
        "extension": "arms",
    }
    assert _quantity(renal, "A_ratio")["coverage"] == "covered"
    assert _quantity(renal, "A_ratio")["n_points"] == 1
    (pair,) = renal.section("arms")["pairs"]
    assert (pair["numerator"], pair["denominator"]) == ("fm_renal", "fm_dose")
    factors = {f["id"]: f for f in renal.factors}
    assert factors["relevance_of_quantities"]["status"] == "assessed"
    assert factors["relevance_of_quantities"]["statement"] == (
        "1 of 1 pairs of arms within 1.25-fold."
    )
    assert factors["output_comparison"]["statement"].startswith(
        f"{len(renal.acceptance)} of {len(renal.acceptance)} acceptance rules met"
    )
    assert factors["output_comparison"]["statement"].endswith(" The arms are paired.")
    # a context without an arm has no contribution
    assert assessment.context("dose").extensions == {}
    assert assessment.extensions == {"arms": {"studies": ["CredA", "CredB"]}}
    read = CredibilityAssessment.from_dict(assessment.to_dict())
    assert read.to_dict() == assessment.to_dict()

    # the rule of the extension decides with the rules of the context
    worse = assess_table(
        _with_arms(),
        fit_table(factors={("CredA", "fm_dose"): 0.7}),
        PROVENANCE,
        extensions=[ARMS],
    )
    renal = worse.context("renal")
    assert renal.acceptance[-1]["met"] is False
    assert all(a["met"] is True for a in renal.acceptance[:-1])
    assert renal.verdict is Verdict.NOT_SUPPORTED


def test_the_rules_of_an_extension_on_other_predictions() -> None:
    """The verification and the uncertainty evaluate the rule of the extension too."""
    from sbmlsim.credibility.verification import INTEGRATOR, TOLERANCES
    from tests.credibility.tables import verification_of
    from tests.credibility_toy.extensions import ARMS

    fits = fit_table()
    renal_rows = (fits.table["experiment"] == "CredA") & (
        fits.table["mapping"] == "fm_renal"
    )
    verification = verification_of(fits)
    # the alternatives of the integrator move the renal arm by 1.5-fold: the
    # ratio leaves its limit, the rules of the context stay met
    integrator = verification.predictions[INTEGRATOR].copy()
    integrator[renal_rows] *= 1.5
    verification = dataclasses.replace(
        verification,
        predictions={
            TOLERANCES: verification.predictions[TOLERANCES],
            INTEGRATOR: integrator,
        },
    )
    assessment = assess_table(
        _with_arms(), fits, PROVENANCE, verification, extensions=[ARMS]
    )
    renal = assessment.context("renal")
    checks = {r["check"]: r for r in renal.verification}
    assert checks["tolerances"]["acceptance_changed"] is False
    assert checks["integrator"]["acceptance_changed"] is True
    assert renal.verdict is Verdict.NOT_VERIFIED
