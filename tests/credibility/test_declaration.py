"""Tests of the declaration of the contexts of use."""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
import pytest

from sbmlsim.credibility import (
    DEFAULT_MIN_POINTS,
    UNCLAIMED,
    Acceptance,
    AfeRange,
    ContextOfUse,
    CouLevel,
    Credibility,
    DeclarationError,
    Domain,
    Evidence,
    EvidenceRole,
    Level,
    MaxGmfe,
    QuantityOfInterest,
    Unclaimed,
    default_acceptance,
    metadata_fields,
    model_risk,
)
from sbmlsim.fit.evaluation import Fold
from sbmlsim.fit.objects import MappingMetaData
from tests.credibility_toy.extensions import Arms, MinimumPairs
from tests.credibility_toy.metadata import Health, Route, ToyMetaData

#: the fields of the metadata of the toy package a domain can claim
FIELDS = metadata_fields(ToyMetaData)

L, M, H = Level.LOW, Level.MEDIUM, Level.HIGH


def _select(key: str, fit_mapping: object) -> bool:
    return True


def _cou(**kw: object) -> ContextOfUse:
    args: dict[str, object] = {
        "id": "dose",
        "title": "Dose",
        "question": "What follows from a dose?",
        "level": CouLevel.C2,
        "influence": M,
        "consequence": M,
        "quantities": (QuantityOfInterest("Cve", "plasma"),),
        "domain": Domain(metadata={"route": (Route.PO,)}),
        "evidence": (Evidence("v", EvidenceRole.VALIDATION, _select),),
    }
    args.update(kw)
    return ContextOfUse(**args)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("i", "c", "expected"),
    [
        (L, L, L), (L, M, L), (L, H, M),
        (M, L, L), (M, M, M), (M, H, H),
        (H, L, M), (H, M, H), (H, H, H),
    ],
)  # fmt: skip
def test_the_risk_table(i: Level, c: Level, expected: Level) -> None:
    assert model_risk(i, c) is expected


def test_the_default_acceptance() -> None:
    expected = {
        L: (["within_2fold"], [">= 70"]),
        M: (["within_2fold", "GMFE"], [">= 80", "<= 1.5"]),
        H: (["within_2fold", "within_1.5fold", "GMFE"], [">= 90", ">= 70", "<= 1.3"]),
    }
    for risk, (names, limits) in expected.items():
        rules = default_acceptance(risk)
        assert [r.name for r in rules] == names
        assert [r.limit for r in rules] == limits


def test_a_rule_evaluates_points() -> None:
    points = pd.DataFrame(
        {"DV": [1, 1, 1, 1], "IPRED": [1.1, 1.9, 2.5, 0.4], "DV_SD": np.nan}
    )
    result = Acceptance(Fold(2.0), 50.0).evaluate(points)
    assert result.value == 50.0
    assert result.met is True
    assert MaxGmfe(1.5).evaluate(points).met is False


def test_a_rule_without_applicable_points_is_not_evaluable() -> None:
    points = pd.DataFrame({"DV": [0, -1], "IPRED": [1, 1], "DV_SD": np.nan})
    result = Acceptance(Fold(2.0), 70.0).evaluate(points)
    assert math.isnan(result.value)
    assert result.met is None


def test_a_cou_takes_the_default_of_its_risk() -> None:
    cou = _cou(influence=M, consequence=L)
    assert cou.risk is L
    assert cou.rules == default_acceptance(L)


def test_the_default_minimum_of_points() -> None:
    assert dict(DEFAULT_MIN_POINTS) == {L: 5, M: 10, H: 20}
    for (influence, consequence), expected in (
        ((L, L), 5),
        ((M, M), 10),
        ((M, H), 20),
    ):
        cou = _cou(influence=influence, consequence=consequence)
        assert cou.min_points is None
        assert cou.minimum_points == expected


def test_a_cou_overrides_the_minimum_of_points() -> None:
    cou = _cou(influence=H, consequence=H, min_points=3)
    assert Credibility(contexts=(cou,)).context("dose").minimum_points == 3


@pytest.mark.parametrize("min_points", [0, -1])
def test_a_minimum_of_points_is_positive(min_points: int) -> None:
    (problem,) = _problems(_cou(min_points=min_points))
    assert f"'dose': the minimum of points must be at least 1, not {min_points}" in (
        problem
    )


def test_the_declaration_reports_every_problem() -> None:
    good = _cou()
    bad = _cou(
        id="Bad-Id",
        question="Not a question.",
        quantities=(),
        evidence=(
            Evidence("u1", EvidenceRole.VALIDATION, UNCLAIMED),
            Evidence("u2", EvidenceRole.VALIDATION, UNCLAIMED),
            Evidence("u3", EvidenceRole.CALIBRATION, UNCLAIMED),
        ),
        domain=Domain(metadata={"dose": (Route.PO,), "health": (Route.PO,)}),
    )
    no_validation = _cou(
        id="nov", evidence=(Evidence("c", EvidenceRole.CALIBRATION, _select),)
    )
    no_calibration = _cou(
        id="noc",
        level=CouLevel.C1,
        evidence=(Evidence("v", EvidenceRole.VALIDATION, _select),),
    )
    with pytest.raises(DeclarationError) as err:
        Credibility(
            contexts=(good, good, bad, no_validation, no_calibration),
            limitations=(Evidence("l", EvidenceRole.VALIDATION, _select),),
        )
    text = "\n".join(err.value.problems)
    for needle in [
        "'dose': the id is not unique",
        "'Bad-Id': the id must match",
        "'Bad-Id': the question must be a question",
        "'Bad-Id': at least one quantity",
        "'Bad-Id': at most one unclaimed",
        "'Bad-Id': the unclaimed evidence 'u3' must have the role validation",
        "'nov': a C2 context needs validation evidence",
        "'noc': a C1 context needs calibration evidence",
        "model limitation 'l' must have the role limitation",
    ]:
        assert needle in text, needle
    assert len(err.value.problems) == 9
    # the domains are checked against the fields of the metadata of a project
    problems = bad.domain_problems(FIELDS)
    assert len(problems) == 2
    assert "'Bad-Id': the domain field 'dose'" in problems[0]
    assert "use one of route, health" in problems[0]
    assert "'Bad-Id': the domain value" in problems[1]
    assert "is not a member of Health" in problems[1]


def test_a_c4_cou_needs_only_its_extrapolation() -> None:
    cou = _cou(
        level=CouLevel.C4,
        evidence=(Evidence("c", EvidenceRole.CALIBRATION, _select),),
        extrapolation="platform qualification",
        domain=Domain(metadata={"route": [Route.PO], "health": [Health.HEALTHY]}),
    )
    assert Credibility(contexts=(cou,)).context("dose") is cou


def test_the_label_of_a_level() -> None:
    assert CouLevel.C2.label == "interpolate"
    assert CouLevel.C2.title() == "C2"


def _problems(cou: ContextOfUse) -> list[str]:
    """Get the problems of a declaration of one context of use."""
    with pytest.raises(DeclarationError) as err:
        Credibility(contexts=(cou,))
    return err.value.problems


def test_a_question_ends_with_a_question_mark() -> None:
    (problem,) = _problems(_cou(question="What? A statement."))
    assert "the question must be a question" in problem
    cou = _cou(question="What follows from a dose?  ")
    assert Credibility(contexts=(cou,)).context("dose") is cou


def test_a_cou_needs_a_quantity_which_is_no_ratio() -> None:
    ratio = QuantityOfInterest("Cve_ratio", "impaired over healthy", ratio=True)
    assert ratio.pairing is None
    (problem,) = _problems(_cou(quantities=(ratio,)))
    assert "'dose': at least one quantity of interest which is no ratio" in problem
    cou = _cou(quantities=(QuantityOfInterest("Cve", "plasma"), ratio))
    assert Credibility(contexts=(cou,)).context("dose") is cou


@pytest.mark.parametrize("level", [CouLevel.C2, CouLevel.C3, CouLevel.C4, CouLevel.C5])
def test_a_cou_above_c1_claims_metadata(level: CouLevel) -> None:
    (problem,) = _problems(
        _cou(level=level, domain=Domain(statement="12.5 to 100 mg"), extrapolation="x")
    )
    assert f"'dose': a {level} context claims at least one value" in problem


def test_a_c1_cou_may_claim_no_metadata() -> None:
    cou = _cou(
        level=CouLevel.C1,
        domain=Domain(),
        evidence=(Evidence("c", EvidenceRole.CALIBRATION, _select),),
    )
    assert Credibility(contexts=(cou,)).context("dose") is cou


def test_a_c5_cou() -> None:
    cou = _cou(level=CouLevel.C5, quantities=(QuantityOfInterest("SBP", "pressure"),))
    assert Credibility(contexts=(cou,)).context("dose").level is CouLevel.C5
    (problem,) = _problems(
        _cou(
            level=CouLevel.C5,
            evidence=(Evidence("c", EvidenceRole.CALIBRATION, _select),),
        )
    )
    assert "a C5 context needs validation evidence" in problem


def test_a_c2_cou_with_extrapolation_and_no_validation() -> None:
    cou = _cou(
        evidence=(Evidence("c", EvidenceRole.CALIBRATION, _select),),
        extrapolation="the decay is linear in the dose",
    )
    assert Credibility(contexts=(cou,)).context("dose") is cou


def test_an_unclaimed_evidence_once_and_as_validation() -> None:
    within = Unclaimed(within=_select)
    cou = _cou(evidence=(Evidence("rest", EvidenceRole.VALIDATION, within),))
    assert Credibility(contexts=(cou,)).context("dose") is cou


def test_the_decisive_role() -> None:
    assert _cou(level=CouLevel.C1).decisive_role is EvidenceRole.CALIBRATION
    for level in (CouLevel.C2, CouLevel.C3, CouLevel.C4, CouLevel.C5):
        assert _cou(level=level).decisive_role is EvidenceRole.VALIDATION


def test_an_unknown_context_is_a_key_error() -> None:
    credibility = Credibility(contexts=(_cou(),))
    with pytest.raises(KeyError, match=r"known: \['dose'\]"):
        credibility.context("renal")


@pytest.mark.parametrize(
    ("ipred", "met"),
    [
        pytest.param(0.8, True, id="the low bound"),
        pytest.param(1.25, True, id="the high bound"),
        pytest.param(1.0, True, id="inside"),
        pytest.param(0.79, False, id="below"),
        pytest.param(1.26, False, id="above"),
    ],
)
def test_the_afe_range_includes_its_bounds(ipred: float, met: bool) -> None:
    points = pd.DataFrame({"DV": [1.0, 1.0], "IPRED": [ipred, ipred], "DV_SD": np.nan})
    result = AfeRange(0.8, 1.25).evaluate(points)
    assert result.value == pytest.approx(ipred)
    assert result.met is met
    assert result.limit == "0.8 to 1.25"


def test_an_afe_range_without_applicable_points_is_not_evaluable() -> None:
    points = pd.DataFrame({"DV": [0.0, -1.0], "IPRED": [1.0, 1.0], "DV_SD": np.nan})
    result = AfeRange(0.8, 1.25).evaluate(points)
    assert math.isnan(result.value)
    assert result.met is None


def test_the_fields_of_the_metadata() -> None:
    assert {"route": Route, "health": Health} == FIELDS
    assert metadata_fields(ToyMetaData, ("health",)) == {"health": Health}
    assert metadata_fields(MappingMetaData) == {}
    with pytest.raises(ValueError, match=r"\['label', 'dose'\] are no fields"):
        metadata_fields(ToyMetaData, ("label", "dose"))


class Tissue(StrEnum):
    PLASMA = "plasma"


@dataclass(kw_only=True)
class OptionalMetaData(MappingMetaData):
    tissue: Tissue | None = None
    count: int = 0


def test_an_optional_enum_is_a_field() -> None:
    assert metadata_fields(OptionalMetaData) == {"tissue": Tissue}
    domain = Domain(metadata={"tissue": (Tissue.PLASMA,)})
    assert domain.problems(metadata_fields(OptionalMetaData)) == []


def test_the_metadata_without_fields() -> None:
    domain = Domain(metadata={"route": (Route.PO,)})
    (problem,) = domain.problems(metadata_fields(MappingMetaData))
    assert "use one of none, it has no fields" in problem
    assert Domain().problems({}) == []


def test_a_ratio_with_a_pairing() -> None:
    arms = Arms(observable="[Cve]", numerator=_select, denominator=_select)
    ratio = QuantityOfInterest("Cve_ratio", "impaired over healthy", ratio=arms)
    assert ratio.pairing is arms
    cou = _cou(quantities=(QuantityOfInterest("Cve", "plasma"), ratio))
    assert cou.ratios == (ratio,)
    assert Credibility(contexts=(cou,)).context("dose") is cou
    empty = Arms(observable=" ", numerator=_select, denominator=_select)
    (problem,) = _problems(
        _cou(
            quantities=(
                QuantityOfInterest("Cve", "plasma"),
                QuantityOfInterest("Cve_ratio", "ratio", ratio=empty),
            )
        )
    )
    assert "'dose': the ratio 'Cve_ratio' needs the observable of its arms" in problem


def test_the_settings_of_the_extensions() -> None:
    cou = _cou(extensions={"arms": MinimumPairs(2), "other": "a plain setting"})
    assert Credibility(contexts=(cou,)).context("dose").extensions["arms"] == (
        MinimumPairs(2)
    )
    with pytest.raises(TypeError):
        cou.extensions["arms"] = MinimumPairs(3)  # ty: ignore[invalid-assignment]
    (problem,) = _problems(_cou(extensions={"arms": MinimumPairs(0)}))
    assert problem == (
        "context of use 'dose': the setting of the extension 'arms': the minimum "
        "0 is below 1"
    )


def test_no_covariates_by_default() -> None:
    assert dict(Credibility(contexts=(_cou(),)).covariates) == {}
    declared = Credibility(contexts=(_cou(),), covariates={"BW": "weight"})
    assert dict(declared.covariates) == {"BW": "weight"}
    with pytest.raises(DeclarationError, match="covariate ' '"):
        Credibility(contexts=(_cou(),), covariates={" ": "weight"})
