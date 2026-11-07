"""The evidence extensions and the sections of their records.

The records of pkpdsim 0.17 to 0.19 in `tests/data/credibility` are records
which pkpdsim wrote: the contexts of use of 0.19 are taken from records of
its tests, the ones of 0.17 and 0.18 are the same context with the fields
those versions wrote.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from sbmlsim.credibility import (
    ContextContribution,
    EvidenceExtension,
    read_sections,
)
from sbmlsim.credibility.extensions import (
    EXTENSIONS,
    LEGACY_ASSESSMENT_SECTIONS,
    LEGACY_CONTEXT_SECTIONS,
)
from sbmlsim.credibility.uncertainty import uncertainty_record_from_dict
from sbmlsim.credibility.variability import variability_record_from_dict
from tests.credibility.tables import fit_table
from tests.credibility_toy.credibility import CREDIBILITY
from tests.credibility_toy.extensions import ARMS

DATA = Path(__file__).parents[1] / "data" / "credibility"


def _read(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_the_sections_of_a_record_of_pkpdsim_0_19() -> None:
    temporal = _read("context-0.19-temporal-inputs.json")
    sections = read_sections(temporal, LEGACY_CONTEXT_SECTIONS)
    assert set(sections) == {"ratios", "temporal", "inputs"}
    assert sections["temporal"] == {"temporal": temporal["temporal"]}
    assert sections["inputs"] == {"inputs": temporal["inputs"]}
    assert sections["ratios"] == {"ratios": [], "ratio_acceptance": None}
    ratios = read_sections(_read("context-0.19-ratios.json"), LEGACY_CONTEXT_SECTIONS)
    (pair,) = ratios["ratios"]["ratios"]
    assert (pair["numerator"], pair["denominator"]) == ("fm_renal", "fm_dose")
    assert ratios["ratios"]["ratio_acceptance"]["name"] == "Guest"
    # the core sections are no sections of an extension
    for name in ("context-0.19-uncertainty.json", "context-0.19-variability.json"):
        data = _read(name)
        assert uncertainty_record_from_dict(data["uncertainty"]) == data["uncertainty"]  # ty: ignore[invalid-argument-type]
        assert variability_record_from_dict(data["variability"]) == data["variability"]  # ty: ignore[invalid-argument-type]
        assert "uncertainty" not in read_sections(data, LEGACY_CONTEXT_SECTIONS)


@pytest.mark.parametrize("version", ["0.17", "0.18"])
def test_a_record_of_pkpdsim_0_17_and_0_18_has_no_sections(version: str) -> None:
    data = _read(f"context-{version}.json")
    assert read_sections(data, LEGACY_CONTEXT_SECTIONS) == {}
    assert uncertainty_record_from_dict(data.get("uncertainty")) is None  # ty: ignore[invalid-argument-type]
    assert variability_record_from_dict(data.get("variability")) is None  # ty: ignore[invalid-argument-type]


def test_the_sections_of_an_assessment_of_pkpdsim_0_19() -> None:
    data = _read("assessment-0.19.json")
    sections = read_sections(data, LEGACY_ASSESSMENT_SECTIONS)
    assert set(sections) == {"studies", "inputs"}
    assert len(sections["studies"]["studies"]) == 2
    assert sections["inputs"] == {"inputs": []}


def test_the_sections_of_a_record_of_sbmlsim() -> None:
    """A record with `extensions` has its sections there, nothing is lifted."""
    data = {
        EXTENSIONS: {"arms": {"pairs": [1, 2]}},
        "temporal": {"ignored": True},
    }
    sections = read_sections(data, LEGACY_CONTEXT_SECTIONS)
    assert sections == {"arms": {"pairs": [1, 2]}}
    sections["arms"] = {}
    assert data[EXTENSIONS] == {"arms": {"pairs": [1, 2]}}


def test_an_extension_contributes_nothing_by_default() -> None:
    fits = fit_table()
    cou = CREDIBILITY.context("dose")
    nothing = EvidenceExtension()
    assert nothing.name == ""
    assert nothing.selection_problems(cou, fits.table, fits.lookup) == []
    assert nothing.prepare(None, CREDIBILITY, fits) is None  # ty: ignore[invalid-argument-type]
    assert nothing.assess_context(cou, fits.table, fits, None) is None
    assert nothing.acceptance(cou, fits.table, fits.lookup) == []
    assert nothing.assessment_section(None) is None
    assert nothing.report_tables({}) == {}
    assert nothing.report_section({}) == ""
    assert nothing.scorecard_columns(None) == {}
    assert nothing.pool({"toy": [None]}) is None
    # an extension overrides what it adds
    assert ARMS.scorecard_columns({"pairs": [1, 2]}) == {"arms": 2}
    assert ARMS.scorecard_columns(None) == {"arms": None}


def test_a_contribution() -> None:
    empty = ContextContribution()
    assert (empty.section, empty.acceptance, dict(empty.quantities)) == (None, (), {})
    contribution = ContextContribution(
        section={"pairs": []}, quantities={"A_ratio": 2}, factors={"model_form": "x"}
    )
    assert contribution.quantities["A_ratio"] == 2
    assert isinstance(pd.DataFrame(contribution.section), pd.DataFrame)
