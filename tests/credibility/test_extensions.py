"""The evidence extensions and the sections of their records.

The records of pkpdsim 0.17 to 0.19 in `tests/data/credibility` are records
which pkpdsim wrote: the contexts of use of 0.19 are taken from records of
its tests, the ones of 0.17 and 0.18 are the same context with the fields
those versions wrote.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, override

import pytest

from sbmlsim.credibility import (
    FACTORS,
    ContextBlock,
    ContextContribution,
    EvidenceExtension,
    FactorRecord,
    FactorStatement,
    FactorStatus,
    ReportSection,
    ScorecardChange,
    combine_factor,
    read_acceptance,
    read_sections,
)
from sbmlsim.credibility.extensions import (
    ACCEPTANCE_EXTENSION,
    CONTEXT_BLOCKS,
    EXTENSIONS,
    LEGACY_ASSESSMENT_SECTIONS,
    LEGACY_CONTEXT_SECTIONS,
    rule_columns,
    scorecard_columns,
    scorecard_number,
    scorecard_true,
)
from sbmlsim.credibility.uncertainty import uncertainty_record_from_dict
from sbmlsim.credibility.variability import variability_record_from_dict
from tests.credibility.tables import fit_table
from tests.credibility_toy import PROJECT
from tests.credibility_toy.credibility import CREDIBILITY
from tests.credibility_toy.extensions import ARMS, Arms


def _all(key: str, fit_mapping: object) -> bool:
    return True


DATA = Path(__file__).parents[1] / "data" / "credibility"


def _read(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_the_sections_of_a_record_of_pkpdsim_0_19() -> None:
    temporal = _read("context-0.19-temporal-inputs.json")
    sections = read_sections(temporal, LEGACY_CONTEXT_SECTIONS)
    assert set(sections) == {"ratios", "temporal", "inputs"}
    assert sections["temporal"] == {"temporal": temporal["temporal"]}
    assert sections["inputs"] == {"inputs": temporal["inputs"]}
    assert sections["ratios"] == {"ratios": []}
    assert read_acceptance(temporal) == temporal["acceptance"]
    data = _read("context-0.19-ratios.json")
    ratios = read_sections(data, LEGACY_CONTEXT_SECTIONS)
    (pair,) = ratios["ratios"]["ratios"]
    assert (pair["numerator"], pair["denominator"]) == ("fm_renal", "fm_dose")
    # the rule of the ratio pairs is the last row of the acceptance
    *rows, guest = read_acceptance(data)
    assert rows == data["acceptance"]
    assert all(ACCEPTANCE_EXTENSION not in row for row in rows)
    assert guest == {**data["ratio_acceptance"], ACCEPTANCE_EXTENSION: "ratios"}
    # the core sections are no sections of an extension
    for name in ("context-0.19-uncertainty.json", "context-0.19-variability.json"):
        data = _read(name)
        assert uncertainty_record_from_dict(data["uncertainty"]) == data["uncertainty"]
        assert variability_record_from_dict(data["variability"]) == data["variability"]
        assert "uncertainty" not in read_sections(data, LEGACY_CONTEXT_SECTIONS)


@pytest.mark.parametrize("version", ["0.17", "0.18"])
def test_a_record_of_pkpdsim_0_17_and_0_18_has_no_sections(version: str) -> None:
    data = _read(f"context-{version}.json")
    assert read_sections(data, LEGACY_CONTEXT_SECTIONS) == {}
    assert read_acceptance(data) == data["acceptance"]
    assert uncertainty_record_from_dict(data.get("uncertainty")) is None
    assert variability_record_from_dict(data.get("variability")) is None


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
    rows = [{"name": "GMFE"}, {"name": "pairs", ACCEPTANCE_EXTENSION: "arms"}]
    record = {"acceptance": rows, EXTENSIONS: {}, "ratio_acceptance": {"x": 1}}
    assert read_acceptance(record) == rows


def test_an_extension_contributes_nothing_by_default() -> None:
    fits = fit_table()
    cou = CREDIBILITY.context("dose")
    nothing = EvidenceExtension()
    assert (nothing.name, nothing.label) == ("", "")
    assert not nothing.claims(Arms(observable="A", numerator=_all, denominator=_all))
    assert nothing.selection_problems(cou, fits.table, fits.lookup) == []
    assert nothing.prepare(PROJECT, CREDIBILITY, fits) is None
    assert nothing.assess_context(cou, fits.table, fits, None) is None
    assert nothing.acceptance(cou, fits.table, fits.lookup) == []
    assert nothing.assessment_section(None) is None
    assert nothing.context_report_blocks({}, None) == ()
    assert nothing.report_section({}) is None
    assert nothing.report_tables({}) == {}
    assert nothing.scorecard_columns(None) == {}
    assert nothing.regressions({}, {}) == []
    assert nothing.pool({"toy": [None]}, {"field": "health"}) is None
    # an extension overrides what it adds
    assert ARMS.claims(Arms(observable="A", numerator=_all, denominator=_all))
    assert ARMS.scorecard_columns({"pairs": [1, 2]}) == {"arms": 2}
    assert ARMS.scorecard_columns(None) == {"arms": None}
    assert ARMS.regressions({"arms": 2}, {"arms": 1}) == [
        ScorecardChange("arms", "2", "1")
    ]
    # a baseline of an older record lacks the columns
    assert ARMS.regressions({}, {"arms": 1}) == []


def test_the_scorecard_columns_of_an_extension() -> None:
    """Only the extension judges its columns, the core compares the rules."""
    assert scorecard_columns(ARMS, {"pairs": [1]}) == {"arms": 1}
    assert scorecard_columns(EvidenceExtension(), None) == {}

    class Unnamed(EvidenceExtension):
        name = "unnamed"

        @override
        def scorecard_columns(
            self, context: Mapping[str, Any] | None
        ) -> Mapping[str, object]:
            return {"pairs": 1}

    with pytest.raises(ValueError, match=r"columns \['pairs'\], not its names \[\]"):
        scorecard_columns(Unnamed(), None)
    columns = ["model", "context", "verdict", "within_2fold", "arms", "GMFE"]
    core = {"model", "context", "verdict"}
    assert rule_columns(columns, core, [ARMS]) == ["within_2fold", "GMFE"]
    assert rule_columns(columns, core, []) == ["within_2fold", "arms", "GMFE"]
    # a column of the extension which drops from 10 to 4 is no rule regression,
    # the extension judges it, also from the texts of a scorecard read back
    assert "arms" not in rule_columns(columns, core, [ARMS])
    (change,) = ARMS.regressions({"arms": "10"}, {"arms": 4})
    assert change == ScorecardChange("arms", "10", "4")
    assert ARMS.regressions({"arms": ""}, {"arms": 4}) == []
    assert ARMS.regressions({"arms": "4"}, {"arms": "10"}) == []


def test_the_values_of_a_scorecard() -> None:
    assert scorecard_number(4) == 4.0
    assert scorecard_number("4.5") == 4.5
    assert math.isnan(scorecard_number(""))
    assert math.isnan(scorecard_number(None))
    assert math.isnan(scorecard_number("n/a"))
    assert scorecard_true(True) and scorecard_true("True")
    assert not scorecard_true("False") and not scorecard_true(None)


def test_a_section_which_is_none_is_no_section() -> None:
    """`{"temporal": None}` of pkpdsim 0.19 is no record of the extension."""
    data = {**_read("context-0.19-temporal-inputs.json"), "temporal": None}
    assert "temporal" not in read_sections(data, LEGACY_CONTEXT_SECTIONS)
    old = _read("context-0.19-uncertainty.json")
    assert old["temporal"] is None
    assert "temporal" not in read_sections(old, LEGACY_CONTEXT_SECTIONS)


def test_a_contribution() -> None:
    empty = ContextContribution()
    assert (empty.section, empty.acceptance, dict(empty.quantities)) == (None, (), {})
    contribution = ContextContribution(
        section={"pairs": []},
        quantities={"A_ratio": 2},
        factors={"model_form": FactorStatement("x")},
    )
    assert contribution.quantities["A_ratio"] == 2
    assert not contribution.factors["model_form"].before_core


def test_the_blocks_of_a_context() -> None:
    assert ContextBlock("<h4>Ratio pairs</h4>").after == "acceptance"
    assert ContextBlock("<h4>Inputs</h4>", after="uncertainty").after in CONTEXT_BLOCKS
    with pytest.raises(ValueError, match="follows one of"):
        ContextBlock("<h4>x</h4>", after="evidence")
    section = ReportSection(id="studies", title="Study data", html="<table></table>")
    assert (section.id, section.title) == ("studies", "Study data")


def _factor(status: FactorStatus, statement: str = "core") -> FactorRecord:
    return FactorRecord(
        id="f", name="F", status=status.value, statement=statement, step=3
    )


def test_a_factor_of_the_core_and_the_extensions() -> None:
    """The statements before the core, the core, the statements after it."""
    assessed = _factor(FactorStatus.ASSESSED, "3 of 3 rules met.")
    assert combine_factor(assessed, []) == assessed
    combined = combine_factor(
        assessed,
        [
            FactorStatement("temporal."),
            FactorStatement("inputs.", before_core=True),
            FactorStatement("other."),
            FactorStatement("first.", before_core=True),
        ],
    )
    assert combined["statement"] == "inputs. first. 3 of 3 rules met. temporal. other."
    assert (combined["status"], combined["step"]) == ("assessed", None)
    # the statement of a factor the core does not assess is dropped
    pending = _factor(FactorStatus.NOT_ASSESSED, "will be assessed in step 3")
    combined = combine_factor(pending, [FactorStatement("2 of 2 pairs within.")])
    assert combined["statement"] == "2 of 2 pairs within."
    assert (combined["status"], combined["step"]) == ("assessed", None)
    assert combine_factor(pending, []) == pending
    # a platform statement stays first
    platform = _factor(FactorStatus.PLATFORM, "released packages.")
    combined = combine_factor(platform, [FactorStatement("the builds.")])
    assert combined["statement"] == "released packages. the builds."
    assert combined["status"] == "assessed"
    # the factors of the core as pkpdsim combined them: the inputs before the
    # variability, the temporal validation after the output comparison
    (inputs,) = [f for f in FACTORS if f.id == "equivalency_of_inputs"]
    variability = FactorRecord(
        id=inputs.id, name=inputs.name, status="assessed", statement="VPC.", step=None
    )
    combined = combine_factor(
        variability, [FactorStatement("Inputs.", before_core=True)]
    )
    assert combined["statement"] == "Inputs. VPC."
