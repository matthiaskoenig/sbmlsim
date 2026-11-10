"""Tests of the kinds of the adopted fits and of the selection of the evidence."""

import dataclasses
import shutil
from pathlib import Path

import pandas as pd
import pytest

from sbmlsim.credibility import (
    UNCLAIMED,
    ContextOfUse,
    CouLevel,
    Credibility,
    Domain,
    Evidence,
    EvidenceRole,
    FitTable,
    KindMismatchError,
    Level,
    QuantityOfInterest,
    SelectionError,
    Selections,
    Unclaimed,
    check_declaration,
    evaluate_fits,
    in_domain,
    kind_differences,
    mapping_table,
    metadata_values,
    normalize_observable,
    recorded_kinds,
    select,
)
from sbmlsim.fit.filters import metadata_filter, named_filter
from sbmlsim.project import CredibilitySettings, Project
from tests.credibility.tables import KINDS, fit_table
from tests.credibility_toy.credibility import CREDIBILITY
from tests.credibility_toy.extensions import ARMS, Arms
from tests.credibility_toy.fitting import DOSE
from tests.credibility_toy.metadata import Health, Route, ToyMetaData

_A = (QuantityOfInterest("A", "the concentration of A"),)


@pytest.fixture(scope="module")
def fits() -> FitTable:
    """Get a table of the data points of the kinds of the test package."""
    return fit_table()


def _cou(
    id: str,
    *evidence: Evidence,
    quantities: tuple[QuantityOfInterest, ...] = _A,
    limitations: tuple[Evidence, ...] = (),
) -> ContextOfUse:
    """Get a context of use of intravenous doses with the evidence."""
    return ContextOfUse(
        id=id,
        title="Dose",
        question="What follows from a dose?",
        level=CouLevel.C2,
        influence=Level.LOW,
        consequence=Level.LOW,
        quantities=quantities,
        domain=Domain(metadata={"route": (Route.IV,)}),
        evidence=evidence,
        limitations=limitations,
        extrapolation="the decay is linear",
    )


def _credibility(
    *evidence: Evidence,
    limitations: tuple[Evidence, ...] = (),
    quantities: tuple[QuantityOfInterest, ...] = _A,
) -> Credibility:
    """Get a declaration of one context of use with the evidence."""
    return Credibility(
        contexts=(_cou("dose", *evidence, quantities=quantities),),
        limitations=limitations,
    )


def _pairs(fits: FitTable, rows: pd.Index) -> set[tuple[str, str]]:
    """Get the experiments and mappings of rows of the table."""
    selected = fits.table.loc[rows]
    return set(zip(selected.experiment, selected.mapping, strict=True))


def _n(fits: FitTable, experiment: str, mapping: str) -> int:
    """Get the number of rows of a mapping."""
    table = fits.table
    return int(((table.experiment == experiment) & (table.mapping == mapping)).sum())


def _problems(credibility: Credibility, fits: FitTable) -> str:
    """Get the message of the selection error of a declaration."""
    with pytest.raises(SelectionError) as err:
        select(credibility, fits)
    assert err.value.problems
    return str(err.value)


def test_normalize_observable() -> None:
    assert normalize_observable("[A]") == "A"
    assert normalize_observable("A") == "A"
    assert normalize_observable("pk.cmax") == "pk.cmax"


def test_a_validation_on_training_data_is_an_error(fits: FitTable) -> None:
    message = _problems(
        _credibility(
            Evidence(
                "fitted",
                EvidenceRole.VALIDATION,
                named_filter({"CredB": {"fm_dose"}}),
            )
        ),
        fits,
    )
    assert "CredB.fm_dose" in message


def test_the_key_alone_does_not_select(fits: FitTable) -> None:
    selections = select(
        _credibility(Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE))),
        fits,
    )
    (dose,) = selections.evidence
    assert _pairs(fits, dose.rows) == {("CredA", "fm_dose")}
    assert len(dose.rows) == _n(fits, "CredA", "fm_dose")


def test_a_calibration_keeps_its_training_points(fits: FitTable) -> None:
    selections = select(
        _credibility(
            Evidence(
                "intravenous",
                EvidenceRole.CALIBRATION,
                metadata_filter(ToyMetaData, route={Route.IV}),
            )
        ),
        fits,
    )
    (calibration,) = selections.evidence
    assert _pairs(fits, calibration.rows) == {
        ("CredA", "fm_train"),
        ("CredB", "fm_dose"),
    }
    assert set(fits.table.loc[calibration.rows, "kind"]) == {"training"}


def test_outliers_are_dropped_and_counted(fits: FitTable) -> None:
    selections = select(
        _credibility(
            Evidence(
                "doses",
                EvidenceRole.VALIDATION,
                named_filter({"CredA": {"fm_dose", "fm_outlier"}}),
            )
        ),
        fits,
    )
    (dose,) = selections.evidence
    assert _pairs(fits, dose.rows) == {("CredA", "fm_dose")}
    assert dose.outliers_dropped == _n(fits, "CredA", "fm_outlier") == 3
    assert dose.limitation_subtracted == 0


def test_limitations_are_subtracted_and_counted(fits: FitTable) -> None:
    limitation = Evidence(
        "scaled", EvidenceRole.LIMITATION, named_filter({"CredA": {"fm_limit"}})
    )
    selections = select(
        _credibility(
            Evidence(
                "doses",
                EvidenceRole.VALIDATION,
                named_filter({"CredA": {"fm_dose", "fm_limit"}}),
            ),
            limitations=(limitation,),
        ),
        fits,
    )
    (dose,) = selections.evidence
    assert _pairs(fits, dose.rows) == {("CredA", "fm_dose")}
    assert dose.limitation_subtracted == _n(fits, "CredA", "fm_limit") == 2
    assert _pairs(fits, selections.limitations) == {("CredA", "fm_limit")}


def test_the_limitations_of_a_context_are_subtracted(fits: FitTable) -> None:
    credibility = _credibility(
        Evidence(
            "doses",
            EvidenceRole.VALIDATION,
            named_filter({"CredA": {"fm_dose", "fm_limit"}}),
        )
    )
    cou = dataclasses.replace(
        credibility.contexts[0],
        limitations=(
            Evidence(
                "scaled",
                EvidenceRole.LIMITATION,
                named_filter({"CredA": {"fm_limit"}}),
            ),
        ),
    )
    selections = select(Credibility(contexts=(cou,)), fits)
    (dose,) = selections.evidence
    assert _pairs(fits, dose.rows) == {("CredA", "fm_dose")}
    assert dose.limitation_subtracted == 2


def test_a_limitation_of_another_context_is_subtracted(fits: FitTable) -> None:
    """A limitation of one context of use is no evidence of another one."""
    scaled = Evidence(
        "scaled", EvidenceRole.LIMITATION, named_filter({"CredA": {"fm_limit"}})
    )
    limited = _cou(
        "limited",
        Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE)),
        limitations=(scaled,),
    )
    other = _cou(
        "other",
        Evidence(
            "doses and scaled",
            EvidenceRole.VALIDATION,
            named_filter({"CredA": {"fm_dose", "fm_limit"}}),
        ),
    )
    selections = select(Credibility(contexts=(limited, other)), fits)
    (_, evidence) = selections.evidence
    assert evidence.context == "other"
    assert _pairs(fits, evidence.rows) == {("CredA", "fm_dose")}
    assert len(evidence.rows) == _n(fits, "CredA", "fm_dose")
    assert evidence.limitation_subtracted == _n(fits, "CredA", "fm_limit") == 2


def test_a_calibration_counts_no_subtracted_validation(fits: FitTable) -> None:
    """A calibration keeps training points only, the limitation it meets is none."""
    selections = select(
        _credibility(
            Evidence(
                "intravenous",
                EvidenceRole.CALIBRATION,
                metadata_filter(ToyMetaData, route={Route.IV}),
            ),
            limitations=(
                Evidence(
                    "scaled",
                    EvidenceRole.LIMITATION,
                    named_filter({"CredA": {"fm_limit"}}),
                ),
            ),
        ),
        fits,
    )
    (calibration,) = selections.evidence
    assert calibration.limitation_subtracted == 0


def test_a_ratio_quantity_does_not_restrict_the_evidence(fits: FitTable) -> None:
    """A declared ratio without a pairing matches no observable and selects nothing."""
    doses = Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE))
    ratio = QuantityOfInterest("A_ratio", "impaired over healthy", ratio=True)
    (dose,) = select(_credibility(doses, quantities=(*_A, ratio)), fits).evidence
    assert _pairs(fits, dose.rows) == {("CredA", "fm_dose")}
    assert len(dose.rows) == _n(fits, "CredA", "fm_dose")


def test_a_fitted_limitation_is_an_error(fits: FitTable) -> None:
    message = _problems(
        _credibility(
            Evidence(
                "doses", EvidenceRole.VALIDATION, named_filter({"CredA": {"fm_dose"}})
            ),
            limitations=(
                Evidence(
                    "fitted",
                    EvidenceRole.LIMITATION,
                    named_filter({"CredA": {"fm_train"}}),
                ),
            ),
        ),
        fits,
    )
    assert "CredA.fm_train" in message
    assert "fitted" in message


def test_unclaimed_selects_the_rest(fits: FitTable) -> None:
    selections = select(CREDIBILITY, fits)
    (unclaimed,) = (
        s
        for s in selections.evidence
        if s.context == "dose" and isinstance(s.evidence.selection, Unclaimed)
    )
    assert _pairs(fits, unclaimed.rows) == {("CredA", "fm_subject")}
    assert unclaimed.limitation_subtracted == _n(fits, "CredA", "fm_limit")

    dose = CREDIBILITY.context("dose")
    everything = dataclasses.replace(
        dose,
        evidence=(
            dose.evidence[0],
            Evidence("the rest", EvidenceRole.VALIDATION, UNCLAIMED),
        ),
    )
    credibility = dataclasses.replace(
        CREDIBILITY,
        contexts=tuple(
            everything if c.id == "dose" else c for c in CREDIBILITY.contexts
        ),
    )
    (unclaimed,) = (
        s for s in select(credibility, fits).evidence if s.evidence.name == "the rest"
    )
    assert _pairs(fits, unclaimed.rows) == {
        ("CredA", "fm_subject"),
        ("CredB", "fm_zero"),
    }


def test_the_unclaimed_validation_data(fits: FitTable) -> None:
    selections = select(CREDIBILITY, fits)
    assert isinstance(selections, Selections)
    assert selections.unclaimed.to_dict("records") == [
        {
            "fit": "DECAY",
            "experiment": "CredB",
            "mapping": "fm_zero",
            "n": _n(fits, "CredB", "fm_zero"),
        }
    ]
    assert _pairs(fits, selections.limitations) == {("CredA", "fm_limit")}


def test_quantities_match_without_brackets(fits: FitTable) -> None:
    doses = Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE))
    for quantity in ("A", "[A]"):
        (dose,) = select(
            _credibility(
                doses, quantities=(QuantityOfInterest(quantity, "the quantity"),)
            ),
            fits,
        ).evidence
        assert _pairs(fits, dose.rows) == {("CredA", "fm_dose")}

    message = _problems(
        _credibility(doses, quantities=(*_A, QuantityOfInterest("B", "no observable"))),
        fits,
    )
    assert "'B'" in message


def test_an_unknown_fit_subset_is_an_error(fits: FitTable) -> None:
    message = _problems(
        _credibility(
            Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE), fit="PD")
        ),
        fits,
    )
    assert "'PD'" in message
    assert "['DECAY']" in message
    assert "selects no point" not in message


def test_the_fit_of_an_evidence_restricts_it(fits: FitTable) -> None:
    other = fits.table.assign(fit="PK")
    table = pd.concat([fits.table, other], ignore_index=True)
    both = dataclasses.replace(fits, table=table)
    doses = Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE), fit="PK")
    (dose,) = select(_credibility(doses), both).evidence
    assert set(table.loc[dose.rows, "fit"]) == {"PK"}
    assert len(dose.rows) == _n(fits, "CredA", "fm_dose")


def test_a_mapping_fitted_in_one_fit_is_fitted_in_every_fit(fits: FitTable) -> None:
    # CredA fm_train is training of DECAY and validation of PD
    other = fits.table.assign(fit="PD")
    other.loc[other.mapping == "fm_train", "kind"] = "validation"
    table = pd.concat([fits.table, other], ignore_index=True)
    both = dataclasses.replace(fits, table=table)
    train = named_filter({"CredA": {"fm_train"}})

    for fit in ("PD", None):
        message = _problems(
            _credibility(Evidence("fitted", EvidenceRole.VALIDATION, train, fit=fit)),
            both,
        )
        assert "CredA.fm_train (PD, training in DECAY)" in message

    selections = select(
        _credibility(
            Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE)),
            Evidence("the rest", EvidenceRole.VALIDATION, UNCLAIMED),
        ),
        both,
    )
    (_, rest) = selections.evidence
    assert ("CredA", "fm_train") not in _pairs(both, rest.rows)
    assert ("CredA", "fm_subject") in _pairs(both, rest.rows)
    assert "fm_train" not in set(selections.unclaimed.mapping)

    message = _problems(
        _credibility(
            Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE)),
            limitations=(Evidence("fitted", EvidenceRole.LIMITATION, train, fit="PD"),),
        ),
        both,
    )
    assert "training in DECAY" in message


def test_an_empty_evidence_is_an_error(fits: FitTable) -> None:
    message = _problems(
        _credibility(
            Evidence(
                "nothing",
                EvidenceRole.VALIDATION,
                named_filter({"CredA": {"fm_nothing"}}),
            )
        ),
        fits,
    )
    assert "'dose'" in message
    assert "'nothing'" in message
    assert "DECAY" in message


def test_every_problem_is_raised_at_once(fits: FitTable) -> None:
    with pytest.raises(SelectionError) as err:
        select(
            _credibility(
                Evidence(
                    "fitted",
                    EvidenceRole.VALIDATION,
                    named_filter({"CredB": {"fm_dose"}}),
                ),
                Evidence("pd", EvidenceRole.VALIDATION, named_filter(DOSE), fit="PD"),
            ),
            fits,
        )
    assert len(err.value.problems) == 2


def test_kind_differences(fits: FitTable) -> None:
    assert kind_differences(fits.table, fits.recorded_kinds) == []
    recorded = {**fits.recorded_kinds, ("DECAY", "CredA", "fm_dose"): "training"}
    (message,) = kind_differences(fits.table, recorded)
    assert "DECAY" in message
    assert "CredA" in message
    assert "fm_dose" in message

    missing = {k: v for k, v in fits.recorded_kinds.items() if k[2] != "fm_zero"} | {
        ("DECAY", "CredC", "fm_new"): "validation"
    }
    messages = kind_differences(fits.table, missing)
    assert len(messages) == 2
    assert any("CredB.fm_zero" in m for m in messages)
    assert any("CredC.fm_new" in m for m in messages)


def test_recorded_kinds(credibility_project: Project, tmp_path: Path) -> None:
    (fit_id,) = credibility_project.adopted_fits().values()
    record = credibility_project.adopted_record(fit_id)
    assert recorded_kinds(record, "DECAY") == {
        ("DECAY", e, m): k for (e, m), k in KINDS.items()
    }

    with pytest.raises(FileNotFoundError, match="adopt the fit again"):
        recorded_kinds(tmp_path, "DECAY")

    table = pd.read_csv(record / "metrics_mappings.tsv", sep="\t")
    twice = pd.concat([table, table.assign(kind="training")], ignore_index=True)
    twice.to_csv(tmp_path / "metrics_mappings.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="fm_dose"):
        recorded_kinds(tmp_path, "DECAY")


def _copied(project: Project, fits_dir: Path) -> tuple[Project, str]:
    """Get the project with a copy of its fits and the id of its adopted fit."""
    assert project.fits_dir is not None
    shutil.copytree(project.fits_dir, fits_dir)
    (fit_id,) = project.adopted_fits().values()
    return project.replace(fits_dir=fits_dir), fit_id


def _edited_project(project: Project, fits_dir: Path) -> Project:
    """Get the project with a copy of its fits whose record has another kind."""
    edited_project, fit_id = _copied(project, fits_dir)
    path = fits_dir / fit_id / "metrics_mappings.tsv"
    table = pd.read_csv(path, sep="\t")
    edited = (table.experiment == "CredA") & (table.mapping == "fm_dose")
    table.loc[edited, "kind"] = "training"
    table.to_csv(path, sep="\t", index=False)
    return edited_project


def test_evaluate_fits(credibility_project: Project, tmp_path: Path) -> None:
    fits = evaluate_fits(credibility_project)
    table = fits.table
    assert {"fit", "DV", "IPRED", "DV_SD", "observable"} <= set(table.columns)
    assert set(table.fit) == {"DECAY"}
    pairs = zip(table.experiment, table.mapping, strict=True)
    kinds = dict(zip(pairs, table.kind, strict=True))
    assert kinds == KINDS
    assert kind_differences(table, fits.recorded_kinds) == []
    assert fits.lookup("CredB", "fm_zero").metadata.route is Route.PO  # ty: ignore[missing-attribute]
    assert set(fits.reports) == {"DECAY"}
    assert fits.lookup("CredA", "fm_renal").metadata.health is Health.RENAL_IMPAIRMENT  # ty: ignore[missing-attribute]

    # the static table has the same mappings with the same observables
    static = mapping_table(credibility_project)
    columns = ["fit", "experiment", "mapping", "kind", "observable", "route", "health"]
    assert sorted(map(tuple, static.table[columns].to_numpy().tolist())) == sorted(
        map(tuple, table[columns].drop_duplicates().to_numpy().tolist())
    )
    assert static.recorded_kinds == fits.recorded_kinds

    edited = _edited_project(credibility_project, tmp_path / "fits")
    with pytest.raises(KindMismatchError, match="adopt") as err:
        evaluate_fits(edited)
    (problem,) = err.value.problems
    assert "CredA.fm_dose" in problem


def test_check_declaration_of_the_test_package(
    credibility_project: Project, tmp_path: Path
) -> None:
    assert check_declaration(credibility_project) == []

    broken = credibility_project.replace(credibility=".credibility:BROKEN")
    messages = check_declaration(broken)
    assert len(messages) == 1
    assert "CredB" in messages[0]

    edited = _edited_project(broken, tmp_path / "fits")
    messages = check_declaration(edited)
    assert any("CredA.fm_dose" in m and "training" in m for m in messages)
    assert any("CredB.fm_dose" in m for m in messages)
    assert len([m for m in messages if "adopt the fit again" in m]) == 1

    (message,) = check_declaration(
        credibility_project.replace(adopted=tmp_path / "none.json")
    )
    assert "no adopted fits" in message

    missing, fit_id = _copied(credibility_project, tmp_path / "missing")
    (tmp_path / "missing" / fit_id / "metrics_mappings.tsv").unlink()
    (message,) = check_declaration(missing)
    assert "adopt the fit again" in message

    twice, _ = _copied(credibility_project, tmp_path / "twice")
    path = tmp_path / "twice" / fit_id / "metrics_mappings.tsv"
    table = pd.read_csv(path, sep="\t")
    pd.concat([table, table.assign(kind="training")]).to_csv(
        path, sep="\t", index=False
    )
    (message,) = check_declaration(twice)
    assert "a mapping has one kind" in message

    outside = credibility_project.replace(credibility=".credibility:OUTSIDE")
    # an invalid declaration selects nothing, the kinds are checked
    first, second = check_declaration(outside)
    assert "the domain field 'label'" in first
    assert "is not a member of Route" in second


def test_the_extensions_check_the_decisive_points(fits: FitTable) -> None:
    """A study of a ratio with two denominators is a problem of the extension."""
    healthy = metadata_filter(ToyMetaData, health={Health.HEALTHY})
    renal = metadata_filter(ToyMetaData, health={Health.RENAL_IMPAIRMENT})
    arms = Arms(observable="[A]", numerator=renal, denominator=healthy)
    ratio = QuantityOfInterest("A_ratio", "renal over healthy", ratio=arms)
    credibility = _credibility(
        Evidence(
            "renal and healthy",
            EvidenceRole.VALIDATION,
            named_filter({"CredA": {"fm_renal", "fm_dose", "fm_subject"}}),
        ),
        quantities=(*_A, ratio),
    )
    # without the extension the pairing restricts the evidence to its observable
    (evidence,) = select(credibility, fits).evidence
    assert _pairs(fits, evidence.rows) == {
        ("CredA", "fm_renal"),
        ("CredA", "fm_dose"),
        ("CredA", "fm_subject"),
    }
    with pytest.raises(SelectionError) as err:
        select(credibility, fits, [ARMS])
    (problem,) = err.value.problems
    assert problem == (
        "context of use 'dose': the ratio 'A_ratio' has the denominators "
        "['fm_dose', 'fm_subject'] in the study CredA"
    )
    one = _credibility(
        Evidence(
            "renal and dose",
            EvidenceRole.VALIDATION,
            named_filter({"CredA": {"fm_renal", "fm_dose"}}),
        ),
        quantities=(*_A, ratio),
    )
    assert len(select(one, fits, [ARMS]).evidence) == 1


def test_a_ratio_of_an_unknown_observable(fits: FitTable) -> None:
    arms = Arms(
        observable="B", numerator=named_filter(DOSE), denominator=named_filter(DOSE)
    )
    message = _problems(
        _credibility(
            Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE)),
            quantities=(*_A, QuantityOfInterest("B_ratio", "ratio", ratio=arms)),
        ),
        fits,
    )
    assert "the ratio 'B_ratio' matches no observable of the fits with 'B'" in message


def test_check_declaration_with_the_extensions(
    credibility_project: Project,
) -> None:
    project = credibility_project.replace(
        credibility_settings=CredibilitySettings(
            metadata_type=ToyMetaData, extensions=(ARMS,)
        )
    )
    assert check_declaration(project) == []


def test_check_declaration_reports_the_extensions(credibility_project: Project) -> None:
    project = credibility_project.replace(
        credibility=".credibility:WITH_ARMS",
        credibility_settings=CredibilitySettings(
            metadata_type=ToyMetaData, extensions=(ARMS,)
        ),
    )
    assert check_declaration(project) == []
    (problem,) = check_declaration(
        project.replace(credibility=".credibility:WITH_TYPO")
    )
    assert "the setting 'armz' names no evidence extension" in problem
    unclaimed = project.replace(
        credibility_settings=CredibilitySettings(metadata_type=ToyMetaData)
    )
    problems = check_declaration(unclaimed)
    assert any("is claimed by no evidence extension" in p for p in problems)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (CredibilitySettings(metadata_type=".fitting:DOSE"), "is no class"),
        (CredibilitySettings(domain_fields=("dose",)), "are no fields"),
        (CredibilitySettings(extensions=(".metadata:Route",)), "are no sbmlsim"),
    ],
)
def test_check_declaration_reports_the_settings(
    credibility_project: Project, settings: CredibilitySettings, message: str
) -> None:
    """Invalid credibility settings are a problem, the fits are checked still."""
    project = credibility_project.replace(credibility_settings=settings)
    (problem,) = check_declaration(project)
    assert message in problem
    with pytest.raises(ValueError, match="no credibility declaration"):
        check_declaration(project.replace(credibility=None))


def test_in_domain(fits: FitTable) -> None:
    table = fits.table
    renal = CREDIBILITY.context("renal")
    inside = in_domain(renal, table)
    assert set(inside["health"]) == {"HEALTHY", "RENAL_IMPAIRMENT"}
    dose = in_domain(CREDIBILITY.context("dose"), table)
    assert set(zip(dose["route"], dose["health"], strict=True)) == {("IV", "HEALTHY")}
    assert len(in_domain(CREDIBILITY.context("describe"), table)) == len(table)
    assert set(metadata_values(table, "route")) == {"IV", "PO"}
    assert set(metadata_values(table, "tissue")) == {""}
