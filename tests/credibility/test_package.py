"""The toy credibility package, its adopted fit and the factors of V&V 40."""

import pandas as pd

from sbmlsim.credibility import FACTORS, CouLevel, FactorStatus
from sbmlsim.project import Project
from tests.credibility.tables import KINDS


def test_load_credibility(credibility_project: Project) -> None:
    assert credibility_project.load_credibility().context("dose").level is CouLevel.C2


def test_the_fit_has_every_kind(credibility_project: Project) -> None:
    (fit_id,) = credibility_project.adopted_fits().values()
    record = credibility_project.adopted_record(fit_id)
    table = pd.read_csv(record / "metrics_mappings.tsv", sep="\t")
    kinds = {
        (e, m): k
        for e, m, k in zip(table.experiment, table.mapping, table.kind, strict=True)
    }
    assert kinds == KINDS


def test_the_factors() -> None:
    ids = [f.id for f in FACTORS]
    assert len(ids) == len(set(ids)) == 13
    assert all("pkpdsim" not in f.statement for f in FACTORS)
    for factor in FACTORS:
        record = factor.to_dict()
        assert record["status"] == factor.status.value
        assert (record["step"] is None) is (
            factor.status is not FactorStatus.NOT_ASSESSED
        )
