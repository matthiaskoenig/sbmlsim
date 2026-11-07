"""The leave-one-study-out cross-validation (`sbmlsim.credibility.cross_validation`).

The adopted fit of the toy package fits the decay rate to the training
mappings of the studies `CredA` and `CredB`; the tests write into an out and
read from a copy of the folder of fits, never from the one of the session.
"""

import json
import math
import shutil
from pathlib import Path

import pytest

from sbmlsim.credibility.cross_validation import (
    CROSS_VALIDATION_FILE,
    CrossValidation,
    HeldOutStudy,
    cross_validate,
    held_out,
    read_cross_validations,
    training_studies,
    write_cross_validation,
)
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.objects import MappingKind
from sbmlsim.project import Project
from sbmlsim.run.fit_kinds import FITS_DIR

DATA = Path(__file__).parents[1] / "data" / "credibility"


@pytest.fixture
def project(credibility_project: Project, tmp_path: Path) -> Project:
    """Get the toy project with a copy of its folder of adopted fits."""
    assert credibility_project.fits_dir is not None
    fits = tmp_path / "fits"
    shutil.copytree(credibility_project.fits_dir, fits)
    return credibility_project.replace(fits_dir=fits)


def _fit_id(project: Project) -> str:
    (fit_id,) = project.adopted_fits().values()
    return fit_id


def _definition(project: Project) -> FitDefinition:
    return project.load_fit_definitions()["DECAY"]


def test_the_training_studies(project: Project) -> None:
    assert training_studies(_definition(project)) == ["CredA", "CredB"]


def test_held_out(project: Project) -> None:
    definition = _definition(project)
    held = held_out(definition, "CredA")
    kinds = {c.kind for c in held.mapping_collections()["CredA"]}
    assert MappingKind.TRAINING not in kinds
    assert training_studies(held) == ["CredB"]
    # the definition of the project keeps its kinds
    assert training_studies(definition) == ["CredA", "CredB"]
    with pytest.raises(KeyError, match="'Nothing' is no study"):
        held_out(definition, "Nothing").mapping_collections()


def test_cross_validate(project: Project, tmp_path: Path) -> None:
    fit_id = _fit_id(project)
    result = cross_validate(project, fit_id, output_dir=tmp_path / "cv")
    assert result.fit == fit_id
    assert result.subset == "DECAY"
    assert [s.study for s in result.studies] == ["CredA", "CredB"]
    for study in result.studies:
        assert study.n_points > 0
        assert math.isfinite(study.gmfe_held_out)
        assert study.gmfe_held_out >= 1.0
        assert math.isfinite(study.cost)
    assert result.n_points == sum(s.n_points for s in result.studies)
    assert result.statement().startswith(
        f"Leave-one-study-out cross-validation of the fit {result.subset} (2 studies, "
    )
    # written into the out in the layout of the adopted fits, not into the fits
    out = tmp_path / "out"
    path = write_cross_validation(result, out)
    assert path == out / FITS_DIR / fit_id / CROSS_VALIDATION_FILE
    assert read_cross_validations(project) == {}
    shutil.copytree(out / FITS_DIR, tmp_path / "fits", dirs_exist_ok=True)
    read = read_cross_validations(project)["DECAY"]
    assert read.to_dict() == result.to_dict()


def test_a_study_of_the_studies(project: Project, tmp_path: Path) -> None:
    result = cross_validate(project, _fit_id(project), ["CredB"], output_dir=tmp_path)
    assert [s.study for s in result.studies] == ["CredB"]


def test_the_studies_must_be_training_studies(project: Project, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no training studies"):
        cross_validate(project, _fit_id(project), ["CredC"], output_dir=tmp_path)
    with pytest.raises(ValueError, match="is not adopted"):
        cross_validate(project, "DECAY_20260101_000000__00000", output_dir=tmp_path)


def test_a_round_trip_without_points() -> None:
    result = CrossValidation(
        fit="PK_20260101_000000__00000",
        subset="PK",
        studies=(
            HeldOutStudy(
                study="A",
                n_points=0,
                gmfe_held_out=math.nan,
                gmfe_fitted=math.nan,
                within_2fold_held_out=math.nan,
                within_2fold_fitted=math.nan,
                cost=1.0,
            ),
        ),
        gmfe_held_out=math.nan,
        gmfe_fitted=math.nan,
        seed=1,
        date="2026-11-04T12:00:00+01:00",
    )
    read = CrossValidation.from_dict(result.to_dict())
    assert read.to_dict() == result.to_dict()
    assert read.to_dict()["gmfe_held_out"] is None


def test_the_cross_validation_of_pkpdsim() -> None:
    """A cross-validation which pkpdsim 0.19 wrote is read as it is."""
    data = json.loads((DATA / "cross_validation-0.19.json").read_text(encoding="utf-8"))
    result = CrossValidation.from_dict(data)
    assert result.subset == "DECAY"
    assert [s.study for s in result.studies] == ["CredA", "CredB"]
    assert result.n_points == 35
    assert result.to_dict() == data
