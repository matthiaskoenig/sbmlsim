"""The leave-one-study-out cross-validation of an adopted fit.

A fit describes its training data; how well it predicts a study it has not
seen is a question of the form of the model. The cross-validation refits the
problem of an adopted fit once per training study without that study (its
training mappings become validation mappings, `held_out`), starting from the
adopted parameters, and predicts the held-out study with the refitted
parameters. Compared with the predictions of the adopted fit for the same
points, the held-out predictions show how much of the description of a study
is fitted to it: a GMFE of the held-out points close to the one of the fitted
points is a model whose form carries across studies, a much larger one a model
whose parameters are fitted to every study.

`cross_validate` runs it, `CrossValidation` is its result, which
`write_cross_validation` writes into an out in the layout of the records of the
adopted fits (`fits/<fit id>/cross_validation.json`, `sbmlsim.run.fit_kinds.FITS_DIR`,
never into the package),
from where it is reviewed and committed next to the record of the adopted fit;
the credibility assessment reads it there for the factor model form
(`read_cross_validations`). It is run on demand, since it fits the problem
once per training study. The format is the one of pkpdsim 0.19.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Self

import numpy as np
import pandas as pd

from sbmlsim.atomic import write_text_atomic
from sbmlsim.run.fit_kinds import FITS_DIR

if TYPE_CHECKING:
    from pathlib import Path

    from sbmlsim.fit.cli import FitDefinition
    from sbmlsim.fit.objects import FitMappingCollection
    from sbmlsim.project import Project

__all__ = [
    "CROSS_VALIDATION_FILE",
    "CrossValidation",
    "HeldOutStudy",
    "cross_validate",
    "held_out",
    "read_cross_validations",
    "training_studies",
    "write_cross_validation",
]

logger = logging.getLogger(__name__)

#: the file of a cross-validation in the record of its adopted fit
CROSS_VALIDATION_FILE: Final = "cross_validation.json"


def _gmfe(dv: np.ndarray, ipred: np.ndarray) -> float:
    """Get the GMFE of the points with a positive finite data value and prediction."""
    usable = np.isfinite(dv) & np.isfinite(ipred) & (dv > 0) & (ipred > 0)
    if not usable.any():
        return math.nan
    return float(10 ** np.mean(np.abs(np.log10(ipred[usable] / dv[usable]))))


def _within(dv: np.ndarray, ipred: np.ndarray, fold: float = 2.0) -> float:
    """Get the percentage of the usable points within a fold."""
    usable = np.isfinite(dv) & np.isfinite(ipred) & (dv > 0) & (ipred > 0)
    if not usable.any():
        return math.nan
    ratio = ipred[usable] / dv[usable]
    return float(100.0 * np.mean((ratio <= fold) & (ratio >= 1 / fold)))


def _write(value: float) -> float | None:
    """Get a float as plain data, `NaN` as `None`."""
    return None if math.isnan(value) else value


def _read(value: float | None) -> float:
    """Get a float from plain data, `None` as `NaN`."""
    return math.nan if value is None else float(value)


@dataclass(frozen=True, kw_only=True)
class HeldOutStudy:
    """The prediction of a training study by the fit without it.

    Attributes:
        study: the study, the id of its experiment.
        n_points: the points of its training mappings with a positive data
            value and positive predictions.
        gmfe_held_out: the GMFE of the predictions of the fit without the study.
        gmfe_fitted: the GMFE of the predictions of the adopted fit.
        within_2fold_held_out: the percentage of the held-out predictions within
            2-fold.
        within_2fold_fitted: the same of the adopted fit.
        cost: the cost of the fit without the study.
    """

    study: str
    n_points: int
    gmfe_held_out: float
    gmfe_fitted: float
    within_2fold_held_out: float
    within_2fold_fitted: float
    cost: float

    def to_dict(self) -> dict[str, object]:
        """Get the study as plain data, `NaN` as `None`."""
        return {
            "study": self.study,
            "n_points": self.n_points,
            "gmfe_held_out": _write(self.gmfe_held_out),
            "gmfe_fitted": _write(self.gmfe_fitted),
            "within_2fold_held_out": _write(self.within_2fold_held_out),
            "within_2fold_fitted": _write(self.within_2fold_fitted),
            "cost": _write(self.cost),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the study from plain data."""
        return cls(
            study=data["study"],
            n_points=int(data["n_points"]),
            gmfe_held_out=_read(data["gmfe_held_out"]),
            gmfe_fitted=_read(data["gmfe_fitted"]),
            within_2fold_held_out=_read(data["within_2fold_held_out"]),
            within_2fold_fitted=_read(data["within_2fold_fitted"]),
            cost=_read(data["cost"]),
        )


@dataclass(frozen=True, kw_only=True)
class CrossValidation:
    """The leave-one-study-out cross-validation of an adopted fit.

    Attributes:
        fit: the id of the adopted fit.
        subset: its fit problem, e.g. `PK`.
        studies: the result of every held-out study.
        gmfe_held_out: the GMFE of all held-out points together.
        gmfe_fitted: the GMFE of the same points with the adopted fit.
        seed: the seed of the fits.
        date: the date of the cross-validation, ISO 8601 with the time zone.
    """

    fit: str
    subset: str
    studies: tuple[HeldOutStudy, ...]
    gmfe_held_out: float
    gmfe_fitted: float
    seed: int
    date: str

    @property
    def n_points(self) -> int:
        """Get the held-out points of all studies."""
        return sum(s.n_points for s in self.studies)

    def statement(self) -> str:
        """Describe the cross-validation in one sentence."""
        studies = len(self.studies)
        return (
            f"Leave-one-study-out cross-validation of the fit {self.subset} "
            f"({studies} {'study' if studies == 1 else 'studies'}, "
            f"{self.n_points} points): GMFE {self.gmfe_held_out:.3g} of the "
            f"held-out predictions, {self.gmfe_fitted:.3g} of the fitted ones."
        )

    def to_dict(self) -> dict[str, object]:
        """Get the cross-validation as plain data."""
        return {
            "fit": self.fit,
            "subset": self.subset,
            "studies": [s.to_dict() for s in self.studies],
            "gmfe_held_out": _write(self.gmfe_held_out),
            "gmfe_fitted": _write(self.gmfe_fitted),
            "seed": self.seed,
            "date": self.date,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Get the cross-validation from plain data."""
        return cls(
            fit=data["fit"],
            subset=data["subset"],
            studies=tuple(HeldOutStudy.from_dict(s) for s in data["studies"]),
            gmfe_held_out=_read(data["gmfe_held_out"]),
            gmfe_fitted=_read(data["gmfe_fitted"]),
            seed=int(data["seed"]),
            date=data["date"],
        )


def training_studies(definition: FitDefinition) -> list[str]:
    """Get the studies of a fit problem with a training mapping, sorted."""
    from sbmlsim.fit.objects import MappingKind

    return sorted(
        study
        for study, collections in definition.mapping_collections().items()
        if any(c.kind is MappingKind.TRAINING and not c.exclude for c in collections)
    )


def held_out(definition: FitDefinition, study: str) -> FitDefinition:
    """Get a fit problem whose training mappings of a study are validation mappings.

    Args:
        definition: the fit problem.
        study: the study to hold out.

    Returns:
        The problem without the study in its training data; the study is
        still simulated, as validation data, so its predictions are reported.

    Raises:
        KeyError: if the study is no study of the problem.
    """
    from sbmlsim.fit.objects import MappingKind

    original: Callable[[], dict[str, list[FitMappingCollection]]] = (
        definition.mapping_collections
    )

    def collections() -> dict[str, list[FitMappingCollection]]:
        by_study = original()
        if study not in by_study:
            raise KeyError(
                f"The study '{study}' is no study of the problem, the studies "
                f"are {sorted(by_study)}."
            )
        changed = []
        for collection in by_study[study]:
            if collection.kind is MappingKind.TRAINING:
                collection = copy.copy(collection)
                collection.kind = MappingKind.VALIDATION
            changed.append(collection)
        return {**by_study, study: changed}

    return dataclasses.replace(definition, mapping_collections=collections)


def _study_points(points: pd.DataFrame, study: str) -> pd.DataFrame:
    """Get the points of the mappings of a study, sorted by mapping and time."""
    rows = points[points["experiment"].astype(str) == study]
    columns = [c for c in ("mapping", "x") if c in rows.columns]
    return rows.sort_values(columns, kind="stable").reset_index(drop=True)


def cross_validate(
    project: Project,
    fit_id: str,
    studies: Sequence[str] | None = None,
    *,
    output_dir: Path,
    seed: int = 1234,
    n_cores: int = 1,
) -> CrossValidation:
    """Cross-validate an adopted fit by leaving out one training study at a time.

    Every refit is a run of least squares from the adopted parameters (the
    sampling `START`) with the adopted fits of the other problems fixed, as a
    fit with a `start` does (`sbmlsim.fit.project_fits`).

    Args:
        project: the project.
        fit_id: an adopted fit of the project.
        studies: the training studies to hold out, all by default.
        output_dir: the directory of the refits.
        seed: the seed of the refits.
        n_cores: the workers of a refit.

    Returns:
        The cross-validation.

    Raises:
        ValueError: if the fit is not adopted or a study is no training study.
    """
    from sbmlsim.fit.cli import report_from_records, run_fit
    from sbmlsim.fit.options import OptimizationAlgorithmType
    from sbmlsim.fit.project_fits import fixed_values, start_point
    from sbmlsim.fit.project_fits import with_start as start_from
    from sbmlsim.fit.report import FitReport
    from sbmlsim.fit.result import fit_subset
    from sbmlsim.fit.sampling import SamplingType

    adopted_fit_ids = list(project.adopted_fits().values())
    if fit_id not in adopted_fit_ids:
        raise ValueError(
            f"The fit '{fit_id}' is not adopted by '{project.name}', the adopted "
            f"fits are {adopted_fit_ids}."
        )
    subset = fit_subset(fit_id)
    definition = project.load_fit_definitions()[subset]
    record = project.adopted_record(fit_id)
    trained = training_studies(definition)
    chosen = list(studies) if studies else trained
    unknown = [s for s in chosen if s not in trained]
    if unknown:
        raise ValueError(
            f"The studies {unknown} are no training studies of the fit "
            f"'{fit_id}', its training studies are {trained}."
        )
    point = start_point(project, definition, fit_id, output_dir)
    _, fixed = fixed_values(project, subset, definition)
    reference = report_from_records(
        definition, record, opid=subset, mapping_figures=False
    )
    fitted_points = reference.metrics(reference.reference_set).datapoints_df()
    results: list[HeldOutStudy] = []
    held_dv: list[np.ndarray] = []
    held_ipred: list[np.ndarray] = []
    fitted_ipred: list[np.ndarray] = []
    for study in chosen:
        logger.info("Fit '%s' without the study '%s'", subset, study)
        runs = run_fit(
            definition=start_from(held_out(definition, study), point),
            opid=f"{subset}_without_{study}",
            algorithm=OptimizationAlgorithmType.LEAST_SQUARE,
            size=1,
            n_cores=n_cores,
            seed=seed,
            output_dir=output_dir,
            fixed_values=fixed,
            sampling=SamplingType.START,
        )
        run = next(iter(runs.values()))
        report = FitReport.from_optimization_result(
            problem=run.problem, opt_result=run.result, mapping_figures=False
        )
        held = _study_points(
            report.metrics(report.reference_set).datapoints_df(), study
        )
        fitted = _study_points(fitted_points, study)
        if len(held) != len(fitted):
            raise ValueError(
                f"The held-out study '{study}' has {len(held)} points, the "
                f"adopted fit {len(fitted)}."
            )
        dv = held["DV"].to_numpy(dtype=float)
        ipred = held["IPRED"].to_numpy(dtype=float)
        adopted = fitted["IPRED"].to_numpy(dtype=float)
        usable = np.isfinite(dv) & np.isfinite(ipred) & (dv > 0) & (ipred > 0)
        usable &= np.isfinite(adopted) & (adopted > 0)
        held_dv.append(dv[usable])
        held_ipred.append(ipred[usable])
        fitted_ipred.append(adopted[usable])
        results.append(
            HeldOutStudy(
                study=study,
                n_points=int(usable.sum()),
                gmfe_held_out=_gmfe(dv[usable], ipred[usable]),
                gmfe_fitted=_gmfe(dv[usable], adopted[usable]),
                within_2fold_held_out=_within(dv[usable], ipred[usable]),
                within_2fold_fitted=_within(dv[usable], adopted[usable]),
                cost=float(run.result.df_fits["cost"].min()),
            )
        )
    all_dv = np.concatenate(held_dv) if held_dv else np.array([])
    return CrossValidation(
        fit=fit_id,
        subset=subset,
        studies=tuple(results),
        gmfe_held_out=_gmfe(
            all_dv, np.concatenate(held_ipred) if held_ipred else all_dv
        ),
        gmfe_fitted=_gmfe(
            all_dv, np.concatenate(fitted_ipred) if fitted_ipred else all_dv
        ),
        seed=seed,
        date=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def write_cross_validation(result: CrossValidation, out: Path) -> Path:
    """Write a cross-validation into an out, in the layout of the adopted fits.

    Args:
        result: the cross-validation.
        out: the directory it is written into; never the package, the file
            is committed next to the record of its adopted fit after a review.

    Returns:
        The file, `fits/<fit id>/cross_validation.json` of the out.
    """
    path = out / FITS_DIR / result.fit / CROSS_VALIDATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(result.to_dict(), indent=2) + "\n")
    return path


def read_cross_validations(project: Project) -> dict[str, CrossValidation]:
    """Read the cross-validations of the adopted fits of a project, by subset.

    A cross-validation lies in the record of its adopted fit
    (`Project.adopted_record`).

    Returns:
        The cross-validation of every adopted fit which has one.
    """
    found: dict[str, CrossValidation] = {}
    for fit_id in project.adopted_fits().values():
        path = project.adopted_record(fit_id) / CROSS_VALIDATION_FILE
        if path.is_file():
            result = CrossValidation.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            found[result.subset] = result
    return found
