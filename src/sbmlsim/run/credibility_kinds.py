"""The kind of the credibility of the contexts of use of a project.

`credibility` (`CredibilitySpec`) assesses the contexts of use of a project
(`sbmlsim.project.Project.credibility`) against its adopted fits
(`sbmlsim.credibility.assess`) and writes the record, the tables and the
report into `out/credibility/` (`CREDIBILITY_DIR`,
`sbmlsim.credibility.report.write_outputs`), with the evidence extensions and
the battery hook of the project. The fields are the options of an
assessment: `verification` (the decisive predictions with tightened
tolerances and a second integration method) and `battery` (the checks of the
variants of the model), `uncertainty` (the propagation of the parameter
uncertainty of the adopted fits, `draws` and `seed`), `variability` (the
predictive check with the declared variability of the subjects, `subjects`
and `studies`), `lock` (the lock of the evidence is written into the out,
`out/credibility.lock.json`, instead of comparing the evidence with the lock
of the project; the package is never written) and `source_commit` (the commit
of the source repository of a release, recorded in the provenance). Two
inputs: `cross_validation`, an out whose `fits/<fit id>/cross_validation.json`
(`sbmlsim.credibility.write_cross_validation`) are the cross-validations of
the adopted fits instead of the ones of the records of the project, and
`record`, an earlier record (a `credibility.json` or an out with
`credibility/credibility.json`), whose outputs are written again without an
assessment, so a report of a record of pkpdsim 0.17 to 0.19 is written with
the current template.

The analyses are imported when the kind plans or runs, not when this module is
imported.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import Field

from sbmlsim.run.kinds import ConfigurationError, Kind, Spec, register

if TYPE_CHECKING:
    from sbmlsim.credibility.assessment import CredibilityAssessment
    from sbmlsim.credibility.cross_validation import CrossValidation
    from sbmlsim.credibility.uncertainty import PropagationSettings
    from sbmlsim.credibility.variability import VariabilitySettings
    from sbmlsim.project import Project
    from sbmlsim.run.context import RunContext

__all__ = ["CREDIBILITY", "CREDIBILITY_DIR", "CredibilitySpec"]

logger = logging.getLogger(__name__)

#: the directory of the outputs of an assessment in the out
CREDIBILITY_DIR: Final = "credibility"


class CredibilitySpec(Spec):
    """The spec of an assessment of the credibility of a project, see the module.

    Attributes:
        project: the project, `module:attribute`.
        verification: verify the decisive predictions.
        battery: run the battery of the checks of the variants with the
            verification.
        lock: write the lock of the evidence into the out instead of comparing
            the evidence with the lock of the project.
        uncertainty: propagate the parameter uncertainty of the adopted fits.
        draws: the draws of every fit of the uncertainty, its default without.
        seed: the seed of the draws of the uncertainty, its default without.
        variability: check the decisive points with the declared variability
            of the subjects.
        subjects: the subjects of every study of the variability.
        studies: the studies of every parameter vector of the variability.
        source_commit: the commit of the source repository of the model,
            recorded in the provenance instead of the commit of the root.
        cross_validation: the input of the cross-validations, an out with
            `fits/<fit id>/cross_validation.json`.
        record: the input of an earlier record whose outputs are written again.
    """

    project: str
    verification: bool = True
    battery: bool = True
    lock: bool = False
    uncertainty: bool = False
    draws: int | None = Field(default=None, ge=2)
    seed: int | None = None
    variability: bool = False
    subjects: int | None = Field(default=None, ge=1)
    studies: int | None = Field(default=None, ge=1)
    source_commit: str | None = None
    cross_validation: Path | None = None
    record: Path | None = None


def _project(spec: CredibilitySpec) -> Project:
    """Load the project of a spec and check that it has a declaration.

    Raises:
        ConfigurationError: for a project which is not loaded or has no
            declaration.
    """
    from sbmlsim.project import load_project

    try:
        project = load_project(spec.project)
    except (ImportError, ValueError) as err:
        raise ConfigurationError(
            f"The project '{spec.project}' is not loaded: {err}"
        ) from err
    if project.credibility is None:
        raise ConfigurationError(
            f"The project '{spec.project}' has no credibility declaration."
        )
    return project


def _propagation(spec: CredibilitySpec) -> PropagationSettings | None:
    """Get the settings of the uncertainty of a spec.

    Raises:
        ConfigurationError: for invalid settings or settings without the
            uncertainty.
    """
    from sbmlsim.credibility.uncertainty import PropagationSettings

    if not spec.uncertainty:
        if spec.draws is not None or spec.seed is not None:
            raise ConfigurationError("draws and seed are settings of the uncertainty.")
        return None
    defaults = PropagationSettings()
    try:
        return PropagationSettings(
            draws=defaults.draws if spec.draws is None else spec.draws,
            seed=defaults.seed if spec.seed is None else spec.seed,
        )
    except ValueError as err:
        raise ConfigurationError(str(err)) from err


def _variability(spec: CredibilitySpec) -> VariabilitySettings | None:
    """Get the settings of the variability of a spec.

    Raises:
        ConfigurationError: for invalid settings or settings without the
            variability.
    """
    from sbmlsim.credibility.variability import VariabilitySettings

    if not spec.variability:
        if spec.subjects is not None or spec.studies is not None:
            raise ConfigurationError(
                "subjects and studies are settings of the variability."
            )
        return None
    defaults = VariabilitySettings()
    try:
        return VariabilitySettings(
            subjects=defaults.subjects if spec.subjects is None else spec.subjects,
            studies=defaults.studies if spec.studies is None else spec.studies,
        )
    except ValueError as err:
        raise ConfigurationError(str(err)) from err


def _check_record_options(spec: CredibilitySpec) -> None:
    """Refuse the options of an assessment for the outputs of a record.

    Raises:
        ConfigurationError: for an option of an assessment with a record.
    """
    if spec.record is None:
        return
    given = [
        name
        for name, value in (
            ("lock", spec.lock),
            ("uncertainty", spec.uncertainty),
            ("variability", spec.variability),
            ("source_commit", spec.source_commit is not None),
            ("cross_validation", spec.cross_validation is not None),
        )
        if value
    ]
    if given:
        raise ConfigurationError(
            f"{', '.join(given)} {'is an option' if len(given) == 1 else 'are options'} "
            "of an assessment, not of the outputs of a record."
        )


def _plan(spec: CredibilitySpec) -> CredibilitySpec:
    """Check the project and the settings, the spec is planned as given."""
    _check_record_options(spec)
    _project(spec)
    _propagation(spec)
    _variability(spec)
    return spec


def _record_file(record: Path) -> Path:
    """Get the record of the input `record`, a file or an out with one.

    Raises:
        ConfigurationError: for an input without a record.
    """
    from sbmlsim.credibility.report import RECORD

    for path in (record, record / RECORD, record / CREDIBILITY_DIR / RECORD):
        if path.is_file():
            return path
    raise ConfigurationError(
        f"The input record '{record}' is neither a {RECORD} nor an out with "
        f"{CREDIBILITY_DIR}/{RECORD}."
    )


def _read_record(record: Path) -> CredibilityAssessment:
    """Read the record of the input `record`.

    Raises:
        ConfigurationError: for a file which is no record of an assessment.
    """
    from sbmlsim.credibility.assessment import CredibilityAssessment

    path = _record_file(record)
    try:
        return CredibilityAssessment.from_json(path)
    except (KeyError, TypeError, AttributeError, ValueError) as err:
        raise ConfigurationError(
            f"The record '{path}' is no assessment of the contexts of use ({err!r})."
        ) from err


def _cross_validations(
    project: Project, directory: Path | None
) -> dict[str, CrossValidation] | None:
    """Read the cross-validations of the adopted fits from the input, `None` without.

    Raises:
        ConfigurationError: for an input which is no directory.
    """
    from sbmlsim.credibility.cross_validation import (
        CROSS_VALIDATION_FILE,
        CrossValidation,
    )
    from sbmlsim.run.fit_kinds import FITS_DIR

    if directory is None:
        return None
    if not directory.is_dir():
        raise ConfigurationError(
            f"The input cross_validation '{directory}' is no directory."
        )
    found: dict[str, CrossValidation] = {}
    for fit_id in project.adopted_fits().values():
        path = directory / FITS_DIR / fit_id / CROSS_VALIDATION_FILE
        if path.is_file():
            result = CrossValidation.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            found[result.subset] = result
    return found


def _summary(
    assessment: CredibilityAssessment, out: Path, directory: Path
) -> dict[str, Any]:
    """Get the summary of an assessment."""
    return {
        "directory": directory.relative_to(out).as_posix(),
        "verdicts": {c.id: c.verdict.value for c in assessment.contexts},
        "verified": sum(c.verified is True for c in assessment.contexts),
        "lock": assessment.provenance.lock,
    }


def _run(spec: CredibilitySpec, ctx: RunContext) -> dict[str, Any]:
    """Assess the project, or read the record, and write the outputs into the out."""
    from sbmlsim.credibility.assessment import assess
    from sbmlsim.credibility.report import print_summary, write_outputs

    _check_record_options(spec)
    project = _project(spec)
    try:
        extensions = project.evidence_extensions()
        hook = project.battery_hook()
    except ValueError as err:
        raise ConfigurationError(str(err)) from err
    if spec.record is not None:
        assessment = _read_record(spec.record)
    else:
        assessment = assess(
            project,
            spec.source_commit,
            verification=spec.verification,
            battery=spec.battery,
            lock=ctx.out if spec.lock else None,
            uncertainty=_propagation(spec),
            variability=_variability(spec),
            cross_validations=_cross_validations(project, spec.cross_validation),
        )
    directory = write_outputs(assessment, ctx.out / CREDIBILITY_DIR, extensions, hook)
    print_summary(assessment)
    return _summary(assessment, ctx.out, directory)


#: the assessment of the credibility of a project
CREDIBILITY: Kind[CredibilitySpec] = Kind(
    name="credibility",
    spec=CredibilitySpec,
    run=_run,
    plan=_plan,
    inputs=("cross_validation", "record"),
    uses_cores=False,
)

register(CREDIBILITY)
