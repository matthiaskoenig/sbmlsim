"""The build of a release repository.

The build runs the analyses of the model package inside the release repository
(`uv run`, in the environment of its locked, released dependencies) through the
run layer of sbmlsim, `python -m sbmlsim.run <kind> specs/<step>.json --out
results/<step>`, and copies the generated models, figures, simulation reports,
fit reports and the assessment of the credibility to their places in the
repository. The specs of the steps (`BuildStep`, `build_steps`) are files of
the release, so a user of the release runs the same analyses with the same
commands. The exported package is never imported: the project of the source
package only gives the paths of the model relative to its package folder, its
analyses and the adopted fits.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from sbmlsim.release.source import ReleaseError, Runner, package_dirs, run, stage
from sbmlsim.run.context import (
    PROGRESS_DIR,
    PROGRESS_FILE,
    RUN_FILE,
    SPEC_FILE,
    SUMMARY_FILE,
)

if TYPE_CHECKING:
    from sbmlsim.project import Project

__all__ = [
    "RUN_RECORDS",
    "SPECS_DIR",
    "BuildPaths",
    "BuildResult",
    "BuildStep",
    "build",
    "build_steps",
    "has_distributions",
    "skipped_steps",
]

logger = logging.getLogger(__name__)

_FACTORY_PATTERNS = ("*.xml", "*.md", "*.ant")
_LICENSES = ("mit.txt", "cc-by-sa-4.0.txt")

#: the folder of the specs of the steps in the release repository
SPECS_DIR = "specs"


class BuildPaths:
    """The folders of the release repository, relative to its root."""

    MODELS = "models"
    FIGURES = "figures"
    SIMULATIONS = "simulations"
    UNCERTAINTY = "simulations_uncertainty"
    FITS = "fit"
    CREDIBILITY = "credibility"
    RESULTS = "results"


#: the records of a run of the run layer of sbmlsim in the folder of a result
#: (the host, local paths and the git state), which a release leaves out
RUN_RECORDS = frozenset(
    {RUN_FILE, SPEC_FILE, SUMMARY_FILE, PROGRESS_FILE, PROGRESS_DIR}
)


@dataclass(frozen=True)
class BuildStep:
    """A step of the build: an analysis of the run layer of sbmlsim.

    Attributes:
        name: the name of the step, the name of its spec file
            `specs/<name>.json` and of its output `results/<name>`.
        kind: the kind of the run layer, e.g. `simulate`.
        spec: the settings of the spec.
        title: what the step does, the line of its stage.
    """

    name: str
    kind: str
    spec: dict[str, Any]
    title: str

    @property
    def spec_file(self) -> PurePosixPath:
        """The spec file of the step, relative to the release repository."""
        return PurePosixPath(SPECS_DIR, f"{self.name}.json")

    @property
    def out(self) -> PurePosixPath:
        """The output of the step, relative to the release repository."""
        return PurePosixPath(BuildPaths.RESULTS, self.name)

    def command(self, workers: int | None = None) -> list[str]:
        """Get the command of the step in the release repository.

        Args:
            workers: the number of workers of a step which simulates, the
                default of the run layer without.

        Returns:
            The command, with POSIX paths relative to the release repository.
        """
        command = ["uv", "run", "python", "-m", "sbmlsim.run", self.kind]
        command += [self.spec_file.as_posix(), "--out", self.out.as_posix()]
        if workers is not None and self.kind == "simulate":
            command += ["--workers", str(workers)]
        return command


def build_steps(
    project: Project,
    reference: str,
    source_commit: str | None = None,
    *,
    root: Path | None = None,
) -> list[BuildStep]:
    """Get the steps of the build of a release of a project, in their order.

    - `factory`: the models, with the factory of the project, which the
      build runs first and copies over the exported package; a project
      without one releases the models of its package.
    - `simulate`: the experiments of a project with experiments.
    - `uncertainty`: the experiments over the draws of the fitted parameters,
      with the defaults of `sbmlsim.experiment.draws.UncertaintySettings`, for
      a project with experiments and adopted fits whose default variant has
      the distributions of its fitted parameters (`has_distributions`, else
      the step is skipped, see `skipped_steps`).
    - `report`: the reports of the adopted fits, one folder per fit id.
    - `credibility`: the assessment of the contexts of use of a project with a
      declaration, with the commit of the source repository in its provenance.

    Args:
        project: the project.
        reference: the project as `module:attribute`, which the release
            repository imports.
        source_commit: the commit of the source repository.
        root: the package whose default variant `has_distributions` reads,
            e.g. the exported package with the models of the factory; the
            root of the project without.

    Returns:
        The steps.

    Raises:
        ReleaseError: if the adopted fits are not read.
    """
    try:
        adopted = bool(project.adopted_fits())
    except ValueError as err:
        raise ReleaseError(str(err)) from err
    base = {"project": reference}
    steps: list[BuildStep] = []
    if project.factory is not None:
        steps.append(BuildStep("factory", "factory", dict(base), "Models"))
    if project.experiments is not None:
        steps.append(BuildStep("simulate", "simulate", dict(base), "Simulations"))
        if adopted and has_distributions(project, root):
            from sbmlsim.experiment.draws import UncertaintySettings

            draws = UncertaintySettings()
            uncertainty = {
                "draws": draws.draws,
                "seed": draws.seed,
                "quantiles": list(draws.quantiles),
            }
            steps.append(
                BuildStep(
                    "uncertainty",
                    "simulate",
                    {**base, "uncertainty": uncertainty},
                    "Simulations with the confidence interval",
                )
            )
    if adopted:
        report = {**base, "subset": "all", "adopted": True}
        steps.append(BuildStep("report", "report", report, "Fit reports"))
    if project.credibility is not None:
        credibility: dict[str, Any] = dict(base)
        if source_commit is not None:
            credibility["source_commit"] = source_commit
        steps.append(
            BuildStep("credibility", "credibility", credibility, "Credibility")
        )
    return steps


def has_distributions(project: Project, root: Path | None = None) -> bool:
    """Tell whether the default variant of a project has distributions.

    The experiments run over the draws of the fitted parameters from the
    distributions which the adopted fits give the parameters of the model
    (`sbmlsim.simulation.sampling.parameter_distributions`).

    Args:
        project: the project.
        root: the package of the variant, e.g. the exported package of a
            release; the root of the project without.

    Returns:
        Whether the SBML of the default variant exists and a parameter of it
        has a distribution.

    Raises:
        ReleaseError: if a distribution of the model is not read.
    """
    from sbmlsim.simulation.sampling import parameter_distributions

    if not project.variants:
        return False
    path = project.model_path
    if root is not None:
        path = root / path.relative_to(project.root)
    if not path.is_file():
        return False
    try:
        return bool(parameter_distributions(path))
    except ValueError as err:
        raise ReleaseError(str(err)) from err


def skipped_steps(project: Project, root: Path | None = None) -> list[dict[str, str]]:
    """Get the steps of the build which a project skips, with the reason.

    Args:
        project: the project.
        root: the package of the default variant, see `has_distributions`.

    Returns:
        `{"step": name, "reason": text}` of every skipped step: `uncertainty`
        for a project with experiments and adopted fits whose default variant
        has no distributions.

    Raises:
        ReleaseError: if the adopted fits are not read.
    """
    try:
        adopted = bool(project.adopted_fits())
    except ValueError as err:
        raise ReleaseError(str(err)) from err
    if project.experiments is None or not adopted or has_distributions(project, root):
        return []
    return [
        {
            "step": "uncertainty",
            "reason": "the default variant has no distributions of its fitted "
            "parameters",
        }
    ]


@dataclass(frozen=True)
class BuildResult:
    """What the build of a release wrote and ran.

    Attributes:
        written: the written paths, relative to the release repository.
        steps: the steps which ran, see `build_steps`.
        changed: the outputs of the factory which differ from the ones of the
            source package, as the kind `factory` names them; the build
            copies them over the exported package, so they are part of the
            release.
        skipped: the steps which were skipped with the reason, see
            `skipped_steps`.
    """

    written: list[Path]
    steps: list[BuildStep]
    changed: list[str]
    skipped: list[dict[str, str]]


def _replace(source: Path, target: Path) -> None:
    """Replace the folder `target` by a copy of the folder `source`.

    The records of the run which wrote the result, at the top of `source`,
    are left out (`RUN_RECORDS`).
    """

    def records(directory: str, names: list[str]) -> set[str]:
        return set(RUN_RECORDS & set(names)) if Path(directory) == source else set()

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=records)


def _remove(path: Path) -> None:
    """Remove a folder, so stale results or outputs are never reused."""
    if path.exists():
        shutil.rmtree(path)


def _step(runner: Runner, args: list[str], cwd: Path) -> None:
    """Run one step of the build with its output shown; a failure names the step."""
    try:
        runner(args, cwd, stream=True)
    except ReleaseError as err:
        raise ReleaseError(f"Build step '{' '.join(args)}' failed: {err}") from err


def _require(path: Path, step: str) -> Path:
    """Check that a step created the folder `path`."""
    if not path.is_dir():
        raise ReleaseError(f"The build step '{step}' did not create '{path}'.")
    return path


def _write_specs(steps: list[BuildStep], release_path: Path) -> list[Path]:
    """Write the specs of the steps into `specs/`."""
    folder = release_path / SPECS_DIR
    folder.mkdir(exist_ok=True)
    written = []
    for step in steps:
        text = json.dumps(step.spec, indent=2, sort_keys=True) + "\n"
        (release_path / step.spec_file).write_text(text, encoding="utf-8", newline="\n")
        written.append(Path(step.spec_file))
    return written


def _copy_models(
    project: Project, base: Path, release_path: Path, *, archive: str | None
) -> list[Path]:
    """Copy the models, the archive, the licenses and the figures into the release.

    `base` is the exported package, with the outputs of the factory copied
    over it. The models are the SBML, markdown and antimony files of its
    `factory_dir`, the licenses and the folder `figures` are taken from the
    parent of it, the archive from its path relative to the root.
    """
    assert project.factory_dir is not None
    factory_dir = base / project.factory_dir.relative_to(project.root)
    output_dir = factory_dir.parent
    written: list[Path] = []
    models_dir = release_path / BuildPaths.MODELS
    _remove(models_dir)
    models_dir.mkdir(parents=True)
    for pattern in _FACTORY_PATTERNS:
        for source in sorted(factory_dir.glob(pattern)):
            shutil.copy2(source, models_dir / source.name)
            written.append(models_dir.relative_to(release_path) / source.name)
    if not any(models_dir.glob("*.xml")):
        raise ReleaseError(f"The release has no SBML model in '{factory_dir}'.")

    if archive is not None:
        path = base / archive
        if not path.is_file():
            raise ReleaseError(f"The factory did not create the archive '{path}'.")
        shutil.copy2(path, release_path / path.name)
        written.append(Path(path.name))
    for name in _LICENSES:
        if (output_dir / name).is_file():
            shutil.copy2(output_dir / name, release_path / name)
            written.append(Path(name))

    figures_dir = output_dir / "figures"
    if figures_dir.is_dir():
        _replace(figures_dir, release_path / BuildPaths.FIGURES)
        written.append(Path(BuildPaths.FIGURES))
    return written


def _apply_factory(project: Project, out: Path, package: Path) -> list[str]:
    """Copy the rebuild of the factory over the exported package.

    Every file the factory wrote (the records of the run left out) replaces
    the one of the package, so the analyses of the release, its models and
    its archive are the ones of the factory, as a factory which builds the
    models in place gives them.

    Args:
        project: the project.
        out: the output of the step `factory`, the rebuild with the layout of
            the package.
        package: the exported package.

    Returns:
        The outputs which differ from the ones of the source package, the
        `changed` of the summary of the kind `factory`.
    """
    for source in sorted(out.rglob("*")):
        relative = source.relative_to(out)
        if relative.parts[0] in RUN_RECORDS or not source.is_file():
            continue
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    summary = out / SUMMARY_FILE
    if not summary.is_file():
        return []
    changed = json.loads(summary.read_text(encoding="utf-8")).get("changed") or []
    for path in changed:
        logger.warning(
            "The output '%s' of the factory of '%s' differs from the one of the "
            "source package, the release has the one of the factory.",
            path,
            project.name,
        )
    return [str(path) for path in changed]


def build(
    project: Project,
    reference: str,
    release_path: Path,
    runner: Runner = run,
    workers: int | None = None,
    *,
    source_commit: str | None = None,
) -> BuildResult:
    """Build the models, simulations, fit reports and the credibility in the release.

    All commands run with `release_path` as working directory: `uv lock`, `uv
    sync --locked` and the steps of `build_steps`, each an analysis of the run
    layer of sbmlsim with its spec file in `specs/` into `results/<step>`
    (ignored by git, removed before the steps). The factory runs first and
    its rebuild is copied over the exported package (`_apply_factory`), so
    the package of the release, its models, its archive and every analysis
    agree; then the steps are decided on that package (`build_steps`,
    `skipped_steps`) and run. The outputs are copied to the tracked folders
    `models`, `figures`, `simulations` and `simulations_uncertainty` (each
    with the static figures of the experiments in `_figures`,
    `sbmlsim.project.collect_figure_files`), `fit/<fit id>` and
    `credibility`; every one of them is replaced, so the reports of fits
    which are no longer adopted disappear, and the folder of a step which
    does not run is removed.

    A declared output is required, an undeclared one is optional: the models
    must have at least one SBML file and, if the settings name an archive, the
    factory must create it; the licenses and the `figures` folder are copied
    if the factory creates them. A project without a factory releases the
    models of its package, from its `factory_dir`.

    Args:
        project: the project of the source package.
        reference: the project as `module:attribute`.
        release_path: the release repository, with the package already exported.
        runner: runs the commands.
        workers: the number of workers of the simulations, the CPUs of the
            quota without.
        source_commit: the commit of the source repository the release is
            built from, recorded by the assessment of the contexts of use.

    Returns:
        What the build wrote (with the new `uv.lock` and the specs of the
        steps), the steps, the outputs the factory changed and the skipped
        steps.

    Raises:
        ReleaseError: if a step fails; the message names the step.
    """
    from sbmlsim.project import collect_figure_files

    archive = None if project.release is None else project.release.archive
    if project.factory_dir is None:
        raise ReleaseError(
            f"The project '{project.name}' has neither a directory of its factory "
            "nor model variants, the release has no models."
        )
    package = release_path / "src" / package_dirs(project)

    stage("Lock and sync of the environment")
    _step(runner, ["uv", "lock"], release_path)
    _step(runner, ["uv", "sync", "--locked"], release_path)
    _remove(release_path / BuildPaths.RESULTS)
    _remove(release_path / SPECS_DIR)
    changed: list[str] = []
    if project.factory is not None:
        factory = BuildStep("factory", "factory", {"project": reference}, "Models")
        _write_specs([factory], release_path)
        stage(factory.title)
        _step(runner, factory.command(workers), release_path)
        changed = _apply_factory(project, release_path / factory.out, package)

    steps = build_steps(project, reference, source_commit, root=package)
    skipped = skipped_steps(project, package)
    for entry in skipped:
        logger.warning(
            "The release of '%s' has no step '%s': %s.",
            project.name,
            entry["step"],
            entry["reason"],
        )
    by_name = {step.name: step for step in steps}
    written = [Path("uv.lock"), *_write_specs(steps, release_path)]
    for step in steps:
        if step.name != "factory":
            stage(step.title)
            _step(runner, step.command(workers), release_path)

    written += _copy_models(project, package, release_path, archive=archive)

    outputs = (
        ("simulate", BuildPaths.SIMULATIONS, None),
        ("uncertainty", BuildPaths.UNCERTAINTY, None),
        ("credibility", BuildPaths.CREDIBILITY, "credibility"),
    )
    for name, target, below in outputs:
        _remove(release_path / target)
        if name not in by_name:
            continue
        step = by_name[name]
        source = release_path / step.out
        if below is not None:
            source = source / below
        _replace(_require(source, " ".join(step.command())), release_path / target)
        if step.kind == "simulate":
            collect_figure_files(release_path / target, show=False)
        written.append(Path(target))

    fits_dir = release_path / BuildPaths.FITS
    _remove(fits_dir)
    fits_dir.mkdir()
    if "report" in by_name:
        reports = release_path / by_name["report"].out
        for fit_id in project.adopted_fits().values():
            _replace(
                _require(reports / fit_id, f"report of the adopted fit {fit_id}"),
                fits_dir / fit_id,
            )
            written.append(Path(BuildPaths.FITS) / fit_id)
    return BuildResult(written=written, steps=steps, changed=changed, skipped=skipped)
