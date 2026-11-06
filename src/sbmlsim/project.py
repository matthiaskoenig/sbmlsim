"""The project of a model package: the analyses it describes.

A `Project` names what a model package analyses: its simulation experiments,
its fit problems, its scans with their sensitivity indices and its uncertainty
analyses, together with the settings of the simulator and of the figures. It
is a description, nothing of it runs when it is created: the experiments, the
scans, the observables and the uncertainty models are python callables which
build their objects when an analysis runs, so a project is cheap to import and
every unit of a run builds the same objects from them, which is why every
callable must build a deterministic object, e.g. a design with a fixed seed.

A model package defines its project as an attribute of a module, e.g.
`examples.hctz_fitting.project:PROJECT`, which `load_project` imports. The run
layer, `python -m sbmlsim.run`, resolves a spec `{project: "module:attribute",
analysis: name}` (or `subset: name` for a fit) against it and lists the
analyses of a project with `python -m sbmlsim.run list <project>`, see
`Project.analyses`.

The adopted fits of a project are the fits whose parameters the package uses,
one fit id per fit subset, kept in the file `ADOPTED_FILE` of the project:
`{"fits": {"<subset>": "<fit id>"}}`, written by `write_adopted`.

`import sbmlsim` does not import this module.
"""

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, InstanceOf

from sbmlsim.atomic import write_text_atomic
from sbmlsim.engine import Engine
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.plot.plotting import FigureSettings
from sbmlsim.simulation import Observable, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.uncertainty import UncertaintyModel

__all__ = [
    "ADOPTED_FILE",
    "Project",
    "ScanAnalysis",
    "SimulatorSettings",
    "UncertaintyAnalysis",
    "load_project",
    "write_adopted",
]

#: the file of the adopted fits of a project, see `write_adopted`
ADOPTED_FILE = "adopted.json"

#: the kinds of the analyses of a fit subset, in the order `Project.analyses`
#: lists them
FIT_KINDS: tuple[str, ...] = ("fit", "identifiability", "fisher", "report", "petab")

#: what an analysis of a fit subset does, the description of its entry
_FIT_DESCRIPTIONS: dict[str, str] = {
    "fit": "fit of the problem '{name}'",
    "identifiability": "profile likelihoods of the fit of the problem '{name}'",
    "fisher": "Fisher information of the fit of the problem '{name}'",
    "report": "report of the fit of the problem '{name}'",
    "petab": "PEtab v2 problem of the fit problem '{name}'",
}

#: the configuration of the models of a project
_CONFIG = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)


class SimulatorSettings(BaseModel):
    """The settings of the simulator of the analyses of a project.

    Attributes:
        engine: the engine which simulates the models.
        integrator: the settings of the integrator by their names in
            roadrunner, see `sbmlsim.simulator.Simulator`.
    """

    model_config = _CONFIG

    engine: Engine = Engine.ROADRUNNER
    integrator: dict[str, Any] = {}

    def simulator(self, n_workers: int | None) -> Simulator:
        """Create the simulator of the settings.

        Args:
            n_workers: the workers of the simulator, see `Simulator`.

        Returns:
            A new simulator.
        """
        return Simulator(n_workers=n_workers, engine=self.engine, **self.integrator)


class ScanAnalysis(BaseModel):
    """A scan of a project, with its sensitivity indices if it has a design.

    The fields are the arguments of `Simulator.run`.

    Attributes:
        model: the model of the scan, a path or a source of
            `Simulator.load`, best an absolute path built from the module
            of the project.
        scan: builds the scan or the simulation, deterministic: a design of
            `sbmlsim.simulation.sampling` with a fixed seed.
        observables: builds the observables of the run.
        time: a grid of times in the time unit of the model the timecourses
            are interpolated onto; the output times of the simulations without.
        keep: the observables or selections the result keeps.
        on_error: `flag` or `raise`, see `Simulator.run`.
        indices: the sensitivity indices of the scan, a function of
            `sbmlsim.sensitivity` matching the design of the scan; a scan
            with indices is also a sensitivity analysis.
        index_options: the keyword arguments of the function of the indices.
        description: what the scan analyses.
    """

    model_config = _CONFIG

    model: Path | str
    scan: Callable[[], Scan | Simulation]
    observables: Callable[[], Sequence[Observable]] | None = None
    time: Sequence[float] | None = None
    keep: Sequence[str] | None = None
    on_error: Literal["raise", "flag"] = "flag"
    indices: Literal["local", "sobol", "fast", "morris"] | None = None
    index_options: dict[str, Any] = {}
    description: str = ""


class UncertaintyAnalysis(BaseModel):
    """An uncertainty analysis of a project.

    The fields are the arguments of `sbmlsim.uncertainty.propagate`.

    Attributes:
        model: the model, a path or a source, see `ScanAnalysis.model`.
        simulation: builds the simulation or the scenarios by their name.
        uq: builds what is uncertain.
        observables: builds the observables of the runs, e.g. a `PK`.
        outputs: the outputs of the summary and the adaptive stopping.
        n: the number of draws; `None` for the adaptive number.
        rel_tol: the tolerance of the adaptive stopping.
        n_start: the draws of the first batch of the adaptive stopping.
        n_max: the most draws of the adaptive stopping.
        design: `sobol`, `lhs` or `random`, the default of `propagate`
            without.
        seed: the seed of the draws.
        keep: the timecourses the results keep.
        time: a grid of times in the time unit of the model the timecourses
            are interpolated onto.
        on_error: `flag` or `raise`.
        description: what the analysis answers.
    """

    model_config = _CONFIG

    model: Path | str
    simulation: Callable[[], Simulation | Mapping[str, Simulation]]
    uq: Callable[[], UncertaintyModel]
    observables: Callable[[], Sequence[Observable]] | None = None
    outputs: Sequence[str] | None = None
    n: int | None = None
    rel_tol: float = 0.1
    n_start: int = 256
    n_max: int = 16384
    design: Literal["sobol", "lhs", "random"] | None = None
    seed: int | None = None
    keep: Sequence[str] | None = None
    time: Sequence[float] | None = None
    on_error: Literal["raise", "flag"] = "flag"
    description: str = ""


def _named[T](kind: str, items: Mapping[str, T], name: str) -> T:
    """Get an analysis of a project by its name.

    Args:
        kind: the kind of the analyses, for the error.
        items: the analyses by their name.
        name: the name of the analysis.

    Returns:
        The analysis.

    Raises:
        KeyError: naming the known names, if the project has no such analysis.
    """
    try:
        return items[name]
    except KeyError:
        raise KeyError(
            f"The project has no {kind} '{name}', it has {sorted(items)}."
        ) from None


class Project(BaseModel):
    """The analyses of a model package, see the module.

    Attributes:
        name: the name of the project.
        root: the directory of the model package.
        experiments: builds the simulation experiment classes of the project.
        base_path: the base path of the experiments, `root` without.
        data_path: the path or paths of the datasets of the experiments.
        fits: the fit problems by their subset.
        adopted: the file of the adopted fits, see `ADOPTED_FILE`.
        scans: the scans by their name.
        uncertainties: the uncertainty analyses by their name.
        simulator: the settings of the simulator of the analyses.
        figure_settings: the settings of the figures of a run.
        description: what the project is.
    """

    model_config = _CONFIG

    name: str
    root: Path
    experiments: Callable[[], Sequence[type[SimulationExperiment]]] | None = None
    base_path: Path | None = None
    data_path: Path | Sequence[Path] | None = None
    fits: Mapping[str, InstanceOf[FitDefinition]] = {}
    adopted: Path | None = None
    scans: Mapping[str, ScanAnalysis] = {}
    uncertainties: Mapping[str, UncertaintyAnalysis] = {}
    simulator: SimulatorSettings = SimulatorSettings()
    figure_settings: InstanceOf[FigureSettings] | None = None
    description: str = ""

    def fit(self, subset: str) -> FitDefinition:
        """Get a fit problem.

        Args:
            subset: the subset of the fit.

        Returns:
            The definition of the fit.

        Raises:
            KeyError: naming the subsets, for an unknown subset.
        """
        return _named("fit subset", self.fits, subset)

    def scan(self, name: str) -> ScanAnalysis:
        """Get a scan.

        Args:
            name: the name of the scan.

        Returns:
            The scan.

        Raises:
            KeyError: naming the scans, for an unknown scan.
        """
        return _named("scan", self.scans, name)

    def uncertainty(self, name: str) -> UncertaintyAnalysis:
        """Get an uncertainty analysis.

        Args:
            name: the name of the analysis.

        Returns:
            The uncertainty analysis.

        Raises:
            KeyError: naming the analyses, for an unknown analysis.
        """
        return _named("uncertainty analysis", self.uncertainties, name)

    def adopted_fits(self) -> dict[str, str]:
        """Get the adopted fits.

        Returns:
            The fit id by subset; empty without the file of the adopted fits.

        Raises:
            ValueError: if the file is no file of adopted fits.
        """
        if self.adopted is None or not self.adopted.exists():
            return {}
        content = json.loads(self.adopted.read_text(encoding="utf-8"))
        fits = content.get("fits") if isinstance(content, dict) else None
        if not isinstance(fits, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in fits.items()
        ):
            raise ValueError(
                f"'{self.adopted}' is no file of adopted fits, "
                '{"fits": {"<subset>": "<fit id>"}}.'
            )
        return dict(fits)

    def analyses(self) -> list[dict[str, str]]:
        """List the analyses of the project.

        Returns:
            One entry `{"kind", "name", "description"}` per analysis: the
            simulation experiments (`simulate`, `all`), the analyses of every
            fit subset (`fit`, `identifiability`, `fisher`, `report`,
            `petab`), every scan (`scan`) and, for a scan with indices, its
            sensitivity analysis (`sensitivity`), and every uncertainty
            analysis (`uncertainty`).
        """
        entries: list[dict[str, str]] = []

        def add(kind: str, name: str, description: str) -> None:
            entries.append({"kind": kind, "name": name, "description": description})

        if self.experiments is not None:
            add("simulate", "all", "the simulation experiments of the project")
        for subset, definition in self.fits.items():
            for kind in FIT_KINDS:
                description = _FIT_DESCRIPTIONS[kind].format(name=subset)
                if kind == "fit" and definition.description:
                    description = definition.description
                add(kind, subset, description)
        for name, scan in self.scans.items():
            add("scan", name, scan.description or f"the scan '{name}'")
        for name, scan in self.scans.items():
            if scan.indices is not None:
                add("sensitivity", name, f"{scan.indices} indices of the scan '{name}'")
        for name, uncertainty in self.uncertainties.items():
            add(
                "uncertainty",
                name,
                uncertainty.description or f"the uncertainty analysis '{name}'",
            )
        return entries


def load_project(ref: str) -> Project:
    """Load a project from its reference.

    Args:
        ref: `module:attribute`, e.g. `examples.hctz_fitting.project:PROJECT`.

    Returns:
        The project.

    Raises:
        ValueError: for a reference without `:`, a module without the
            attribute or an attribute which is no `Project`.
        ImportError: if the module is not imported.
    """
    module_name, colon, attribute = ref.partition(":")
    if not colon or not module_name or not attribute:
        raise ValueError(f"A project is 'module:attribute', not '{ref}'.")
    module = importlib.import_module(module_name)
    try:
        project = getattr(module, attribute)
    except AttributeError:
        raise ValueError(
            f"The module '{module_name}' has no attribute '{attribute}'."
        ) from None
    if not isinstance(project, Project):
        raise ValueError(f"'{ref}' is no Project, it is a '{type(project).__name__}'.")
    return project


def write_adopted(path: Path, fits: Mapping[str, str]) -> None:
    """Write the adopted fits of a project, atomically.

    Args:
        path: the file, see `ADOPTED_FILE`.
        fits: the fit id by subset.
    """
    content = json.dumps({"fits": dict(fits)}, indent=2, sort_keys=True) + "\n"
    write_text_atomic(path, content)
