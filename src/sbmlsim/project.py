"""The project of a model package: its models, analyses and settings.

A `Project` describes a model package once: its model variants and the factory
which builds them, its simulation experiments, its fit problems and the records
of its adopted fits, its scans with their sensitivity indices, its
sensitivity analyses over the groups of its simulations, its uncertainty
analyses and its actions (analyses of its own), the settings of
the simulator and of the figures, the known exceptions of its checks
(`sbmlsim.project_checks`) and its release metadata. It is a description,
nothing of it runs when it is created.

A model package defines its project as an attribute of a module, e.g.
`examples.hctz_fitting.project:PROJECT`, which `load_project` imports. The run
layer, `python -m sbmlsim.run`, resolves a spec `{project: "module:attribute",
analysis: name}` (or `subset: name` for a fit) against it and lists the
analyses of a project with `python -m sbmlsim.run list <project>`, see
`Project.analyses`.

The experiments, the fits, the factory, the credibility declaration, the
sensitivity analyses and the functions of the actions of a project are objects
or references `"module:attribute"` to them, which are resolved when they are
used (`resolve`) and kept, so that importing the module of a project imports none
of them: the experiments import the module of their project for its paths,
so the project cannot import them, and a package of many models stays cheap to
import. A module which starts with a dot is relative to
`Project.package`. The scans, the observables and the uncertainty models are
python callables which build their objects when an analysis runs, so every
unit of a run builds the same objects from them, which is why every callable
must build a deterministic object, e.g. a design with a fixed seed.

A relative path of a project (its model variants, the paths of its experiments,
the directory of its fit records and the file of its adopted fits) is relative
to its `root`, the directory of the model package; the outputs of a run are
not: they go where the run is told, e.g. `Project.run_experiments(..., out)`.

The adopted fits of a project are the fits whose parameters the package uses,
one fit id per fit subset, kept in the file `ADOPTED_FILE`, by default in the
directory of the records of the adopted fits (`Project.fits_dir`):
`{"fits": {"<subset>": "<fit id>"}}`, written by `write_adopted`.

`import sbmlsim` does not import this module.
"""

import dataclasses
import importlib
import json
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    InstanceOf,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from sbmlsim.atomic import write_text_atomic
from sbmlsim.engine import Engine, JitOptimization
from sbmlsim.reference import check_reference
from sbmlsim.release.settings import ReleaseSettings
from sbmlsim.simulation import Observable, Scan, Simulation

if TYPE_CHECKING:
    import roadrunner

    from sbmlsim.credibility.battery import BatteryHook
    from sbmlsim.experiment import (
        ExperimentResult,
        ExperimentRunner,
        SimulationExperiment,
    )
    from sbmlsim.experiment.draws import UncertaintySettings
    from sbmlsim.fit.cli import FitDefinition
    from sbmlsim.fit.records import AdoptedRecords
    from sbmlsim.model.tolerances import AbsoluteTolerance
    from sbmlsim.plot.plotting import FigureSettings
    from sbmlsim.release.hooks import ReleaseHook
    from sbmlsim.simulator import Simulator
    from sbmlsim.uncertainty import UncertaintyModel
else:
    # the module stays light: it imports neither the experiments, the fits nor
    # roadrunner. pydantic reads the annotations of the fields at runtime, where
    # these types are `Any`; the validators check the instances (`_instance_of`)
    # and the methods import the modules when they run.
    SimulationExperiment = FitDefinition = AbsoluteTolerance = Any
    FigureSettings = UncertaintyModel = Any

__all__ = [
    "ADOPTED_FILE",
    "ALL_SUBSETS",
    "COLLECTED_FORMATS",
    "FIGURES_DIR",
    "ActionFunction",
    "CheckSettings",
    "CredibilitySettings",
    "FigureOutputs",
    "Project",
    "ProjectAction",
    "ScanAnalysis",
    "SimulatorSettings",
    "UncertaintyAnalysis",
    "check_reference",
    "collect_figure_files",
    "load_project",
    "normalize_formats",
    "resolve",
    "same_antimony",
    "same_sbml",
    "write_adopted",
]

#: the formats of the figures which a run of experiments with `collect_figures`
#: copies into `FIGURES_DIR` of its output, see `Project.run_experiments`
COLLECTED_FORMATS: tuple[str, ...] = ("png", "svg")

#: the directory of the collected figures in the output of a run of experiments
FIGURES_DIR = "_figures"

#: the file of the adopted fits of a project, see `write_adopted`
ADOPTED_FILE = "adopted.json"

#: the group of the experiments of a project which has every experiment, see
#: `Project.load_experiments`
ALL_EXPERIMENTS = "all"

#: the subset of the report of the adopted fits which reports every adopted
#: fit, no fit subset of a project
ALL_SUBSETS = "all"

#: the name of the analysis of the factory in `Project.analyses`
FACTORY_ANALYSIS = "models"

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

#: the settings of the simulator which are fields of `SimulatorSettings` or
#: arguments of the simulator and not settings of the integrator
_NOT_INTEGRATOR_SETTINGS = frozenset({"engine", "n_workers", "jit_optimization"})

#: the configuration of the models of a project
_CONFIG = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

#: a line of the SBML of a model with a date of its model history
_HISTORY_DATE = re.compile(r"<dcterms:(created|modified)\b|<dcterms:W3CDTF>")

#: a line of the antimony of a model with a date of its history, e.g.
#: `  model created "2026-10-17T19:35:15Z"`
_ANTIMONY_DATE = re.compile(r'^\s*model (created|modified) "')

#: the suffixes of the outputs of a factory which are compared as SBML
_SBML_SUFFIXES = frozenset({".xml", ".sbml"})

#: the suffixes of the outputs of a factory which are compared as antimony
_ANTIMONY_SUFFIXES = frozenset({".ant"})


def same_sbml(a: str, b: str) -> bool:
    """Compare the SBML of two models without the dates of their model history.

    A factory writes the dates of its run into the model history, so a rebuild
    of an unchanged model differs from the model only in the lines of these
    dates.

    Args:
        a: the content of one file.
        b: the content of the other.

    Returns:
        Whether they are equal but for the lines with a date.
    """

    def lines(text: str) -> list[str]:
        return [line for line in text.splitlines() if not _HISTORY_DATE.search(line)]

    return lines(a) == lines(b)


def same_antimony(a: str, b: str) -> bool:
    """Compare the antimony of two models without the dates of their history.

    A factory which writes the antimony of a model writes the dates of its run
    as `model created "..."` and `model modified "..."`.

    Args:
        a: the content of one file.
        b: the content of the other.

    Returns:
        Whether they are equal but for the lines with a date.
    """

    def lines(text: str) -> list[str]:
        return [line for line in text.splitlines() if not _ANTIMONY_DATE.match(line)]

    return lines(a) == lines(b)


def _same_output(a: Path, b: Path) -> bool:
    """Compare two outputs of a factory, SBML and antimony without their dates."""
    suffix = a.suffix.lower()
    compare = (
        same_sbml
        if suffix in _SBML_SUFFIXES
        else same_antimony
        if suffix in _ANTIMONY_SUFFIXES
        else None
    )
    if compare is not None:
        try:
            return compare(a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            pass
    return a.read_bytes() == b.read_bytes()


def resolve(reference: str, package: str | None = None) -> Any:
    """Resolve a reference `"module:attribute"` to the object it names.

    Args:
        reference: the reference; a module which starts with a dot is relative
            to `package`.
        package: the package of relative references.

    Returns:
        The object.

    Raises:
        ValueError: if the reference is malformed or relative without a package.
        AttributeError: if the module has no such attribute; the message names
            the reference.
    """
    check_reference(reference)
    module_name, _, attribute = reference.partition(":")
    if module_name.startswith(".") and package is None:
        raise ValueError(f"The relative reference '{reference}' needs a package.")
    obj: Any = importlib.import_module(module_name, package)
    for name in attribute.split("."):
        try:
            obj = getattr(obj, name)
        except AttributeError as err:
            raise AttributeError(f"'{reference}' cannot be resolved: {err}") from err
    return obj


def normalize_formats(formats: Sequence[str]) -> tuple[str, ...]:
    """Get figure formats as a lower-case tuple.

    Args:
        formats: the formats, e.g. `["SVG", "png"]`.

    Returns:
        The formats in lower case, e.g. `("svg", "png")`.

    Raises:
        TypeError: if the formats are a single string, e.g. `"svg"` instead of
            `("svg",)`; a string is a sequence of characters.
    """
    if isinstance(formats, str):
        raise TypeError(
            f"The figure formats must be a sequence of strings, not {formats!r}; "
            f"use ({formats!r},)."
        )
    return tuple(f.lower() for f in formats)


def _instance_of(value: object, module: str, name: str) -> bool:
    """Check that a value is an instance of a class without importing its module.

    An instance of a class exists only once its module is imported, so a
    module which is not imported has no instances.

    Args:
        value: the value.
        module: the module of the class.
        name: the name of the class.

    Returns:
        Whether the value is an instance of the class.
    """
    loaded = sys.modules.get(module)
    cls = getattr(loaded, name, None) if loaded is not None else None
    return isinstance(cls, type) and isinstance(value, cls)


class _Resolved(dict[tuple[str | None, str], Any]):
    """The objects of the resolved references of a project, by package and reference.

    A cache, which is no part of the value of a project: two projects are
    equal whatever they resolved, and a copy of a project shares the cache,
    which is keyed by the reference and its package.
    """

    @override
    def __eq__(self, other: object) -> bool:
        """Every cache equals every other one."""
        return isinstance(other, _Resolved)

    __hash__ = None


class SimulatorSettings(BaseModel):
    """The settings of the simulator of the analyses of a project.

    The processes are no setting of a project, they belong to a run, see
    `simulator`.

    Attributes:
        engine: the engine which simulates the models.
        absolute_tolerance: the absolute tolerance of the integrator, one value
            or an `AbsoluteTolerance` per kind and state; `None` for the
            default of `Simulator`.
        relative_tolerance: the relative tolerance of the integrator; `None`
            for the default of `Simulator`.
        jit_optimization: the optimization level LLJIT compiles the models of
            the experiments, the fits and the tests of the project with
            (`JitOptimization`, `"O0"` to `"O3"`), `None` for the level of
            `resolve_jit_optimization`. `"O0"` compiles a model with large
            initial assignments several times faster and simulates a little
            slower. The fits of the project carry it into their problems and
            workers, see `Project.load_fit_definitions`.
        integrator: the other settings of the integrator by their names in
            roadrunner, see `sbmlsim.simulator.Simulator`, e.g.
            `{"variable_step_size": False}`.
    """

    model_config = _CONFIG

    engine: Engine = Engine.ROADRUNNER
    absolute_tolerance: float | AbsoluteTolerance | None = None
    relative_tolerance: float | None = None
    jit_optimization: JitOptimization | None = None
    integrator: dict[str, Any] = {}

    @field_validator("absolute_tolerance")
    @classmethod
    def _check_tolerance(cls, value: Any) -> Any:
        """Check that the absolute tolerance is a number or an `AbsoluteTolerance`.

        Raises:
            ValueError: for another value.
        """
        if value is None or isinstance(value, float):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if not _instance_of(value, "sbmlsim.model.tolerances", "AbsoluteTolerance"):
            raise ValueError(
                f"The absolute tolerance is a number or an AbsoluteTolerance, not "
                f"{value!r}."
            )
        return value

    @model_validator(mode="after")
    def _check_integrator(self) -> SimulatorSettings:
        """Check that a setting is given once.

        Raises:
            ValueError: if the integrator settings repeat a tolerance which is
                a field, or name the engine, the processes or the optimization
                level.
        """
        names = sorted(set(self.integrator) & _NOT_INTEGRATOR_SETTINGS)
        if names:
            raise ValueError(
                f"The integrator settings name {names}, which are no settings of "
                "the integrator: the engine and the optimization level are fields "
                "of the simulator settings, the processes belong to a run."
            )
        for name in ("absolute_tolerance", "relative_tolerance"):
            if getattr(self, name) is not None and name in self.integrator:
                raise ValueError(
                    f"The simulator settings give '{name}' twice, as a field and "
                    "in the integrator settings."
                )
        return self

    def integrator_settings(self) -> dict[str, Any]:
        """Get the settings of the integrator, the tolerances with the others.

        Returns:
            The settings by their names in roadrunner.
        """
        settings = dict(self.integrator)
        if self.absolute_tolerance is not None:
            settings["absolute_tolerance"] = self.absolute_tolerance
        if self.relative_tolerance is not None:
            settings["relative_tolerance"] = self.relative_tolerance
        return settings

    def simulator(self, n_workers: int | None) -> Simulator:
        """Create the simulator of the settings.

        Args:
            n_workers: the workers of the simulator, see `Simulator`.

        Returns:
            A new simulator.
        """
        from sbmlsim.simulator import Simulator

        return Simulator(
            n_workers=n_workers, engine=self.engine, **self.integrator_settings()
        )


class FigureOutputs(BaseModel):
    """The files of the figures of the runs of a project.

    The rendering of a figure (the resolution, the sizes of the panels, the
    fonts and the legends) is `Project.figure_settings`, the
    `sbmlsim.plot.FigureSettings` of every figure of a run; these are the
    formats of the files and the parameters of matplotlib while the figures of
    a run are drawn.

    Attributes:
        formats: the formats of the figures, static images drawn by matplotlib
            (`svg`, `png`) and interactive pages drawn by plotly (`html`); SVG
            only by default. Stored in lower case.
        savefig_dpi: the resolution of the static images, the `savefig.dpi` of
            matplotlib, `None` keeps the one of matplotlib. An interactive page
            has 72 pixels per inch.
        rc: other parameters of matplotlib while the figures are drawn, e.g.
            `{"font.family": "serif"}`.
    """

    model_config = _CONFIG

    formats: tuple[str, ...] = ("svg",)
    savefig_dpi: float | None = None
    rc: dict[str, Any] = {}

    @field_validator("formats", mode="before")
    @classmethod
    def _normalize_formats(cls, formats: Any) -> Any:
        """Refuse a single string and write the formats in lower case.

        Raises:
            TypeError: if the formats are a single string.
            ValueError: if there is no format.
        """
        if isinstance(formats, Sequence):
            formats = normalize_formats(formats)
            if not formats:
                raise ValueError("A project needs at least one figure format.")
        return formats

    def rc_params(self) -> dict[str, Any]:
        """Get the parameters of matplotlib while the figures of a run are drawn.

        Returns:
            `rc` with `savefig.dpi` of `savefig_dpi`.
        """
        params = dict(self.rc)
        if self.savefig_dpi is not None:
            params["savefig.dpi"] = self.savefig_dpi
        return params


class CheckSettings(BaseModel):
    """The known exceptions of the checks of a model package, each with its reason.

    The checks are the functions of `sbmlsim.project_checks`.

    Attributes:
        experiment_modules: the module of the experiments relative to the
            package, a package whose modules define the experiment classes or
            a module which defines them, see
            `sbmlsim.project_checks.experiment_modules`.
        not_registered: experiment classes which are defined but not in the
            group `all` of the experiments on purpose, by class name.
        not_importable: modules below the package which do not import in the
            test environment (an optional dependency), by their name relative to
            the package, e.g. `experiments.zonation.plot_pyvista`.
        one_at_a_time: why the checks run the experiments one at a time (the
            memory a run of all of them needs), `None` to run them together.
        battery: the checks of the battery of the model which fail, by the
            name of the check, e.g. `steady_state`; an assessment shows a
            known failure with its reason and still fails it.
    """

    model_config = _CONFIG

    experiment_modules: str = "experiments"
    not_registered: Mapping[str, str] = {}
    not_importable: Mapping[str, str] = {}
    one_at_a_time: str | None = None
    battery: Mapping[str, str] = {}

    @model_validator(mode="after")
    def _check_reasons(self) -> CheckSettings:
        """Check that an exception has a reason.

        Raises:
            ValueError: if `one_at_a_time` or the reason of a failure of the
                battery is empty.
        """
        if self.one_at_a_time is not None and not self.one_at_a_time.strip():
            raise ValueError(
                "one_at_a_time needs a reason, e.g. the memory a run of all "
                "experiments needs; None runs them together."
            )
        for check, reason in self.battery.items():
            if not check.strip() or not reason.strip():
                raise ValueError(
                    f"The failure of the battery '{check}' needs a name and a reason."
                )
        return self


class CredibilitySettings(BaseModel):
    """The settings of the credibility of a project which belong to its domain.

    The declaration of the contexts of use (`Project.credibility`) is checked
    and assessed with them, see `sbmlsim.credibility`.

    Attributes:
        metadata_type: the metadata of the fit mappings of the project, a
            dataclass derived from `sbmlsim.fit.objects.MappingMetaData`, or a
            reference to it; `MappingMetaData` without, which has no fields.
        domain_fields: the fields of the metadata a domain of a context of
            use can claim, every field whose type is an enum without, see
            `sbmlsim.credibility.metadata_fields`.
        extensions: the evidence extensions of the project, instances of
            `sbmlsim.credibility.extensions.EvidenceExtension` or references
            to them, see `Project.evidence_extensions`.
        battery: the battery hook of the project, which adds the simulations
            and checks of its domain to the battery of its variants, an
            instance of `sbmlsim.credibility.battery.BatteryHook` or a
            reference to it, see `Project.battery_hook`; `None` for the
            generic checks only.
        battery_settings: the settings of the battery over the ones of the
            hook (the defaults without one), by the names of the fields of
            `sbmlsim.credibility.battery.BatterySettings`, e.g.
            `{"engines": ["copasi"], "horizon": 60.0, "steady_state": True}`;
            checked when the battery runs.
    """

    model_config = _CONFIG

    metadata_type: Any = None
    domain_fields: tuple[str, ...] | None = None
    extensions: tuple[Any, ...] = ()
    battery: Any = None
    battery_settings: Mapping[str, Any] = {}

    @field_validator("metadata_type")
    @classmethod
    def _check_metadata_type(cls, value: Any) -> Any:
        """Check that the metadata is a class or a reference.

        Raises:
            ValueError: for something else.
        """
        if value is None or isinstance(value, type):
            return value
        if isinstance(value, str):
            check_reference(value)
            return value
        raise ValueError(
            f"The metadata type is a class or a reference 'module:attribute', not "
            f"{value!r}."
        )

    @field_validator("extensions")
    @classmethod
    def _check_extensions(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        """Check the references of the extensions, an object is checked when used.

        Raises:
            ValueError: for a malformed reference.
        """
        for extension in value:
            if isinstance(extension, str):
                check_reference(extension)
        return value

    @field_validator("battery")
    @classmethod
    def _check_battery(cls, value: Any) -> Any:
        """Check the reference of the battery hook, an object is checked when used.

        Raises:
            ValueError: for a malformed reference.
        """
        if isinstance(value, str):
            check_reference(value)
        return value


class ScanAnalysis(BaseModel):
    """A scan of a project, with its sensitivity indices if it has a design.

    The fields are the arguments of `Simulator.run`.

    Attributes:
        model: the model of the scan, a path or a source of
            `Simulator.load`; a relative path is relative to `Project.root`.
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


class ActionFunction(Protocol):
    """The function of an action of a project, see `ProjectAction`.

    It is called with the project, the output directory of the run, which
    exists and which the action writes into, the number of processes the run
    may use and the arguments of the spec of the run as keywords, and returns
    the summary of the run, JSON values by name, or `None`.
    """

    def __call__(
        self, project: Project, out: Path, /, *, workers: int, **arguments: Any
    ) -> Mapping[str, Any] | None:
        """Run the action into `out`."""
        ...


class ProjectAction(BaseModel):
    """An action of a project, an analysis of its own, e.g. a table of its studies.

    The kind `action` of the run layer runs it into its output directory,
    `python -m sbmlsim.run action <spec>` with the spec `{project, name,
    arguments}`, see `sbmlsim.run.project_kinds`.

    Attributes:
        function: the function of the action, see `ActionFunction`, or a
            reference `"module:attribute"` to it.
        description: what the action does, its entry in `Project.analyses`.
    """

    model_config = _CONFIG

    function: Callable[..., Any] | str
    description: str = ""

    @field_validator("function")
    @classmethod
    def _check_function(cls, value: Any) -> Any:
        """Check the form of a reference, without resolving it.

        Raises:
            ValueError: for a malformed reference.
        """
        if isinstance(value, str):
            check_reference(value)
        return value


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


#: the experiments of a project as they resolve: experiment classes, groups of
#: them by name, or a callable which returns either
type _Experiments = (
    Sequence[type[SimulationExperiment]]
    | Mapping[str, Sequence[type[SimulationExperiment]]]
)


class Project(BaseModel):
    """A model package: its models, analyses and settings, see the module.

    Attributes:
        name: the name of the project.
        root: the directory of the model package; the relative paths of the
            project are relative to it.
        package: the dotted name of the model package, e.g. `__name__` in its
            `__init__.py`; relative references are relative to it.
        title: the title of the model in its reports, e.g. the title of the
            report of a fit; the name without.
        description: what the project is.
        variants: the SBML files of the model by the name of its variant, e.g.
            `{"default": ..., "renal": ...}`, the first the default, see
            `model_path`; every variant is a file of its own.
        factory: builds the models, a function called with the directory it
            writes into, or a reference to it, see `create_models`.
        factory_dir: the directory the factory writes into in the package,
            the directory of the first variant without; inside of `root`.
        factory_outputs: the files the factory writes, relative to `root`;
            the factory may write next to `factory_dir` as well, e.g. an
            archive into its parent, see `create_models`.
        experiments: the simulation experiment classes, a sequence of them,
            groups of them by name (the group `all` has every experiment, the
            groups together without it), a callable which returns either, or
            a reference to one of these; see `load_experiments`.
        base_path: the base path of the experiments, `root` without.
        data_path: the path or paths of the datasets of the experiments.
        fits: the fit problems by their subset, or a reference to them; see
            `load_fit_definitions`.
        fits_dir: the directory of the records of the adopted fits, one
            directory per fit id, see `adopted_record`.
        adopted: the file of the adopted fits, `ADOPTED_FILE` in `fits_dir`
            without, see `adopted_file`.
        credibility: the declaration of the contexts of use of the model, or a
            reference to it, see `load_credibility`.
        credibility_settings: the metadata of the fit mappings, the fields of
            it a domain can claim and the evidence extensions of the
            credibility of the project.
        scans: the scans by their name.
        uncertainties: the uncertainty analyses by their name.
        sensitivities: the sensitivity analyses of the model over the groups
            of its simulations by their name, see
            `sbmlsim.sensitivity.runner`: a `SensitivityRunner`, the runners
            of the subfolders of a model with several simulations by the name
            of their subfolder, or a reference to either; a runner without a
            project is bound to this one, see `sensitivity`. A name is no name
            of a scan with indices, which is a sensitivity analysis as well.
        actions: the actions of the project by their name, analyses of its
            own which the kind `action` runs, see `ProjectAction`; a function
            or a reference is an action without a description.
        simulator: the settings of the simulator of the analyses.
        figure_settings: the settings of the rendering of the figures of a run.
        figures: the formats of the figures of a run and the parameters of
            matplotlib while they are drawn.
        checks: the known exceptions of the checks of the model.
        release: the release metadata, `None` for a project which is not
            released.
    """

    model_config = _CONFIG

    name: str
    root: Path
    package: str | None = None
    title: str = ""
    description: str = ""
    variants: Mapping[str, Path] = {}
    factory: Callable[[Path], Any] | str | None = None
    factory_dir: Path | None = None
    factory_outputs: Sequence[Path] = ()
    experiments: Callable[[], _Experiments] | _Experiments | str | None = None
    base_path: Path | None = None
    data_path: Path | Sequence[Path] | None = None
    fits: Mapping[str, FitDefinition] | str = {}
    fits_dir: Path | None = None
    adopted: Path | None = None
    credibility: Any = None
    credibility_settings: CredibilitySettings = CredibilitySettings()
    scans: Mapping[str, ScanAnalysis] = {}
    uncertainties: Mapping[str, UncertaintyAnalysis] = {}
    sensitivities: Mapping[str, Any] = {}
    actions: Mapping[str, ProjectAction] = {}
    simulator: SimulatorSettings = SimulatorSettings()
    figure_settings: FigureSettings | None = None
    figures: FigureOutputs = FigureOutputs()
    checks: CheckSettings = CheckSettings()
    release: InstanceOf[ReleaseSettings] | None = None

    #: the objects of the references, resolved on their first use
    _resolved: _Resolved = PrivateAttr(default_factory=_Resolved)

    # ------------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------------
    @field_validator("experiments", "fits", "factory", "credibility")
    @classmethod
    def _check_reference(cls, value: Any, info: ValidationInfo) -> Any:
        """Check the form of a reference, without resolving it.

        Raises:
            ValueError: for a malformed reference or a relative one without a
                package.
        """
        if isinstance(value, str):
            check_reference(value)
            if value.startswith(".") and info.data.get("package") is None:
                raise ValueError(
                    f"The relative reference '{value}' needs the package of the "
                    "project."
                )
        return value

    @field_validator("sensitivities")
    @classmethod
    def _check_sensitivities(cls, value: Any, info: ValidationInfo) -> Any:
        """Check the sensitivity analyses, a reference is checked when used.

        Raises:
            ValueError: for an analysis which is neither a runner, runners by
                subfolder nor a reference, a relative reference without a
                package, and a name of a scan with indices.
        """
        runner = "sbmlsim.sensitivity.runner"
        for name, target in value.items():
            if isinstance(target, str):
                check_reference(target)
                if target.startswith(".") and info.data.get("package") is None:
                    raise ValueError(
                        f"The relative reference '{target}' of the sensitivity "
                        f"analysis '{name}' needs the package of the project."
                    )
                continue
            runners = list(target.values()) if isinstance(target, Mapping) else [target]
            if not runners or not all(
                _instance_of(r, runner, "SensitivityRunner") for r in runners
            ):
                raise ValueError(
                    f"The sensitivity analysis '{name}' is a SensitivityRunner, a "
                    "mapping of them by subfolder or a reference to either, not "
                    f"{target!r}."
                )
        scans = info.data.get("scans") or {}
        clashes = sorted(
            name
            for name in value
            if name in scans and getattr(scans[name], "indices", None) is not None
        )
        if clashes:
            raise ValueError(
                f"The sensitivity analyses {clashes} are scans with indices as "
                "well, which the kind 'sensitivity' runs by the same name: give "
                "them other names."
            )
        return value

    @field_validator("actions", mode="before")
    @classmethod
    def _check_actions(cls, value: Any, info: ValidationInfo) -> Any:
        """Get the actions, a function or a reference an action of its own.

        Raises:
            ValueError: for an action without a name and a relative reference
                without a package.
        """
        if not isinstance(value, Mapping):
            return value
        actions: dict[Any, Any] = {}
        for name, action in value.items():
            if isinstance(name, str) and not name.strip():
                raise ValueError("An action of a project needs a name.")
            if isinstance(action, str) or (
                callable(action) and not isinstance(action, (ProjectAction, type))
            ):
                action = ProjectAction(function=action)
            function = (
                action.function
                if isinstance(action, ProjectAction)
                else action.get("function")
                if isinstance(action, Mapping)
                else None
            )
            if (
                isinstance(function, str)
                and function.startswith(".")
                and info.data.get("package") is None
            ):
                raise ValueError(
                    f"The relative reference '{function}' of the action '{name}' "
                    "needs the package of the project."
                )
            actions[name] = action
        return actions

    @field_validator("base_path", "fits_dir", "adopted", "factory_dir")
    @classmethod
    def _relative_to_root(cls, value: Path | None, info: ValidationInfo) -> Any:
        """Get a relative path relative to the root."""
        root = info.data.get("root")
        return value if value is None or root is None else root / value

    @field_validator("data_path")
    @classmethod
    def _data_relative_to_root(
        cls, value: Path | Sequence[Path] | None, info: ValidationInfo
    ) -> Any:
        """Get the relative paths of the data relative to the root."""
        root = info.data.get("root")
        if value is None or root is None:
            return value
        if isinstance(value, Path):
            return root / value
        return [root / path for path in value]

    @field_validator("variants")
    @classmethod
    def _check_variants(cls, value: Mapping[str, Path], info: ValidationInfo) -> Any:
        """Check the variants, their relative paths relative to the root.

        Raises:
            ValueError: for a variant without a name or two variants of one
                file.
        """
        root = info.data.get("root")
        variants = {
            name: path if root is None else root / path for name, path in value.items()
        }
        if any(not name.strip() for name in variants):
            raise ValueError("A model variant needs a name.")
        paths = list(variants.values())
        repeated = sorted({str(p) for p in paths if paths.count(p) > 1})
        if repeated:
            raise ValueError(
                f"The model variants share the files {repeated}: every variant is "
                "a file of its own."
            )
        return variants

    @field_validator("factory_outputs")
    @classmethod
    def _check_factory_outputs(cls, value: Sequence[Path]) -> tuple[Path, ...]:
        """Check that the outputs of the factory are inside of the root.

        Raises:
            ValueError: for an absolute output or one with `..`.
        """
        outputs = tuple(value)
        outside = [str(p) for p in outputs if p.is_absolute() or ".." in p.parts]
        if outside:
            raise ValueError(
                f"The outputs of the factory are paths relative to the root of the "
                f"project, not {outside}."
            )
        return outputs

    @field_validator("fits")
    @classmethod
    def _check_fits(cls, value: Any) -> Any:
        """Check that the fits are fit definitions, a reference is checked when used.

        Raises:
            ValueError: for a fit which is no `FitDefinition`.
        """
        if isinstance(value, str):
            return value
        wrong = [
            subset
            for subset, definition in value.items()
            if not _instance_of(definition, "sbmlsim.fit.cli", "FitDefinition")
        ]
        if wrong:
            raise ValueError(f"The fits {wrong} are no FitDefinition.")
        return value

    @field_validator("figure_settings")
    @classmethod
    def _check_figure_settings(cls, value: Any) -> Any:
        """Check that the figure settings are a `FigureSettings`.

        Raises:
            ValueError: for other settings.
        """
        if value is not None and not _instance_of(
            value, "sbmlsim.plot.plotting", "FigureSettings"
        ):
            raise ValueError(
                f"The figure settings are a sbmlsim.plot.FigureSettings, not {value!r}."
            )
        return value

    @model_validator(mode="after")
    def _check_factory_dir(self) -> Self:
        """Set the directory of the factory, the one of the first variant without.

        Raises:
            ValueError: for a directory of the factory outside of the root.
        """
        if self.factory_dir is None and self.variants:
            first = next(iter(self.variants.values()))
            self.__dict__["factory_dir"] = first.parent
        if (
            self.factory is not None
            and self.factory_dir is not None
            and not self.factory_dir.is_relative_to(self.root)
        ):
            raise ValueError(
                f"The directory of the factory {self.factory_dir} is outside of "
                f"the root {self.root} of the project."
            )
        return self

    def replace(self, **changes: Any) -> Self:
        """Get a copy of the project with other fields, validated again.

        The copy is created by the constructor, so the references are checked
        and relative paths of the changes are relative to `root`, unlike
        `model_copy(update=...)`, which validates nothing. The paths of the
        project are stored relative to its root, i.e. absolute: a change of
        `root` alone does not move them, and the directory of the factory
        which the project derived from its first variant stays unless it is
        changed as well.

        Args:
            **changes: the fields and their new values.

        Returns:
            The new project.

        Raises:
            pydantic.ValidationError: for invalid fields.
        """
        return type(self)(**{**dict(self), **changes})

    # ------------------------------------------------------------------------
    # references
    # ------------------------------------------------------------------------
    def _resolve(self, value: Any) -> Any:
        """Resolve a reference of the project once, an object is itself.

        Args:
            value: a reference `"module:attribute"` or an object.

        Returns:
            The object.
        """
        if not isinstance(value, str):
            return value
        key = (self.package, value)
        if key not in self._resolved:
            self._resolved[key] = resolve(value, self.package)
        return self._resolved[key]

    @override
    def __getstate__(self) -> dict[Any, Any]:
        """Get the state of a project without its resolved references."""
        state = super().__getstate__()
        private = dict(state.get("__pydantic_private__") or {})
        private["_resolved"] = _Resolved()
        state["__pydantic_private__"] = private
        return state

    # ------------------------------------------------------------------------
    # models
    # ------------------------------------------------------------------------
    @property
    def model_path(self) -> Path:
        """Get the SBML file of the default variant of the model.

        Returns:
            The path of the first variant.

        Raises:
            ValueError: if the project has no model variants.
        """
        return self.variant_path()

    def variant_path(self, variant: str | None = None) -> Path:
        """Get the SBML file of a variant of the model.

        Args:
            variant: the name of the variant, the first variant without.

        Returns:
            The path of the SBML file.

        Raises:
            ValueError: if the project has no model variants.
            KeyError: naming the variants, for an unknown variant.
        """
        if not self.variants:
            raise ValueError(f"The project '{self.name}' has no model variants.")
        if variant is None:
            return next(iter(self.variants.values()))
        return _named("model variant", self.variants, variant)

    def load_roadrunner(self, variant: str | None = None) -> roadrunner.RoadRunner:
        """Load a variant of the model in roadrunner at the level of the project.

        The model is compiled with LLJIT at `SimulatorSettings.jit_optimization`
        (`sbmlsim.model.model_roadrunner.load_roadrunner`), as the experiments
        of the project load it, and not with roadrunner's default MCJIT, which
        compiles a model with large initial assignments much slower: the
        loader of the tests of a model.

        Args:
            variant: the name of the variant, the first variant without.

        Returns:
            The roadrunner instance.

        Raises:
            ValueError: if the project has no model variants.
            KeyError: for an unknown variant.
        """
        from sbmlsim.model import model_roadrunner

        return model_roadrunner.load_roadrunner(
            self.variant_path(variant), self.simulator.jit_optimization
        )

    def load_factory(self) -> Callable[[Path], Any]:
        """Get the function which builds the models.

        Raises:
            ValueError: if the project has no factory.
            TypeError: if the factory is not callable.
        """
        if self.factory is None:
            raise ValueError(f"The project '{self.name}' has no factory.")
        factory = self._resolve(self.factory)
        if not callable(factory):
            raise TypeError(
                f"The factory of '{self.name}' ({self.factory}) is a "
                f"{type(factory).__name__}, not a function."
            )
        return factory

    def create_models(self, out: Path | None = None) -> Path:
        """Build the models with the factory, in place or into a rebuild.

        By default the factory builds the models in place, into `factory_dir`.
        A rebuild into another directory `out` keeps the layout of the
        package: the factory writes into `out / factory_dir.relative_to(root)`,
        so an output `o` of `factory_outputs` is `out / o` there and `root / o`
        in the package, see `factory_paths` and `changed_outputs`.

        Args:
            out: the root of a rebuild, the package (`root`) without; a
                relative directory is relative to the current directory.

        Returns:
            The directory the factory wrote into.

        Raises:
            ValueError: if the project has no factory or no directory of it.
        """
        factory = self.load_factory()
        if self.factory_dir is None:
            raise ValueError(
                f"The project '{self.name}' has neither a directory of its factory "
                "nor model variants: give factory_dir."
            )
        directory = (
            self.factory_dir
            if out is None
            else Path(out) / self.factory_dir.relative_to(self.root)
        )
        factory(directory)
        return directory

    def factory_paths(self, out: Path | None = None) -> list[Path]:
        """Get the files the factory writes, `factory_outputs`.

        Args:
            out: the root of a rebuild with the layout of the package, see
                `create_models`; the package (`root`) without.

        Returns:
            The paths of the outputs.
        """
        base = self.root if out is None else Path(out)
        return [base / path for path in self.factory_outputs]

    def compared_outputs(self, out: Path) -> list[Path]:
        """Get the outputs of the factory which a rebuild compares with the package.

        These are the `factory_outputs`. A project which declares none
        compares what the rebuild wrote: the variants of the model and every
        file the factory wrote into its directory which the package has as
        well (a file the package does not have, e.g. an ignored one, is left
        out), so a rebuild is never compared on nothing.

        Args:
            out: the root of a rebuild, see `create_models`.

        Returns:
            The outputs relative to the root, the declared ones in their
            order, else sorted.
        """
        if self.factory_outputs:
            return list(self.factory_outputs)
        outputs = {path.relative_to(self.root) for path in self.variants.values()}
        if self.factory_dir is not None:
            relative = self.factory_dir.relative_to(self.root)
            rebuilt = Path(out) / relative
            if rebuilt.is_dir():
                outputs |= {
                    path.relative_to(out)
                    for path in rebuilt.rglob("*")
                    if path.is_file() and (self.root / path.relative_to(out)).is_file()
                }
        return sorted(outputs)

    def changed_outputs(self, out: Path) -> list[Path]:
        """Get the outputs of the factory whose rebuild differs from the package.

        Every output of `compared_outputs` is compared with the one of the
        package, the SBML of a model without the dates of its model history
        (`same_sbml`), its antimony (`.ant`) without the dates of its history
        (`same_antimony`), any other file by its bytes; an output which only one
        of them has differs.

        Args:
            out: the root of a rebuild, see `create_models`.

        Returns:
            The outputs which differ, relative to the root, in the order of
            `compared_outputs`.
        """
        changed: list[Path] = []
        for output in self.compared_outputs(out):
            rebuilt, committed = Path(out) / output, self.root / output
            if not (rebuilt.is_file() and committed.is_file()) or not _same_output(
                rebuilt, committed
            ):
                changed.append(output)
        return changed

    def load_credibility(self) -> Any:
        """Get the declaration of the contexts of use of the model.

        A declaration of `sbmlsim.credibility` is checked against the
        credibility settings of the project: its domains against the fields of
        the metadata (`domain_fields`), the settings of the extensions of its
        contexts and the pairings of its ratio quantities against the evidence
        extensions (`evidence_extensions`,
        `sbmlsim.credibility.Credibility.extension_problems`); any other
        object is returned as it is.

        Raises:
            ValueError: if the project has no declaration, or its credibility
                settings are invalid (see `domain_fields`,
                `evidence_extensions`).
            sbmlsim.credibility.DeclarationError: with the problems of the
                domains, the settings and the pairings of a declaration.
        """
        if self.credibility is None:
            raise ValueError(
                f"The project '{self.name}' has no credibility declaration."
            )
        credibility = self._resolve(self.credibility)
        if _instance_of(credibility, "sbmlsim.credibility.declaration", "Credibility"):
            from sbmlsim.credibility.declaration import DeclarationError

            problems = [
                *credibility.domain_problems(self.domain_fields()),
                *credibility.extension_problems(self.evidence_extensions()),
            ]
            if problems:
                raise DeclarationError(problems)
        return credibility

    def metadata_type(self) -> type:
        """Get the metadata of the fit mappings of the project.

        Returns:
            The class of `CredibilitySettings.metadata_type`,
            `sbmlsim.fit.objects.MappingMetaData` without.

        Raises:
            ValueError: if a reference is no class.
        """
        value = self.credibility_settings.metadata_type
        if value is None:
            from sbmlsim.fit.objects import MappingMetaData

            return MappingMetaData
        metadata = self._resolve(value)
        if not isinstance(metadata, type):
            raise ValueError(
                f"The metadata type '{value}' of the project '{self.name}' is no class."
            )
        return metadata

    def domain_fields(self) -> dict[str, type[StrEnum]]:
        """Get the fields of the metadata a domain can claim with their enum.

        Returns:
            The enum of every field, see `sbmlsim.credibility.metadata_fields`.
        """
        from sbmlsim.credibility.declaration import metadata_fields

        return metadata_fields(
            self.metadata_type(), self.credibility_settings.domain_fields
        )

    def evidence_extensions(self) -> list[Any]:
        """Get the evidence extensions of the credibility of the project.

        Returns:
            The extensions, the references resolved, in their order.

        Raises:
            ValueError: if an extension is no
                `sbmlsim.credibility.extensions.EvidenceExtension` or two have
                one name.
        """
        from sbmlsim.credibility.extensions import EvidenceExtension

        extensions = [self._resolve(e) for e in self.credibility_settings.extensions]
        wrong = [e for e in extensions if not isinstance(e, EvidenceExtension)]
        if wrong:
            raise ValueError(
                f"The evidence extensions {wrong!r} of the project '{self.name}' "
                "are no sbmlsim.credibility.extensions.EvidenceExtension."
            )
        names = [e.name for e in extensions]
        repeated = sorted({n for n in names if names.count(n) > 1 or not n})
        if repeated:
            raise ValueError(
                f"The evidence extensions of the project '{self.name}' need a name "
                f"of their own, not {repeated}."
            )
        return extensions

    def battery_hook(self) -> BatteryHook | None:
        """Get the battery hook of the credibility of the project.

        Returns:
            The hook, the reference resolved, `None` without one, see
            `sbmlsim.credibility.battery.BatteryHook`.

        Raises:
            ValueError: if the hook is no
                `sbmlsim.credibility.battery.BatteryHook`.
        """
        if self.credibility_settings.battery is None:
            return None
        from sbmlsim.credibility import battery

        hook = self._resolve(self.credibility_settings.battery)
        if not isinstance(hook, battery.BatteryHook):
            raise ValueError(
                f"The battery hook {hook!r} of the project '{self.name}' is no "
                "sbmlsim.credibility.battery.BatteryHook."
            )
        return hook

    def release_hook(self) -> ReleaseHook:
        """Get the release hook of the project, see `ReleaseSettings.hook`.

        Returns:
            The hook, the reference resolved; the hook which adds nothing,
            `sbmlsim.release.hooks.ReleaseHook`, without one.

        Raises:
            ValueError: if the hook is no `sbmlsim.release.hooks.ReleaseHook`.
        """
        from sbmlsim.release.hooks import ReleaseHook

        if self.release is None or self.release.hook is None:
            return ReleaseHook()
        hook = self._resolve(self.release.hook)
        if not isinstance(hook, ReleaseHook):
            raise ValueError(
                f"The release hook {hook!r} of the project '{self.name}' is no "
                "sbmlsim.release.hooks.ReleaseHook."
            )
        return hook

    def action(self, name: str) -> ProjectAction:
        """Get an action of the project.

        Args:
            name: the name of the action.

        Returns:
            The action.

        Raises:
            KeyError: naming the actions, for an unknown action.
        """
        return _named("action", self.actions, name)

    def load_action(self, name: str) -> ActionFunction:
        """Get the function of an action of the project, see `ActionFunction`.

        Args:
            name: the name of the action.

        Returns:
            The function.

        Raises:
            KeyError: naming the actions, for an unknown action.
            TypeError: if the function is not callable.
        """
        action = self.action(name)
        function = self._resolve(action.function)
        if not callable(function):
            raise TypeError(
                f"The action '{name}' of '{self.name}' ({action.function}) is a "
                f"{type(function).__name__}, not a function."
            )
        return function

    # ------------------------------------------------------------------------
    # experiments
    # ------------------------------------------------------------------------
    def load_experiments(self) -> dict[str, list[type[SimulationExperiment]]]:
        """Get the groups of the simulation experiments by their name.

        The group `ALL_EXPERIMENTS` (`all`) has every experiment of the
        project: the group of that name, else every experiment of the groups
        once in their order; experiments which are not grouped are the group
        `all` alone.

        Returns:
            The experiment classes by group.

        Raises:
            ValueError: if the project has no experiments, or two experiment
                classes have one name.
            TypeError: if an experiment is no `SimulationExperiment` class.
        """
        from sbmlsim.experiment import SimulationExperiment

        if self.experiments is None:
            raise ValueError(
                f"The project '{self.name}' has no simulation experiments."
            )
        experiments = self._resolve(self.experiments)
        if callable(experiments) and not isinstance(experiments, (Mapping, type)):
            experiments = experiments()
        if isinstance(experiments, Mapping):
            groups = {name: list(group) for name, group in experiments.items()}
        elif isinstance(experiments, Sequence) and not isinstance(experiments, str):
            groups = {ALL_EXPERIMENTS: list(experiments)}
        else:
            raise TypeError(
                f"The experiments of '{self.name}' are a "
                f"{type(experiments).__name__}, not experiment classes or groups "
                "of them."
            )
        for group in groups.values():
            for cls in group:
                if not (
                    isinstance(cls, type) and issubclass(cls, SimulationExperiment)
                ):
                    raise TypeError(
                        f"The experiments of '{self.name}' have {cls!r}, which is "
                        "no SimulationExperiment class."
                    )
        named: dict[str, type[SimulationExperiment]] = {}
        for group in groups.values():
            for cls in group:
                other = named.setdefault(cls.__name__, cls)
                if other is not cls:
                    raise ValueError(
                        f"The experiments of '{self.name}' have two classes named "
                        f"'{cls.__name__}', {other.__module__}.{other.__qualname__} "
                        f"and {cls.__module__}.{cls.__qualname__}: an experiment is "
                        "named by its class."
                    )
        if ALL_EXPERIMENTS not in groups:
            every = {cls.__name__: cls for group in groups.values() for cls in group}
            groups[ALL_EXPERIMENTS] = list(every.values())
        return groups

    def experiment_classes(
        self, names: Sequence[str] | None = None
    ) -> list[type[SimulationExperiment]]:
        """Get experiment classes by the names of their groups or classes.

        Args:
            names: names of groups or of experiment classes, every experiment
                of the project (the group `all`) without.

        Returns:
            The experiment classes, every class once in the order of the names.

        Raises:
            ValueError: if the project has no experiments, or two experiment
                classes have one name.
            KeyError: naming the groups and the experiments, for a name which
                is neither.
        """
        groups = self.load_experiments()
        if names is None:
            return list(groups[ALL_EXPERIMENTS])
        experiments = {cls.__name__: cls for cls in groups[ALL_EXPERIMENTS]}
        for group in groups.values():
            experiments.update({cls.__name__: cls for cls in group})
        classes: dict[str, type[SimulationExperiment]] = {}
        unknown: list[str] = []
        for name in names:
            if name in groups:
                classes.update({cls.__name__: cls for cls in groups[name]})
            elif name in experiments:
                classes[name] = experiments[name]
            else:
                unknown.append(name)
        if unknown:
            raise KeyError(
                f"The project '{self.name}' has no group or experiment {unknown}, "
                f"it has the groups {sorted(groups)} and the experiments "
                f"{sorted(experiments)}."
            )
        return list(classes.values())

    def _classes(
        self,
        experiments: Sequence[str]
        | Sequence[type[SimulationExperiment]]
        | type[SimulationExperiment]
        | None,
    ) -> list[type[SimulationExperiment]]:
        """Get the classes of the experiments a run names, see `run_experiments`.

        Raises:
            TypeError: for experiments which are neither all names nor all
                experiment classes.
        """
        from sbmlsim.experiment import SimulationExperiment

        if experiments is None:
            return self.experiment_classes()
        if isinstance(experiments, type) and issubclass(
            experiments, SimulationExperiment
        ):
            return [experiments]
        names = [e for e in experiments if isinstance(e, str)]
        classes = [
            e
            for e in experiments
            if isinstance(e, type) and issubclass(e, SimulationExperiment)
        ]
        if len(names) == len(experiments):
            return self.experiment_classes(names)
        if len(classes) == len(experiments):
            return classes
        raise TypeError(
            "The experiments of a run are names of groups or experiments or "
            f"experiment classes, not {list(experiments)!r}."
        )

    def experiment_runner(
        self,
        experiment_classes: Iterable[type[SimulationExperiment]],
        *,
        simulator: Simulator | None = None,
        on_error: Literal["raise", "log"] = "log",
    ) -> ExperimentRunner:
        """Initialize experiments of the project, as its runs do.

        The experiments get the paths of the project (its `base_path`, else its
        `root`, and its `data_path`) and their models are compiled at its level
        (`SimulatorSettings.jit_optimization`).

        Args:
            experiment_classes: the classes of the experiments.
            simulator: the simulator of the tasks, see `ExperimentRunner`.
            on_error: whether an experiment which cannot be initialized
                raises or is logged, see `ExperimentRunner`.

        Returns:
            The runner with the initialized experiments.
        """
        from sbmlsim.experiment import ExperimentRunner

        return ExperimentRunner(
            experiment_classes=list(experiment_classes),
            base_path=self.base_path or self.root,
            data_path=self.data_path,
            simulator=simulator,
            on_error=on_error,
            jit_optimization=self.simulator.jit_optimization,
        )

    def run_experiments(
        self,
        experiments: Sequence[str]
        | Sequence[type[SimulationExperiment]]
        | type[SimulationExperiment]
        | None,
        out: Path | str,
        *,
        n_workers: int | None = None,
        scan_workers: int | None = None,
        simulator: SimulatorSettings | None = None,
        figure_formats: Sequence[str] | None = None,
        save_results: bool = False,
        reduced_selections: bool = True,
        show_report: bool = False,
        progress: bool | None = None,
        raise_on_failure: bool = True,
        uncertainty: UncertaintySettings | None = None,
        collect_figures: bool = False,
    ) -> list[ExperimentResult]:
        """Run simulation experiments of the project and write their report.

        The experiments run with the settings of the simulator of the project,
        their figures with its figure settings and in its formats, see
        `sbmlsim.experiment.runner.run_experiments`. Every experiment runs, also
        after another one failed. The run layer (the kind `simulate`) calls this
        implementation as `Project.run_experiments(project, ...)`, so a subclass
        may override the method with another signature and result without
        breaking it.

        With `uncertainty` the experiments run over draws of the fitted
        parameters of the model: the parameters of the default variant which
        have a distribution are drawn once (`UncertaintyRun.create`), every
        experiment runs every simulation over the draws and draws the curves
        of the simulations as bands, see `sbmlsim.experiment.draws`.

        Args:
            experiments: the experiment classes, or names of groups or classes
                (`experiment_classes`), every experiment without.
            out: the directory of the results and the report; a relative
                directory is relative to the current directory.
            n_workers: the processes of the run, of the pool of the experiments
                and, without `scan_workers`, of the simulator, see
                `run_experiments`.
            scan_workers: the processes of the scans of the simulator, `n_workers`
                without; `n_workers=None, scan_workers=1` runs the experiments
                in a pool and the scans of each serially in its worker, which
                is faster than scans in a pool for many experiments of small
                scans.
            simulator: the settings of the simulator, the ones of the project
                without.
            figure_formats: the formats of the figures, the ones of the project
                (`FigureOutputs.formats`) without or when empty.
            save_results: write the results of the tasks.
            reduced_selections: simulate only the selections an experiment uses.
            show_report: open the report in a web browser.
            progress: show the progress and the timings on the console, see
                `ExperimentRunner.run_experiments`.
            raise_on_failure: raise when an experiment or a figure failed, after
                every experiment ran and the report was written (and the
                figures were collected).
            uncertainty: run the experiments over draws of the fitted
                parameters, `None` runs them once.
            collect_figures: copy the static figures of every experiment
                (`COLLECTED_FORMATS`) into `FIGURES_DIR` of `out`.

        Returns:
            The results of the experiments.

        Raises:
            ValueError: if the project has no experiments, or with
                `uncertainty` no parameter of its default variant has a
                distribution.
            KeyError: for an unknown name of an experiment.
            TypeError: if `figure_formats` is a single string.
            ExperimentRunError: with `raise_on_failure`, if an experiment or a
                figure failed; its `failures` name them.
        """
        import matplotlib

        from sbmlsim import display
        from sbmlsim.experiment import ExperimentRunError
        from sbmlsim.experiment.draws import UncertaintyRun
        from sbmlsim.experiment.runner import run_experiments

        classes = self._classes(experiments)
        formats = normalize_formats(figure_formats or ()) or self.figures.formats
        settings = simulator or self.simulator
        run: UncertaintyRun | None = None
        if uncertainty is not None:
            run = UncertaintyRun.create(self, uncertainty)
            low, high = uncertainty.quantiles
            display.key_values(
                {
                    "uncertainty": f"{uncertainty.draws} draws of "
                    f"{len(run.dimension.values)} parameters, seed "
                    f"{uncertainty.seed}, bands {low:g} to {high:g}"
                }
            )
        output_path = Path(out)
        rc: dict[Any, Any] = self.figures.rc_params()
        with matplotlib.rc_context(rc):
            results = run_experiments(
                classes,
                output_path=output_path,
                base_path=self.base_path or self.root,
                data_path=self.data_path,
                show_report=show_report,
                raise_on_failure=False,
                figure_settings=self.figure_settings,
                n_workers=n_workers,
                progress=progress,
                simulator=settings.simulator(
                    n_workers if scan_workers is None else scan_workers
                ),
                figure_formats=list(formats),
                save_results=save_results,
                reduced_selections=reduced_selections,
                jit_optimization=settings.jit_optimization,
                uncertainty=run,
            )
        if collect_figures:
            collect_figure_files(output_path)
        if raise_on_failure and any(result.failed for result in results):
            raise ExperimentRunError(results)
        return results

    # ------------------------------------------------------------------------
    # fits
    # ------------------------------------------------------------------------
    def load_fit_definitions(self) -> dict[str, FitDefinition]:
        """Get the fit problems by their subset, at the level of the project.

        A definition without an optimization level gets the one of the
        simulator of the project (`SimulatorSettings.jit_optimization`), so its
        problems compile their models at it in every process.

        Returns:
            The definitions by subset, empty for a project without fits.

        Raises:
            TypeError: if the fits are no fit definitions by subset.
            ValueError: for a subset named `ALL_SUBSETS`, which names every
                adopted fit of a report.
        """
        fits = self._resolve(self.fits)
        if not isinstance(fits, Mapping) or not all(
            isinstance(key, str)
            and _instance_of(value, "sbmlsim.fit.cli", "FitDefinition")
            for key, value in fits.items()
        ):
            raise TypeError(
                f"The fits of '{self.name}' ({self.fits}) are no fit definitions "
                "by subset."
            )
        if ALL_SUBSETS in fits:
            raise ValueError(
                f"The project '{self.name}' has a fit subset '{ALL_SUBSETS}', the "
                "name of every adopted fit of a report: rename the subset."
            )
        level = self.simulator.jit_optimization
        return {
            subset: definition
            if level is None or definition.jit_optimization is not None
            else dataclasses.replace(definition, jit_optimization=level)
            for subset, definition in fits.items()
        }

    def fit(self, subset: str) -> FitDefinition:
        """Get a fit problem.

        Args:
            subset: the subset of the fit.

        Returns:
            The definition of the fit, see `load_fit_definitions`.

        Raises:
            KeyError: naming the subsets, for an unknown subset.
            TypeError: if the fits are no fit definitions by subset.
            ValueError: for a subset named `ALL_SUBSETS`.
        """
        return _named("fit subset", self.load_fit_definitions(), subset)

    @property
    def adopted_file(self) -> Path | None:
        """The file of the adopted fits: `adopted`, else `ADOPTED_FILE` in `fits_dir`."""
        if self.adopted is not None:
            return self.adopted
        return None if self.fits_dir is None else self.fits_dir / ADOPTED_FILE

    def adopted_fits(self) -> dict[str, str]:
        """Get the adopted fits.

        Returns:
            The fit id by subset; empty without the file of the adopted fits.

        Raises:
            ValueError: if the file is no file of adopted fits.
        """
        path = self.adopted_file
        if path is None or not path.exists():
            return {}
        content = json.loads(path.read_text(encoding="utf-8"))
        fits = content.get("fits") if isinstance(content, dict) else None
        if not isinstance(fits, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in fits.items()
        ):
            raise ValueError(
                f"'{path}' is no file of adopted fits, "
                '{"fits": {"<subset>": "<fit id>"}}.'
            )
        return dict(fits)

    def adopted_record(self, fit_id: str) -> Path:
        """Get the directory of the records of an adopted fit.

        Args:
            fit_id: the id of the fit, e.g. `PK_20261016_185018__3efe3`.

        Returns:
            The directory `fits_dir/<fit id>`.

        Raises:
            ValueError: if the id is not the id of a fit
                (`sbmlsim.fit.result.fit_subset`), its subset is no fit subset
                of the project or the project has no `fits_dir`.
            FileNotFoundError: if the record of the fit has no parameters.
        """
        from sbmlsim.fit.result import fit_subset
        from sbmlsim.run.fit_kinds import PARAMETERS_FILE

        subset = fit_subset(fit_id)
        definitions = self.load_fit_definitions()
        if subset not in definitions:
            raise ValueError(
                f"The subset '{subset}' of the fit '{fit_id}' is no fit subset of "
                f"the project '{self.name}', which has {sorted(definitions)}."
            )
        if self.fits_dir is None:
            raise ValueError(
                f"The project '{self.name}' has no directory of fit records, fits_dir."
            )
        record = self.fits_dir / fit_id
        parameters = record / PARAMETERS_FILE
        if not parameters.exists():
            raise FileNotFoundError(
                f"The record of the adopted fit '{fit_id}' has no {parameters}."
            )
        return record

    def fit_records(self) -> AdoptedRecords:
        """Get the records of the adopted fits: the fitted values and their uncertainty.

        A model file of the factory takes the value of a fitted parameter
        with its uncertainty from them, `fit_records().parameter(pid,
        unit=...)`, see `sbmlsim.fit.records.AdoptedRecords`.

        Returns:
            The records of the adopted fits of `adopted_fits`, in `fits_dir`;
            none for a project without adopted fits.

        Raises:
            ValueError: if the project has adopted fits and no `fits_dir`, or
                the file of the adopted fits is no such file.
            FileNotFoundError: if the record of an adopted fit does not exist.
        """
        from sbmlsim.fit.records import AdoptedRecords

        fit_ids = list(self.adopted_fits().values())
        if self.fits_dir is None:
            if fit_ids:
                raise ValueError(
                    f"The project '{self.name}' has adopted fits {fit_ids} and no "
                    "directory of fit records, fits_dir."
                )
            return AdoptedRecords(self.root, [])
        return AdoptedRecords(self.fits_dir, fit_ids)

    # ------------------------------------------------------------------------
    # analyses
    # ------------------------------------------------------------------------
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

    def sensitivity(self, name: str) -> Any:
        """Get a sensitivity analysis over the groups of the simulations, resolved.

        Args:
            name: the name of the analysis, a key of `sensitivities`.

        Returns:
            The `SensitivityRunner`, or the runners by subfolder, bound to the
            project (`sbmlsim.sensitivity.runner.bind`).

        Raises:
            KeyError: naming the analyses, for an unknown analysis.
            TypeError: for a reference to neither runners nor a runner.
            ValueError: for a runner of another project.
        """
        from sbmlsim.sensitivity.runner import bind

        target = self._resolve(_named("sensitivity analysis", self.sensitivities, name))
        return bind(target, self)

    def analyses(self, experiments: bool = False) -> list[dict[str, str]]:
        """List the analyses of the project.

        Resolves the fits of the project and, with `experiments`, its
        experiments.

        Args:
            experiments: also list the groups of the simulation experiments
                and every experiment, which the kind `simulate` runs by these
                names; this imports the experiments.

        Returns:
            One entry `{"kind", "name", "description"}` per analysis: the
            simulation experiments (`simulate`, `all`; with `experiments`
            every other group and every experiment class), the analyses of
            every fit subset (`fit`, `identifiability`, `fisher`, `report`,
            `petab`), every scan (`scan`) and, for a scan with indices, its
            sensitivity analysis (`sensitivity`), every sensitivity analysis
            over groups (`sensitivity`), every uncertainty analysis
            (`uncertainty`) and every action (`action`) and the factory
            (`factory`, `models`).
        """
        entries: list[dict[str, str]] = []

        def add(kind: str, name: str, description: str) -> None:
            entries.append({"kind": kind, "name": name, "description": description})

        if self.experiments is not None:
            add("simulate", "all", "the simulation experiments of the project")
            if experiments:
                groups = self.load_experiments()
                for group, classes in groups.items():
                    if group != ALL_EXPERIMENTS:
                        names = ", ".join(cls.__name__ for cls in classes)
                        add("simulate", group, f"the group of experiments: {names}")
                for cls in groups[ALL_EXPERIMENTS]:
                    doc = (cls.__doc__ or "").strip().splitlines()
                    add(
                        "simulate",
                        cls.__name__,
                        doc[0] if doc else f"the experiment '{cls.__name__}'",
                    )
        for subset, definition in self.load_fit_definitions().items():
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
        for name, target in self.sensitivities.items():
            add("sensitivity", name, _sensitivity_description(name, target))
        for name, uncertainty in self.uncertainties.items():
            add(
                "uncertainty",
                name,
                uncertainty.description or f"the uncertainty analysis '{name}'",
            )
        for name, action in self.actions.items():
            add("action", name, action.description or f"the action '{name}'")
        if self.factory is not None:
            add("factory", FACTORY_ANALYSIS, "the models built by the factory")
        return entries


def _sensitivity_description(name: str, target: Any) -> str:
    """Describe a sensitivity analysis of a project without resolving a reference."""
    description = getattr(target, "description", "")
    if description:
        return description
    if isinstance(target, Mapping):
        return (
            f"the sensitivity analyses '{name}' of the subfolders {', '.join(target)}"
        )
    analyses = getattr(target, "analyses", None)
    if analyses:
        return f"the sensitivity analyses '{name}': {', '.join(analyses)}"
    return f"the sensitivity analyses '{name}'"


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


def collect_figure_files(output_path: Path, *, show: bool = True) -> Path:
    """Copy the static figures of a run of experiments into one directory.

    Args:
        output_path: the output of the run.
        show: show the link of the directory on the console of the package;
            a command whose stdout carries its result, e.g. the build of a
            release, shows none.

    Returns:
        The directory of the figures, `FIGURES_DIR` of the output, which has
        every figure of `COLLECTED_FORMATS` of the experiments.
    """
    from sbmlsim import display

    figures_dir = output_path / FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in COLLECTED_FORMATS:
        for path in sorted(output_path.glob(f"**/*.{suffix}")):
            if path.parent != figures_dir:
                shutil.copy2(path, figures_dir / path.name)
    if show:
        display.link("figures", figures_dir)
    return figures_dir


def write_adopted(path: Path, fits: Mapping[str, str]) -> None:
    """Write the adopted fits of a project, atomically.

    Args:
        path: the file, see `ADOPTED_FILE`.
        fits: the fit id by subset.
    """
    content = json.dumps({"fits": dict(fits)}, indent=2, sort_keys=True) + "\n"
    write_text_atomic(path, content)
