"""The static bundle of the panels of a model, `panelbundle/1`.

    python -m sbmlsim.serve --model ID=REFERENCE [...] --export DIR

A page whose panels run without a server, e.g. on GitHub Pages, answers the
routes of the service inside the page from a bundle: the bodies the service
answers for the app, the declaration and the study data of every panel, and
for every panel the recipe of its simulation, which an engine in the browser
(sbmlsim-core) runs on the SBML of the bundle. The bundle is written by the
export of `export_bundles` into `DIR/<model id>/`:

    DIR/<model id>/bundle.json
    DIR/<model id>/models/<model key>/model.xml

and is built from what the service has (`build_bundle`), so it carries the very
bodies the routes answer: the registry of the service and its checks
(`sbmlsim.serve.runner.ModelRegistry.of`), the app of `AppResponse.of`, the
declaration of `sbmlsim.panel.panel_payload`, the study data of
`sbmlsim.serve.data.panel_data` in the units of `PanelRunner.units`, and the
units, the factors and the x of a run as the runner resolves them
(`knob_unit`, `knob_factor`, `curve_factor`, `delivered_times`).

The answers are keyed by the query `variant` of a request: the empty key for
a request without one and every variant the app is served in
(`ModelEntry.app_variants`), the default variant alone for a model without an
app, so a reader of the bundle never resolves a variant itself.

Nothing of the semantics of a simulation is left to the reader of the bundle:
the run of a panel names the model it runs on, its times, its curves with the
factor from the unit of the model to the delivered one and its knobs with the
target each sets, the factor into the unit of the target and the parameter of
the model the formulas of the derived targets read. A panel without derived
targets runs on the SBML of its variant, as sbmlsim loads it (`SBMLModel.sbml`,
flattened); a panel with derived targets runs on a model of its own per set of
derived targets and knob parameters (`model_key`), the SBML of the variant
with a constant parameter `panel_knob__<knob id>` per knob a formula reads and
an initial assignment of every derived target to its formula over them, as
MathML written from the parse tree of the formula
(`sbmlsim.simulator.formula.formula_mathml`). A value `v` of a knob sets its
target to `v * factor` and its parameter to `v`, the pre-initialization of a
point is these together with the `preinit` of the panel, and the initial
assignments compute the derived targets from the parameters, as the service
computes them from the numbers of the knobs.

A panel which declares an uncertainty (`sbmlsim.panel.PanelUncertainty`) also
carries its sample: the parameters of the model which have a distrib
distribution and one vector of `max_draws` draws per parameter, drawn as the
service draws them (`uncertainty_dimension`, a Latin hypercube of the seed the
panel declares), in the units of the model. A reader runs the draws it was
asked for, which are the first ones of that sample, and reduces them to the
band itself: a prefix of a Latin hypercube is a sample of its own, so the band
settles as a reader asks for more draws instead of jumping.

The settings of the integrator are the ones of the simulator of the project
in the names of sbmlsim-core, and the absolute tolerance of every state is the
one the project's settings give it in roadrunner
(`RoadrunnerSBMLModel.tolerances`).

A bundle runs on sbmlsim-core whatever engine the project simulates with. Its
recipe is no plan and reaches no executor, so the refusals of
`sbmlsim.simulator.executor_core.check_supported`, which every plan of the
engine passes, do not reach it; none of them is one a declaration of a panel
can hit.

This module imports neither fastapi nor uvicorn: the export runs in an install
without the extra `web`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import libsbml
import numpy as np

import sbmlsim
from sbmlsim.atomic import atomic_directory
from sbmlsim.engine import Engine
from sbmlsim.hashing import canonical_json_sha256
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.panel import Panel, panel_payload
from sbmlsim.serve.data import panel_data
from sbmlsim.serve.payload import AppResponse, PanelDataResponse, SeriesPayload
from sbmlsim.serve.runner import (
    ModelEntry,
    ModelRegistry,
    PanelRunner,
    X,
    curve_factor,
    curve_unit,
    delivered_times,
    knob_factor,
    knob_unit,
    output_times,
    time_factor,
    time_unit,
)
from sbmlsim.simulation.sampling.model_distributions import (
    uncertainty_dimension,
    uncertainty_name,
)
from sbmlsim.simulator.formula import compile_formula, formula_mathml

logger = logging.getLogger(__name__)

__all__ = [
    "BUNDLE_FILE",
    "CONTRACT",
    "DEFAULT_SETTINGS",
    "KNOB_PREFIX",
    "MODELS_DIR",
    "SBML_FILE",
    "Bundle",
    "build_bundle",
    "export_bundles",
    "write_bundle",
]

#: the version of the contract of a bundle, in its `bundle.json`
CONTRACT: Final = "panelbundle/1"
#: the file of a bundle which holds its answers and its models
BUNDLE_FILE: Final = "bundle.json"
#: the directory of the models of a bundle, one directory per model key
MODELS_DIR: Final = "models"
#: the SBML of a model of a bundle, in the directory of its key
SBML_FILE: Final = "model.xml"
#: the prefix of the parameter of a knob which a formula of a derived target
#: reads, see `_with_derived`
KNOB_PREFIX: Final = "panel_knob__"
#: the length of the hash in the key of a model of derived targets
KEY_DIGITS: Final = 12

#: the settings of the integrator of sbmlsim-core which a project does not
#: set, the defaults of `sbmlsim.simulator.Simulator` and of CVODE
DEFAULT_SETTINGS: Final[Mapping[str, Any]] = {
    "relative_tolerance": 1e-10,
    "absolute_tolerance": {"scalar": AbsoluteTolerance().smallest()},
    "maximum_num_steps": 20000,
    "maximum_time_step": 0.0,
    "initial_time_step": 0.0,
    "stiff": True,
    "linear_solver": "auto",
    "jacobian": "auto",
}

#: the settings of the integrator of roadrunner which sbmlsim-core takes, by
#: the same names
_CORE_SETTINGS: Final = frozenset(
    {"maximum_num_steps", "maximum_time_step", "initial_time_step", "stiff"}
)


@dataclass(frozen=True)
class Bundle:
    """The bundle of a model, see the module.

    Attributes:
        model: the id of the model, the directory of the bundle.
        content: the content of `bundle.json`.
        sbml: the SBML of every model of the bundle, by its key.
    """

    model: str
    content: Mapping[str, Any]
    sbml: Mapping[str, str]


@dataclass
class _BundleModel:
    """A model of a bundle while it is built.

    Attributes:
        variant: the variant whose SBML it is.
        sbml: the SBML.
        tolerances: the absolute tolerance of every state by its id.
        selections: the selections of the curves of the panels run on it.
    """

    variant: str
    sbml: str
    tolerances: Mapping[str, float]
    selections: set[str] = field(default_factory=set)


def export_bundles(registry: ModelRegistry, out: Path) -> list[Path]:
    """Write the bundle of every model of a registry, see the module.

    Args:
        registry: the models, see `ModelRegistry.of`.
        out: the directory of the bundles, created if it does not exist.

    Returns:
        The directory of every bundle, `out/<model id>`, in the order of the
        registry.

    Raises:
        ValueError: for a panel the bundle cannot carry, see `build_bundle`.
    """
    out.mkdir(parents=True, exist_ok=True)
    runner = PanelRunner(registry)
    created = datetime.now(UTC)
    written = []
    for model_id in registry.ids():
        bundle = build_bundle(registry, model_id, runner=runner, created=created)
        written.append(write_bundle(bundle, out))
        logger.info("The bundle of the model '%s' is written.", model_id)
    return written


def write_bundle(bundle: Bundle, out: Path) -> Path:
    """Write a bundle into `out/<model id>`, which it replaces as a whole.

    The directory is written atomically, so a bundle is complete or the one
    before it, and an earlier export, the compiled models of sbmlsim-core
    included, is replaced: a compiled model of another SBML would be wrong.

    Args:
        bundle: the bundle.
        out: the directory of the bundles, which must exist.

    Returns:
        The directory of the bundle.

    Raises:
        ValueError: for a number which is not finite, which JSON does not
            carry, and for a model id which is no name of a directory.
    """
    if bundle.model in ("", ".", "..") or Path(bundle.model).name != bundle.model:
        raise ValueError(
            f"The model id '{bundle.model}' is no name of a directory, which its "
            f"bundle is written into."
        )
    try:
        text = json.dumps(bundle.content, indent=1, ensure_ascii=False, allow_nan=False)
    except ValueError as err:
        raise ValueError(
            f"The bundle of the model '{bundle.model}' carries a number which is "
            f"not finite: {err}"
        ) from err
    path = out / bundle.model
    with atomic_directory(path) as directory:
        (directory / BUNDLE_FILE).write_text(f"{text}\n", encoding="utf-8")
        for key, sbml in bundle.sbml.items():
            folder = directory / MODELS_DIR / key
            folder.mkdir(parents=True)
            (folder / SBML_FILE).write_text(sbml, encoding="utf-8")
    return path


def build_bundle(
    registry: ModelRegistry,
    model_id: str,
    *,
    runner: PanelRunner | None = None,
    created: datetime | None = None,
    stamped: bool = True,
) -> Bundle:
    """Build the bundle of a model of a registry, see the module.

    Args:
        registry: the models, see `ModelRegistry.of`.
        model_id: the id of the model.
        runner: the runner whose warm models resolve the units, a new one of
            the registry without.
        created: the time of the export, now without.
        stamped: whether the bundle records the time of its export
            (`created`); a bundle without is the same for the same models,
            versions and declarations, as the web app of a model is
            (`sbmlsim.webapp`).

    Returns:
        The bundle.

    Raises:
        UnknownModel: for a model which is not in the registry.
        ValueError: naming the panel, for a unit which is no factor of the
            unit of the model; naming the target, for a derived target which
            is no parameter or compartment of the model or which a rule
            sets.
    """
    runner = PanelRunner(registry) if runner is None else runner
    entry = registry.entry(model_id)
    models: dict[str, _BundleModel] = {}
    answers: dict[str, Any] = {}
    queries: list[str | None] = [None]
    queries += [name for name in entry.app_variants() if name is not None]
    for query in queries:
        declared = entry.project.app
        app = (
            None
            if declared is None
            else AppResponse.of(
                declared,
                entry.shared(query),
                entry.pages(query),
                entry.variant_name(query) or "",
            ).model_dump(mode="json")
        )
        panels = {
            panel.id: {
                "declaration": panel_payload(panel),
                "data": _data(runner, entry, panel, query),
                "run": _run(runner, entry, panel, query, models),
            }
            for panel in entry.project.panels_of(query).values()
        }
        answers[query or ""] = {"app": app, "panels": panels}
    settings = _settings(entry)
    content = {
        "contract": CONTRACT,
        "model": entry.id,
        "title": entry.title,
        "sbmlsim": sbmlsim.__version__,
        "answers": answers,
        "models": {
            key: {
                "sbml": f"{MODELS_DIR}/{key}/{SBML_FILE}",
                "compiled": f"{MODELS_DIR}/{key}",
                "variant": model.variant,
                "selections": sorted(model.selections),
                "settings": settings,
                "absolute_tolerances": dict(model.tolerances),
            }
            for key, model in models.items()
        },
    }
    if stamped:
        content["created"] = (created or datetime.now(UTC)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return Bundle(
        model=entry.id,
        content=content,
        sbml={key: model.sbml for key, model in models.items()},
    )


def _data(
    runner: PanelRunner, entry: ModelEntry, panel: Panel, query: str | None
) -> dict[str, Any]:
    """Get the body of the study data of a panel, as the route answers it.

    Args:
        runner: the runner, whose models resolve the units.
        entry: the model.
        panel: the declaration of the panel in the variant of the query.
        query: the variant of the query, `None` for none.

    Returns:
        The body of `GET .../panels/{panel}/data`.
    """
    if not panel.data:
        return PanelDataResponse(series=[]).model_dump(mode="json")
    units = runner.units(entry, panel, query)
    series = panel_data(entry, panel, units)
    return PanelDataResponse(
        series=[SeriesPayload.of(one) for one in series]
    ).model_dump(mode="json")


def _run(
    runner: PanelRunner,
    entry: ModelEntry,
    panel: Panel,
    query: str | None,
    models: dict[str, _BundleModel],
) -> dict[str, Any]:
    """Get the recipe of the simulation of a panel, and add its model.

    Args:
        runner: the runner, whose models resolve the units.
        entry: the model.
        panel: the declaration of the panel in the variant of the query.
        query: the variant of the query, `None` for none.
        models: the models of the bundle by their key, which the model of the
            run is added to.

    Returns:
        The run, see the module.
    """
    project = entry.project
    # the variant which runs, as the service names it in its provenance
    variant = query or panel.simulation.variant or next(iter(project.variants))
    path = project.variant_path(variant).resolve()
    model = runner.model(entry, variant)
    parameters, key = _knob_parameters(model, panel, variant)
    if key not in models:
        models[key] = _BundleModel(
            variant=variant,
            sbml=_with_derived(model, panel, parameters),
            tolerances=_tolerances(entry, model, path),
        )
    models[key].selections.update(curve.id for curve in panel.curves)
    x, x_unit = delivered_times(
        model,
        panel,
        model.Q(output_times(panel), time_unit(model) or "dimensionless"),
    )
    # the checks of the units and the limits of the panel a run makes
    runner.units(entry, panel, query)
    units: dict[str, str | None] = {X: x_unit}
    curves: dict[str, Any] = {}
    for curve in panel.curves:
        try:
            factor = curve_factor(model, curve)
        except ValueError as err:
            raise ValueError(f"The panel '{panel.id}' is no bundle: {err}") from err
        curves[curve.id] = {"selection": curve.id, "factor": factor}
        units[curve.id] = curve_unit(model, curve).text or None
    knob_of = {knob.id: knob for knob in panel.knobs}
    knobs: dict[str, Any] = {}
    for knob in panel.knobs:
        try:
            factor = None if knob.target is None else knob_factor(model, knob)
        except ValueError as err:
            raise ValueError(f"The panel '{panel.id}' is no bundle: {err}") from err
        knobs[knob.id] = {
            "target": knob.target,
            "factor": factor,
            "parameter": parameters.get(knob.id),
            "unit": knob_unit(model, knob),
        }
    return {
        "variant": variant,
        "model": key,
        "model_sha256": runner.sha256(path),
        "start": 0.0,
        "end": float(panel.simulation.end),
        "points": panel.simulation.points,
        "x": list(x),
        "units": units,
        "curves": curves,
        "preinit": [
            [target, float(value)]
            for target, value in sorted(panel.simulation.preinit.items())
        ],
        "knobs": knobs,
        "dosing": [
            {
                "target": dosing.target,
                "dose": dosing.dose,
                "interval": dosing.interval,
                "interval_factor": time_factor(model, knob_of[dosing.interval]),
                "duration": dosing.duration,
                "duration_factor": (
                    None
                    if dosing.duration is None
                    else time_factor(model, knob_of[dosing.duration])
                ),
            }
            for dosing in panel.simulation.dosing
        ],
        "uncertainty": _uncertainty(panel, path),
    }


def _uncertainty(panel: Panel, path: Path) -> dict[str, Any] | None:
    """Get the sample of the uncertainty of a panel, the draws of its band.

    The draws are the Latin hypercube of the distrib distributions of the
    parameters of the model, independent as those distributions are, and in
    the units of the model, so a reader sets a draw exactly as it sets the
    pre-initialization of a point. They are `max_draws` of them, the most a
    request may ask for, and a request for fewer runs the first ones.

    Args:
        panel: the declaration of the panel.
        path: the SBML of the variant the panel runs on.

    Returns:
        The sample, `None` for a panel which declares no uncertainty.

    Raises:
        ValueError: naming the panel, for a panel which declares one and a
            model whose parameters carry no distribution.
    """
    declared = panel.simulation.uncertainty
    if declared is None:
        return None
    try:
        dimension = uncertainty_dimension(
            path, draws=declared.max_draws, seed=declared.seed
        )
    except ValueError as err:
        raise ValueError(f"The panel '{panel.id}' is no bundle: {err}") from err
    parameters = sorted(dimension.values)
    columns = [
        np.asarray(
            getattr(dimension.values[name], "magnitude", dimension.values[name]),
            dtype=float,
        )
        for name in parameters
    ]
    return {
        "fit": uncertainty_name(path),
        "level": declared.level,
        "seed": declared.seed,
        "default_draws": declared.draws,
        "max_draws": declared.max_draws,
        "parameters": parameters,
        "draws": [[float(value) for value in row] for row in np.column_stack(columns)],
    }


def _knob_parameters(
    model: SBMLModel, panel: Panel, variant: str
) -> tuple[dict[str, str], str]:
    """Get the parameters of the knobs a panel's formulas read, and its model key.

    Args:
        model: the model of the variant.
        panel: the panel.
        variant: the name of the variant.

    Returns:
        The parameter of every knob a formula of a derived target reads, by
        the id of the knob, and the key of the model of the panel: the name of
        the variant for a panel without derived targets, else the name with
        the first `KEY_DIGITS` of the sha256 of its derived targets and the
        knobs they read with their defaults.
    """
    derived = panel.simulation.derived
    if not derived:
        return {}, variant
    read = sorted({s for f in derived.values() for s in compile_formula(f).symbols})
    defaults = {knob.id: knob.default for knob in panel.knobs}
    digest = canonical_json_sha256(
        {
            "derived": sorted([target, formula] for target, formula in derived.items()),
            "knobs": {knob: float(defaults[knob]) for knob in read},
        }
    )
    doc = _document(model.sbml)
    taken: set[str] = set()
    parameters: dict[str, str] = {}
    for knob in read:
        name = f"{KNOB_PREFIX}{knob}"
        suffix = 1
        while doc.getModel().getElementBySId(name) is not None or name in taken:
            suffix += 1
            name = f"{KNOB_PREFIX}{knob}_{suffix}"
        taken.add(name)
        parameters[knob] = name
    return parameters, f"{variant}-{digest[:KEY_DIGITS]}"


def _document(sbml: str) -> libsbml.SBMLDocument:
    """Read an SBML document.

    Args:
        sbml: the SBML.

    Returns:
        The document.

    Raises:
        ValueError: for SBML without a model.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    if doc.getModel() is None:
        raise ValueError("The SBML of the model has no model.")
    return doc


def _with_derived(model: SBMLModel, panel: Panel, parameters: Mapping[str, str]) -> str:
    """Get the SBML of the model of a panel, see the module.

    Args:
        model: the model of the variant.
        panel: the panel.
        parameters: the parameter of every knob its formulas read, see
            `_knob_parameters`.

    Returns:
        The SBML of the variant for a panel without derived targets, else that
        SBML with the parameters of the knobs and an initial assignment of
        every derived target to its formula.

    Raises:
        ValueError: naming the panel and the target, for a derived target
            which is no parameter or compartment of the model or which an
            assignment or rate rule sets.
    """
    derived = panel.simulation.derived
    if not derived:
        return model.sbml
    doc = _document(model.sbml)
    sbml_model: libsbml.Model = doc.getModel()
    defaults = {knob.id: knob.default for knob in panel.knobs}
    for knob, name in parameters.items():
        parameter: libsbml.Parameter = sbml_model.createParameter()
        _checked(parameter.setId(name), name)
        _checked(parameter.setValue(float(defaults[knob])), name)
        _checked(parameter.setConstant(True), name)
        if doc.getLevel() > 2:
            _checked(parameter.setUnits("dimensionless"), name)
    for target, formula in derived.items():
        element = sbml_model.getElementBySId(target)
        if not isinstance(element, (libsbml.Parameter, libsbml.Compartment)):
            raise ValueError(
                f"The derived target '{target}' of the panel '{panel.id}' is no "
                f"parameter and no compartment of the model, so no bundle carries it."
            )
        rule = sbml_model.getRuleByVariable(target)
        if rule is not None:
            raise ValueError(
                f"The derived target '{target}' of the panel '{panel.id}' is set by "
                f"a rule of the model, so its initial assignment is not its value."
            )
        math = libsbml.readMathMLFromString(formula_mathml(formula, parameters))
        if math is None:
            raise ValueError(
                f"The formula '{formula}' of the derived target '{target}' of the "
                f"panel '{panel.id}' is no MathML libsbml reads."
            )
        assignment = sbml_model.getInitialAssignmentBySymbol(target)
        if assignment is None:
            assignment = sbml_model.createInitialAssignment()
            _checked(assignment.setSymbol(target), target)
        _checked(assignment.setMath(math), target)
    return libsbml.writeSBMLToString(doc)


def _checked(code: int, what: str) -> None:
    """Check the return code of a setter of libsbml.

    Args:
        code: the return code.
        what: the element the setter sets, which a problem names.

    Raises:
        ValueError: for a code which is no success.
    """
    if code != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ValueError(
            f"libsbml does not set '{what}': {libsbml.OperationReturnValue_toString(code)}"
        )


def _tolerances(entry: ModelEntry, model: SBMLModel, path: Path) -> dict[str, float]:
    """Get the absolute tolerance of every state as roadrunner integrates it.

    Args:
        entry: the model, whose project gives the settings of the simulator.
        model: the warm model of the variant, used when it is one of
            roadrunner.
        path: the SBML file of the variant, loaded in roadrunner otherwise.

    Returns:
        The tolerance of every state by its id, see
        `RoadrunnerSBMLModel.tolerances`.
    """
    if not isinstance(model, RoadrunnerSBMLModel):
        settings = entry.project.simulator
        loaded = load_model(
            path, Engine.ROADRUNNER, jit_optimization=settings.jit_optimization
        )
        loaded.set_integrator_settings(
            **{
                "absolute_tolerance": AbsoluteTolerance(),
                "relative_tolerance": DEFAULT_SETTINGS["relative_tolerance"],
                **settings.integrator_settings(),
            }
        )
        if not isinstance(loaded, RoadrunnerSBMLModel):  # pragma: no cover
            raise TypeError("roadrunner loads a model of roadrunner")
        model = loaded
    frame = model.tolerances()
    return {
        str(sid): float(tolerance)
        for sid, tolerance in zip(
            frame["sid"], frame["absolute_tolerance"], strict=True
        )
    }


def _settings(entry: ModelEntry) -> dict[str, Any]:
    """Get the settings of the integrator of sbmlsim-core of a project.

    Args:
        entry: the model, whose project gives the settings of its simulator.

    Returns:
        The settings, `DEFAULT_SETTINGS` with the tolerances of the project
        and the settings of its integrator which sbmlsim-core takes by the
        same name; a setting it has not is logged and not carried.
    """
    simulator = entry.project.simulator
    settings = dict(DEFAULT_SETTINGS)
    if simulator.relative_tolerance is not None:
        settings["relative_tolerance"] = float(simulator.relative_tolerance)
    tolerance = simulator.absolute_tolerance
    if tolerance is not None:
        settings["absolute_tolerance"] = {
            "scalar": float(AbsoluteTolerance.of(tolerance).smallest())
        }
    for name, value in simulator.integrator.items():
        if name in _CORE_SETTINGS:
            settings[name] = bool(value) if name == "stiff" else value
        else:
            logger.warning(
                "The setting '%s' of the integrator of the model '%s' is no setting "
                "of sbmlsim-core and is not carried by the bundle.",
                name,
                entry.id,
            )
    return settings
