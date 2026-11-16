"""The web assets of the models of a project, written with its factory.

`write_assets` is what `sbmlsim.project.Project.create_models` writes after the
factory, unless the project switches it off (`Project.web_app`): every variant
of the model compiled by sbmlsim-core into `<directory>/wasm/<stem>/`
(`model.wasm` and `model.json`, the layout of `sbmlsim-core compile`, `stem`
the name of the SBML file without its suffix) and the app of the model,
`<directory>/<name>_app.html` (`sbmlsim.webapp.page`). They are written from
the SBML the factory just wrote, so the compiled models and the app are always
the ones of the models next to them, also in a rebuild into another directory.

A model which declares no panel and no app gets the panels of
`Project.default_panels`, by default the panel of `default_panel`, the time
courses of its species, so every model has an app (a framework overrides it
with the panels it knows of its models); `app_project` is
the project of one SBML file, which `python -m sbmlsim.webapp` writes the app
of.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import libsbml

from sbmlsim.panel import MAX_CURVES, Panel, PanelCurve, PanelSimulation
from sbmlsim.project import Project
from sbmlsim.webapp import WASM_DIR
from sbmlsim.webapp.compile import compile_sbml, prepare_sbml, write_compiled
from sbmlsim.webapp.page import write_app

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_END",
    "DEFAULT_PANEL",
    "WASM_DIR",
    "app_project",
    "default_panel",
    "write_assets",
    "write_wasm",
]

#: the id of the default panel of a model without panels
DEFAULT_PANEL: Final = "species"
#: the end of the simulation of the default panel, in the time unit of the model
DEFAULT_END: Final = 100.0


def default_panel(sbml: Path) -> Panel:
    """Get the panel of a model which declares none: the time courses of its states.

    The curves are the species of the model in their order, the concentration
    of a species in a compartment and the amount of one which has only
    substance units; a model without species shows the parameters and
    compartments its rules set, the ones of rate rules first. At most
    `MAX_CURVES` are shown, labelled by their name. The simulation runs from 0
    to `DEFAULT_END` in the time unit of the model. A hierarchical model is
    read flattened.

    Args:
        sbml: the SBML file of the model.

    Returns:
        The panel `DEFAULT_PANEL`.

    Raises:
        ValueError: if the model has neither species nor rules, nothing which
            changes over time.
    """
    text = prepare_sbml(sbml.read_text(encoding="utf-8"), base_path=sbml.parent)
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(text)
    model: libsbml.Model | None = doc.getModel()
    if model is None:
        raise ValueError(f"The model '{sbml}' cannot be read.")
    candidates: list[tuple[str, str]] = []
    for index in range(model.getNumSpecies()):
        species: libsbml.Species = model.getSpecies(index)
        sid = species.getId()
        selection = sid if species.getHasOnlySubstanceUnits() else f"[{sid}]"
        candidates.append((selection, species.getName() or sid))
    kind = "species"
    if not candidates:
        kind = "variables of the rules"
        rules = [model.getRule(i) for i in range(model.getNumRules())]
        for rule in sorted(rules, key=lambda rule: not rule.isRate()):
            element: libsbml.SBase | None = model.getElementBySId(rule.getVariable())
            if isinstance(element, (libsbml.Parameter, libsbml.Compartment)):
                candidates.append(
                    (rule.getVariable(), element.getName() or element.getId())
                )
    if not candidates:
        raise ValueError(
            f"The model '{sbml}' has neither species nor rules to show in a "
            "default panel."
        )
    name = model.getName() or model.getId() or sbml.stem
    shown = (
        f"the {kind}"
        if len(candidates) <= MAX_CURVES
        else f"the first {MAX_CURVES} of {len(candidates)} {kind}"
    )
    return Panel(
        id=DEFAULT_PANEL,
        title=name,
        subtitle=f"{shown} of the model over time",
        curves=tuple(
            PanelCurve(id=sid, label=label) for sid, label in candidates[:MAX_CURVES]
        ),
        simulation=PanelSimulation(end=DEFAULT_END),
    )


def app_project(sbml: Path) -> Project:
    """Get the project of one SBML file, the default panel if it declares none.

    Args:
        sbml: the SBML file.

    Returns:
        The project named after the file, its one variant `model`.
    """
    sbml = sbml.resolve()
    project = Project(name=sbml.stem, root=sbml.parent, variants={"model": sbml})
    return _with_default_panels(project, project)


def _with_default_panels(project: Project, rooted: Project) -> Project:
    """Get the project of the app with the default panels if it declares none.

    Args:
        project: the project, whose `Project.default_panels` gives them.
        rooted: the project of the models the factory wrote, see `_rooted`.
    """
    if not rooted.variants or rooted.app is not None or rooted.panels_of(None):
        return rooted
    return rooted.replace(panels=project.default_panels(rooted.variant_path(None)))


#: the fields of a project which its app reads: the panels and the app, the
#: simulator of the runs and the data under the panels
_APP_FIELDS: Final = (
    "name",
    "title",
    "description",
    "panels",
    "variant_panels",
    "app",
    "simulator",
    "data_path",
)


def _rooted(project: Project, directory: Path) -> Project:
    """Get the project of the models the factory wrote into a directory.

    A build in place (`directory` is the directory of the factory) is the
    project as it is. A rebuild writes the layout of the package into another
    root (`Project.create_models`), so the app is written of a project of the
    rebuild: a `Project` of the fields the app reads (`_APP_FIELDS`, the data
    under the panels stays in the package), its root the ancestor of
    `directory` at the depth of the directory of the factory in the package
    and a variant in the directory of the factory the file the factory wrote
    into `directory`. A subclass of `Project` (e.g. the model of pkpdsim) may
    bind more of its paths to its root, which a rebuild does not move, so the
    project of the rebuild is a plain `Project`.
    """
    if project.factory_dir is None:
        return project
    factory_dir = project.factory_dir.resolve()
    directory = directory.resolve()
    if directory == factory_dir:
        return project
    root = directory
    for _ in factory_dir.relative_to(project.root.resolve()).parts:
        root = root.parent
    variants = {}
    for name, path in project.variants.items():
        resolved = path.resolve()
        variants[name] = (
            directory / resolved.relative_to(factory_dir)
            if resolved.is_relative_to(factory_dir)
            else path
        )
    fields = {field: getattr(project, field) for field in _APP_FIELDS}
    return Project(**fields, root=root, factory_dir=directory, variants=variants)


def write_wasm(project: Project, directory: Path) -> list[Path]:
    """Write every variant of a project compiled by sbmlsim-core, see the module.

    Args:
        project: the project, its variants the SBML files to compile.
        directory: the directory, the compiled models go into `wasm/<stem>/`.

    Returns:
        The directories of the compiled variants, in the order of the variants.

    Raises:
        WebAppError: for a variant sbmlsim-core cannot compile.
    """
    written = []
    for path in project.variants.values():
        compiled = compile_sbml(path.read_text(encoding="utf-8"), base_path=path.parent)
        written.append(write_compiled(compiled, directory / WASM_DIR / path.stem))
    return written


def write_assets(project: Project, directory: Path) -> list[Path]:
    """Write the compiled variants and the app of a project, see the module.

    Args:
        project: the project.
        directory: the directory the factory wrote the models into.

    Returns:
        The written directories of the compiled variants and the app; none
        for a project without model variants.

    Raises:
        ValueError: for an app or a panel the service refuses.
        WebAppError: for a model sbmlsim-core cannot compile.
    """
    if not project.variants:
        logger.info(
            "The project '%s' has no model variants, so it has no web app.",
            project.name,
        )
        return []
    shown = _with_default_panels(project, _rooted(project, directory))
    written = write_wasm(shown, directory)
    written.append(write_app(shown, directory))
    return written
