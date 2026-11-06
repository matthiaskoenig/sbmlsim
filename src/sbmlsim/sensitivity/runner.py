"""The sensitivity analyses of a project over the groups of its simulations.

A `SensitivityRunner` runs the analyses of a model (`LocalAnalysis`,
`SamplingAnalysis`, `FastAnalysis`, `SobolAnalysis`) over the simulations of
their groups (`SensitivityGroup`, each on a variant of the model of the
project, `sbmlsim.project.Project.variants`) and writes the table of the
parameters (`sbmlsim.sensitivity.parameters`) and the tables and figures of
its outputs (`SensitivityOutput`) in the layout of sbmlsim 0.8.5, which
manuscripts read. The designs are the ones of `sbmlsim.simulation.sampling`,
the indices the ones of `sbmlsim.sensitivity`; the runner keeps only what is
the model's own, its parameters, groups, outputs and observables:

```py
SENSITIVITY = SensitivityRunner(
    project=PROJECT,
    parameters=sensitivity_parameters,
    outputs=OUTPUTS,
    observables=OBSERVABLES,
    analyses={
        "local": LocalAnalysis(groups=GROUPS[:1]),
        "sampling": SamplingAnalysis(groups=GROUPS),
    },
    out=Path("results/sensitivity"),
)

if __name__ == "__main__":
    SENSITIVITY.main()
```

A project lists its runners in `Project.sensitivities`, a runner or a mapping
of runners by subfolder (`run_subfolders`, a model with several simulations),
or a reference `"module:attribute"` to either; the kind `sensitivity` of
`python -m sbmlsim.run` runs them into its `--out`, one unit per analysis and
subfolder (`prepare_run`).

The layout of the results below the folder of a run:

| analysis | files |
| --- | --- |
| every run | `parameters.tsv`, `parameters.tex` |
| local | `local/local_d<delta>_nvar<n_var>_<k>_<group>.tsv` and `.png` |
| sampling | `sampling/<group>/samples.tsv`, `sampling/sampling_N<n>.tsv`, `_compact.tsv`, `.tex`, `_sensitivity_samples.png`, `_sensitivity_outputs.png` |
| fast, sobol | `<analysis>/<group>/<output>.S1.tsv` and `.ST.tsv`, `<analysis>/<prefix>_sensitivity_<k>_<group>_S1.png`, `_ST.png` and `_<ko>_<output>.png` |
| subfolders | `<subfolder>/<analysis>/...`, the table once in the folder |
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple, override

import numpy as np
import pandas as pd
from matplotlib import cbook
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from sbmlsim import display
from sbmlsim.console import console
from sbmlsim.sensitivity import indices, plots
from sbmlsim.sensitivity.classification import (
    sensitivity_classification,
    sensitivity_classification_symbol,
    uncertainty_classification,
    uncertainty_classification_symbol,
)
from sbmlsim.sensitivity.parameters import (
    SensitivityParameter,
    build_values,
    design_factors,
    distributions,
    free_parameters,
    parameters_to_df,
    relative_design,
    write_parameters,
)
from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult
from sbmlsim.simulation import PK, Dimension, Observable, Scan, Simulation, sampling

if TYPE_CHECKING:
    from sbmlsim.model.model_sbml import SBMLModel
    from sbmlsim.project import Project
    from sbmlsim.result import ScanResult
    from sbmlsim.simulator import Simulator

__all__ = [
    "CUTOFF_085",
    "SAMPLE_SIZE",
    "SEED",
    "Analysis",
    "CommandLine",
    "FastAnalysis",
    "LocalAnalysis",
    "PreparedRun",
    "SamplingAnalysis",
    "SensitivityGroup",
    "SensitivityOutput",
    "SensitivityRunner",
    "SensitivityUnit",
    "SobolAnalysis",
    "bind",
    "prepare_run",
    "run_subfolders",
    "sample_size",
]

logger = logging.getLogger(__name__)

#: the environment variable which replaces the number of samples of a sampled
#: analysis, for a smoke run of the analyses
SAMPLE_SIZE = "SBMLSIM_SENSITIVITY_N"

#: the seed of the designs of the analyses
SEED = 1234

#: the cutoff of the heatmaps of sbmlsim 0.8.5
CUTOFF_085 = 0.05


def sample_size(n: int) -> int:
    """Get the number of samples of a sampled analysis, `n` unless a smoke run sets another.

    The environment variable `SBMLSIM_SENSITIVITY_N` (a positive int)
    replaces `n`, e.g. `SBMLSIM_SENSITIVITY_N=16` for a quick run of the
    analyses in a verification.

    Args:
        n: the number of samples of the analysis.

    Returns:
        The number of samples.

    Raises:
        ValueError: if the variable is set and is no positive int.
    """
    value = os.environ.get(SAMPLE_SIZE, "")
    if not value:
        return n
    try:
        size = int(value)
    except ValueError:
        size = 0
    if size < 1:
        raise ValueError(
            f"{SAMPLE_SIZE} is a positive number of samples, not '{value}'."
        )
    return size


@dataclass(frozen=True)
class SensitivityOutput:
    """An output of the analyses, an observable of the simulation of a group.

    Attributes:
        uid: the id of the output in the tables and the names of the files:
            the columns `output` and `uid` of the tables of sbmlsim 0.8.5, the
            columns of `samples.tsv`, `<uid>.S1.tsv`, `<uid>.ST.tsv` and the
            figure of its indices, `..._<o>_<uid>.png`.
        name: the name of the output in the titles of the figures, in the
            heatmaps and in the column `name` of the tables of sbmlsim 0.8.5,
            unique among the outputs of a runner.
        observable: the id of the observable in the result of the scan, the uid
            when empty; an id of an output which is no identifier, e.g.
            `[Cve_api]_cmax`, names an observable `Cve_api_cmax`. A parameter
            of a `PK` observable is `<id>.<parameter>`, e.g. `pk.cmax`.
    """

    uid: str
    name: str
    observable: str = ""

    def __post_init__(self) -> None:
        """Let the output be its own observable when it names none."""
        if not self.observable:
            object.__setattr__(self, "observable", self.uid)


@dataclass(frozen=True)
class SensitivityGroup:
    """A group of the analyses, e.g. a degree of renal impairment.

    Attributes:
        uid: the id of the group, the folder of its tables and figures.
        name: the name of the group in the titles of the figures.
        simulation: the simulation of the group, its changes in
            `preinit_changes`; an analysis varies no parameter which it sets,
            see `free_parameters`.
        color: the color of the group in the figures of `sampling` which
            compare the groups, a color of matplotlib; the color of the
            property cycle of matplotlib (`C0`, `C1`, ...) of its position if
            `None`.
        variant: the variant of the model of the project which the group runs
            (`Project.variants`), the default variant (the first) if `None`;
            the design of an analysis is the same for every group, relative to
            the values of its variant, see `design_factors`.
        prepare: a function of the scan of the group over a design and the
            SBML file of its variant (`(scan, model_path) -> scan`) which gives
            the scan that runs, `None` for the scan as it is; e.g. the
            drug-free equilibration of every point of a physiologically based
            model, which replaces a long pre-run from a negative start that
            leaves a slow state off its steady state. The module knows no
            model: what the function does is the model's.
    """

    uid: str
    name: str
    simulation: Simulation
    color: str | None = None
    variant: str | None = None
    prepare: Callable[[Scan, Path], Scan] | None = None


@dataclass
class _Context:
    """What a run hands to its analyses.

    Attributes:
        simulator: the simulator of the run.
        project: the project, whose variants the groups run.
        outputs: the outputs of the runner.
        observables: the observables of the runner.
        seed: the seed of the designs.
        builds: the values of the parameters in the variants of the groups
            other than the default one, which the table has, by variant.
        models: the models loaded by the run, by the SBML file of a variant.
    """

    simulator: Simulator
    project: Project
    outputs: Sequence[SensitivityOutput]
    observables: Sequence[Observable]
    seed: int
    builds: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    models: dict[Path, SBMLModel] = field(default_factory=dict)

    def model_path(self, group: SensitivityGroup) -> Path:
        """Get the SBML file of the variant which a group runs."""
        return self.project.variant_path(group.variant)

    def model(self, group: SensitivityGroup) -> SBMLModel:
        """Get the model of the variant which a group runs, loaded once per run.

        The model is loaded in the engine and at the optimization level of the
        simulator settings of the project.
        """
        from sbmlsim.model import model_sbml

        path = self.model_path(group)
        if path not in self.models:
            settings = self.project.simulator
            self.models[path] = self.simulator.load(
                model_sbml.load_model(
                    path, settings.engine, jit_optimization=settings.jit_optimization
                )
            )
        return self.models[path]

    def design(
        self,
        group: SensitivityGroup,
        design: Dimension,
        varied: Sequence[SensitivityParameter],
    ) -> Dimension:
        """Get the global design of a group, around the values of its variant.

        The design of the table for a group of the default variant, the same
        relative design around the values of the variant of the group otherwise
        (`design_factors`, `relative_design`).
        """
        values = self.builds.get(_variant(self.project, group))
        if values is None:
            return design
        return relative_design(design, design_factors(varied, values))

    def scan(self, group: SensitivityGroup, design: Dimension) -> ScanResult:
        """Run a design over the simulation of a group, the outputs per point.

        The scan of a group with `prepare` is the one the function gives on
        the variant of the group (`SensitivityGroup.prepare`).
        """
        scan = Scan(group.simulation, [design])
        if group.prepare is not None:
            scan = group.prepare(scan, self.model_path(group))
        return self.simulator.run(
            self.model(group),
            scan,
            list(self.observables),
            keep=[o.observable for o in self.outputs],
        )

    def write_indices(
        self, result: SensitivityResult, names: Sequence[str], path: Path
    ) -> None:
        """Write every index of every output as a table, `<output>.<index>.tsv`."""
        for output in self.outputs:
            for index in names:
                column = f"{output.uid}.{index}"
                result.to_dataframe(f"{output.observable}.{index}").rename(
                    columns={f"{output.observable}.{index}": column}
                ).to_csv(path / f"{column}.tsv", sep="\t")

    def unit(self, result: ScanResult, output: SensitivityOutput) -> str:
        """Get the unit of an output: as its observable gives it, else of the result.

        The unit of a `Formula` or `Custom` observable is the one the runner
        was given (`mM*min`), as in the tables of sbmlsim 0.8.5; a `PK`
        observable has the unit of the result (`millimolar * minute`).
        """
        for observable in self.observables:
            if observable.id == output.observable:
                unit = getattr(observable, "unit", None)
                if unit:
                    return str(unit)
        return str(result.units.get(output.observable, ""))


def _variant(project: Project, group: SensitivityGroup) -> str:
    """Get the name of the variant of a group, the default variant without one."""
    if group.variant is not None:
        return str(group.variant)
    return next(iter(project.variants))


def _repeated(values: Iterable[str]) -> list[str]:
    """Get the values which are given more than once, sorted."""
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        (repeated if value in seen else seen).add(value)
    return sorted(repeated)


def _sequence_of_names(names: Sequence[str], what: str) -> tuple[str, ...]:
    """Get names as a tuple, refusing a single string, which is no sequence of names.

    Raises:
        TypeError: for a string, e.g. `run("local")` instead of `run(["local"])`.
    """
    if isinstance(names, str):
        raise TypeError(
            f"The {what} are a sequence of names, not the string {names!r}."
        )
    return tuple(names)


def _check_subfolder(subfolder: str) -> None:
    """Check that a subfolder lies below the folder of the results.

    Raises:
        ValueError: for an empty folder, one with a drive or a root (`/tmp`,
            which windows takes as relative to the drive) or with `..`.
    """
    if not subfolder or Path(subfolder).anchor or ".." in Path(subfolder).parts:
        raise ValueError(
            f"The subfolder {subfolder!r} is a relative folder below the folder "
            "of the results."
        )


def _reaches(result: SensitivityResult, index: str, cutoff: float) -> bool:
    """Check whether an index reaches a cutoff, as `plot_heatmap` keeps its rows.

    A row is kept if a value reaches the cutoff or one is undefined.
    """
    values = np.asarray(result.index(index).values, dtype=float)
    finite = np.isfinite(values)
    magnitude = np.abs(np.where(finite, values, 0.0))
    return bool((~finite).any() or (magnitude >= cutoff).any())


def _heatmap(
    result: SensitivityResult,
    index: str,
    *,
    cutoff: float,
    path: Path,
    **options: Any,
) -> None:
    """Draw a heatmap of `plot_heatmap`, every parameter if none reaches the cutoff.

    `plot_heatmap` refuses an index which no parameter reaches, e.g. the local
    sensitivities of a group whose parameters do not move its outputs; the
    heatmap then shows every parameter, which a warning names.
    """
    if cutoff > 0 and not _reaches(result, index, cutoff):
        logger.warning(
            "No parameter reaches the cutoff %g of '%s' in '%s', the heatmap "
            "shows every parameter.",
            cutoff,
            index,
            path.name,
        )
        cutoff = 0.0
    plots.plot_heatmap(result, index, cutoff=cutoff, path=path, **options)


def _labelled(
    result: SensitivityResult,
    parameters: Sequence[SensitivityParameter],
    outputs: Sequence[SensitivityOutput],
) -> SensitivityResult:
    """Get a result labelled as the heatmaps of sbmlsim 0.8.5.

    The parameters are `uid: name`, the observables the names of the outputs.
    """
    names = {o.observable: o.name for o in outputs}
    renames: dict[str, str] = {}
    for variable in result.ds.data_vars:
        observable, index = str(variable).rsplit(".", 1)
        if observable in names:
            renames[str(variable)] = f"{names[observable]}.{index}"
    labels = {p.uid: f"{p.uid}: {p.name}" for p in parameters}
    ds = result.ds.rename(renames).assign_coords(
        {PARAMETER: [labels.get(p, p) for p in result.parameters]}
    )
    ds.attrs = {
        **ds.attrs,
        "units": {renames.get(k, k): v for k, v in result.units.items()},
    }
    return SensitivityResult(ds)


def _classes(
    value: float,
    classify: Callable[[float], StrEnum | None],
    symbolize: Callable[[float], str],
) -> tuple[str, str]:
    """Get the symbol and the class of a value, empty if it is undefined.

    The classifications of sbmlsim give `None` for `NaN`, the sensitivity of
    an output which is zero at the reference or the uncertainty of one with the
    median zero, and classify a negative uncertainty by its magnitude.
    """
    classification = classify(value)
    if classification is None:
        return "", ""
    return symbolize(value), classification.value


def _figshape(n: int) -> tuple[int, int]:
    """Get the rows and columns of a figure of `n` panels, as sbmlsim 0.8.5 did."""
    if n <= 4:
        return 1, n
    ncols = np.ceil(np.sqrt(n))
    n_empty = ncols * ncols - n
    nrows = ncols - np.floor(n_empty / ncols)
    return int(nrows), int(ncols)


def _boxes(
    ax: Axes, data: Sequence[np.ndarray], colors: Sequence[str], labels: Sequence[str]
) -> None:
    """Draw a box per group at the positions `0, 1, ...`, filled with its color.

    The boxes of `Axes.boxplot(patch_artist=True, showfliers=False)` with
    black lines: the box from the first to the third quartile, the median and
    the whiskers to the last values within 1.5 interquartile ranges, drawn
    without `boxplot`, which reads every parameter of matplotlib and so
    resolves the backend of pyplot (#317). A group without values has no box.
    """
    width = float(np.clip(0.15 * max(len(data) - 1, 1), 0.15, 0.5))
    line: dict[str, Any] = {"color": "black", "linewidth": 1.0, "zorder": 2.0}
    median: dict[str, Any] = {**line, "zorder": 2.1}
    for k, (values, color) in enumerate(zip(data, colors, strict=True)):
        if values.size == 0:
            continue
        (stats,) = cbook.boxplot_stats(values)
        left, right = k - width / 2, k + width / 2
        ax.add_patch(
            Rectangle(
                (left, stats["q1"]),
                width,
                stats["q3"] - stats["q1"],
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=2.0,
            )
        )
        ax.plot([left, right], [stats["med"]] * 2, **median)
        for low, high in (
            (stats["whislo"], stats["q1"]),
            (stats["q3"], stats["whishi"]),
        ):
            ax.plot([k, k], [low, high], **line)
        for y in (stats["whislo"], stats["whishi"]):
            ax.plot([k - width / 4, k + width / 4], [y, y], **line)
    ax.set_xticks(range(len(labels)), labels)


def _plot_groups(
    panels: Sequence[tuple[str, Sequence[np.ndarray]]],
    groups: Sequence[SensitivityGroup],
    path: Path,
    seed: int,
) -> None:
    """Draw the values of every group as box, violin and points, a panel per quantity.

    The figures of the sampling of sbmlsim 0.8.5: a box (without outliers), a
    violin (the mean and the median) and the points of every group side by
    side in the color of the group. A group with fewer than two distinct
    values has no violin, which needs a spread.

    Args:
        panels: the label of the axis and the values of every group.
        groups: the groups.
        path: the file of the figure.
        seed: the seed of the jitter of the points.
    """
    nrows, ncols = _figshape(len(panels))
    # a `Figure`, not a pyplot figure: pyplot binds it to the backend, which
    # aborts the process with TkAgg next to a pool of processes
    fig = Figure(figsize=(4 * ncols, 4 * nrows), layout="constrained")
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    colors = [g.color or f"C{k}" for k, g in enumerate(groups)]
    rng = np.random.default_rng(seed)
    for k, ax in enumerate(axes.flat):
        if k >= len(panels):
            ax.axis("off")
            continue
        label, values = panels[k]
        data = [v[np.isfinite(v)] for v in values]
        _boxes(ax, data, colors, [g.uid for g in groups])
        for kg, (d, color) in enumerate(zip(data, colors, strict=True)):
            if np.unique(d).size > 1:
                violin = ax.violinplot(
                    d,
                    positions=[kg + 0.3],
                    showmeans=True,
                    showmedians=True,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(color)
                # the lines of the mean and the median in one color, as the one
                # call of `violinplot` of sbmlsim 0.8.5 drew them
                for key in ["cmeans", "cmedians"]:
                    violin[key].set_color("C0")
            ax.scatter(
                rng.normal(kg + 0.3, 0.02, d.size),
                d,
                alpha=0.7,
                s=30,
                color="white",
                edgecolors="black",
            )
        ax.set_ylabel(label, fontsize=13, fontweight="bold")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.tick_params(axis="x", labelrotation=90)
    fig.savefig(path, dpi=300, bbox_inches="tight")


@dataclass(frozen=True, kw_only=True)
class Analysis(ABC):
    """An analysis of a runner over the simulations of its groups.

    The parameters it varies are the parameters of the runner which no
    simulation of its groups sets (`varied`, `free_parameters`). It writes one
    result per quantity: the tables and figures of sbmlsim 0.8.5, with their
    names, into `<analysis>`, and what sbmlsim 0.8.5 wrote no file of (the
    samples, the tables of the global indices) per group into
    `<analysis>/<group>`. The analyses are the four of this module,
    `LocalAnalysis`, `SamplingAnalysis`, `FastAnalysis` and `SobolAnalysis`; a
    subclass may vary other parameters by overriding `varied`.

    Attributes:
        groups: the groups of the analysis, at least one, the ids unique;
            stored as a tuple.
    """

    groups: Sequence[SensitivityGroup]

    def __post_init__(self) -> None:
        """Check the groups.

        Raises:
            ValueError: for no group or two groups of one id.
        """
        object.__setattr__(self, "groups", tuple(self.groups))
        if not self.groups:
            raise ValueError(f"{type(self).__name__} needs at least one group.")
        repeated = _repeated(g.uid for g in self.groups)
        if repeated:
            raise ValueError(
                f"The groups {repeated} of {type(self).__name__} are given twice."
            )

    def varied(
        self, parameters: Sequence[SensitivityParameter]
    ) -> list[SensitivityParameter]:
        """Get the parameters the analysis varies, which no group sets.

        Args:
            parameters: the parameters of the runner.

        Returns:
            The parameters, see `free_parameters`.
        """
        return free_parameters(parameters, [g.simulation for g in self.groups])

    def _check(
        self, parameters: Sequence[SensitivityParameter]
    ) -> list[SensitivityParameter]:
        """Check what the analysis needs, at the start of a run.

        Args:
            parameters: the parameters of the runner.

        Returns:
            The parameters the analysis varies.

        Raises:
            ValueError: if it varies no parameter.
        """
        varied = self.varied(parameters)
        if not varied:
            raise ValueError(
                f"{type(self).__name__} varies no parameter: the simulations of "
                f"its groups {[g.uid for g in self.groups]} set every parameter."
            )
        return varied

    @staticmethod
    def _group_path(path: Path, group: SensitivityGroup) -> Path:
        """Get the folder of a group, created if it does not exist."""
        group_path = path / group.uid
        group_path.mkdir(parents=True, exist_ok=True)
        return group_path

    @abstractmethod
    def _run(
        self, context: _Context, varied: Sequence[SensitivityParameter], path: Path
    ) -> None:
        """Run the analysis and write its tables and figures.

        Args:
            context: the simulator, the project, the outputs and observables
                and the seed of the run.
            varied: the parameters the analysis varies, see `_check`.
            path: the folder of the analysis.
        """


@dataclass(frozen=True, kw_only=True)
class LocalAnalysis(Analysis):
    """The local sensitivities of every group, each parameter by `+-delta`, as sbmlsim 0.8.5.

    Per group the designs `sampling.local` of `delta * k / n_var`,
    `k = 1, ..., n_var` (their values are those of the simulation of the
    group), the indices `sensitivity.local`, and the table of sbmlsim 0.8.5,
    `local_d<delta>_nvar<n_var>_<k>_<group>.tsv` (a row per parameter and
    output: `parameter`, `output`, `effect`, `Sij`, `|Sij|`,
    `classification`, `method`, `difference`, `n_var`, sorted by `|Sij|`;
    `Sij` the mean of `normalized` over the designs, as sbmlsim 0.8.5 averaged
    its steps), and its heatmap (`.png`, the parameters with their names, the
    outputs by name, cutoff 0.05, no clustering), in the folder of the
    analysis.

    Attributes:
        delta: the relative change of a parameter, in (0, 1).
        n_var: the number of steps of sbmlsim 0.8.5 up to `delta`, at least 1.
    """

    delta: float = 0.01
    n_var: int = 3

    def __post_init__(self) -> None:
        """Check the groups, the change and its steps.

        Raises:
            ValueError: see `Analysis`, for a change outside of (0, 1), which
                `sampling.local` refuses, and for a number of steps which is no
                positive int.
        """
        super().__post_init__()
        if not 0 < self.delta < 1:
            raise ValueError(
                f"The delta of a local analysis is in (0, 1), not {self.delta}."
            )
        if (
            isinstance(self.n_var, bool)
            or not isinstance(self.n_var, int)
            or self.n_var < 1
        ):
            raise ValueError(
                f"The n_var of a local analysis is a positive int, not {self.n_var!r}."
            )

    @override
    def _run(
        self, context: _Context, varied: Sequence[SensitivityParameter], path: Path
    ) -> None:
        """Run the analysis, see the class."""
        targets = [p.uid for p in varied]
        prefix = f"local_d{self.delta}_nvar{self.n_var}"
        path.mkdir(parents=True, exist_ok=True)
        for kg, group in enumerate(self.groups):
            results = [
                indices.local(
                    context.scan(
                        group,
                        sampling.local(
                            targets,
                            delta=kv / self.n_var * self.delta,
                            model=context.model(group),
                            simulation=group.simulation,
                        ),
                    )
                )
                for kv in range(1, self.n_var + 1)
            ]
            mean = self._mean(results, context.outputs)
            self._table(mean, results, varied, context.outputs).to_csv(
                path / f"{prefix}_{kg:>02}_{group.uid}.tsv", sep="\t", index=False
            )
            _heatmap(
                _labelled(mean, varied, context.outputs),
                "normalized",
                cutoff=CUTOFF_085,
                cluster_rows=False,
                cmap="seismic",
                vmin=-2.0,
                vmax=2.0,
                path=path / f"{prefix}_{kg:>02}_{group.uid}.png",
            )

    def _mean(
        self, results: Sequence[SensitivityResult], outputs: Sequence[SensitivityOutput]
    ) -> SensitivityResult:
        """Get the mean of `normalized` over the steps, the `Sij` of sbmlsim 0.8.5."""
        ds = results[0].ds.copy()
        for output in outputs:
            key = f"{output.observable}.normalized"
            ds[key] = sum(r.ds[key] for r in results) / len(results)
        return SensitivityResult(ds)

    def _table(
        self,
        mean: SensitivityResult,
        results: Sequence[SensitivityResult],
        varied: Sequence[SensitivityParameter],
        outputs: Sequence[SensitivityOutput],
    ) -> pd.DataFrame:
        """Get the table of sbmlsim 0.8.5 and log the nonlinear sensitivities.

        A sensitivity whose `raw` values of the steps differ by more than 10 %
        of their mean while it is larger than 0.01 is logged as an error, as
        sbmlsim 0.8.5 did: the output is nonlinear in the range or is not
        computed accurately.
        """
        rows = []
        for p in varied:
            for output in outputs:
                s = float(
                    mean[f"{output.observable}.normalized"].sel({PARAMETER: p.uid})
                )
                raw = np.array(
                    [
                        float(r[f"{output.observable}.raw"].sel({PARAMETER: p.uid}))
                        for r in results
                    ]
                )
                if not np.isclose(raw.mean(), 0.0) and s > 0.01:
                    max_diff = (raw.max() - raw.min()) / raw.mean()
                    if max_diff > 0.10:
                        logger.error(
                            "Large delta difference: %s%% for %s. Parameter '%s' on "
                            "output '%s'.",
                            format(max_diff * 100, ".1f"),
                            raw,
                            p.uid,
                            output.uid,
                        )
                effect, classification = _classes(
                    s, sensitivity_classification, sensitivity_classification_symbol
                )
                rows.append(
                    {
                        "parameter": p.uid,
                        "output": output.uid,
                        "effect": effect,
                        "Sij": s,
                        "|Sij|": np.abs(s),
                        "classification": classification,
                        "method": "LocalSensitivity",
                        "difference": self.delta,
                        "n_var": self.n_var,
                    }
                )
        df = pd.DataFrame(rows)
        return df.sort_values(
            by="|Sij|", ascending=False, na_position="last", ignore_index=True
        )


@dataclass(frozen=True, kw_only=True)
class _SampledAnalysis(Analysis):
    """An analysis of a design of `n` samples, see `sample_size`.

    Attributes:
        n: the number of samples of the design.
    """

    n: int

    def __post_init__(self) -> None:
        """Check the groups and the number of samples.

        Raises:
            ValueError: see `Analysis`, and for a number of samples which the
                design refuses.
        """
        super().__post_init__()
        self._check_size(self.n)

    def _check_size(self, n: int) -> None:
        """Check a number of samples, as the design of the analysis does.

        Raises:
            ValueError: if it is no positive int.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(
                f"The number of samples of {type(self).__name__} is a positive "
                f"int, not {n!r}."
            )

    def _samples(self) -> int:
        """Get the number of samples of a run, `n` or the one of a smoke run.

        Raises:
            ValueError: for a value of `SBMLSIM_SENSITIVITY_N` which is no
                positive int or which the design refuses.
        """
        n = sample_size(self.n)
        try:
            self._check_size(n)
        except ValueError as err:
            err.add_note(
                f"{SAMPLE_SIZE} replaces the number of samples {self.n} of "
                f"{type(self).__name__} by {n}."
            )
            raise
        return n

    @override
    def _check(
        self, parameters: Sequence[SensitivityParameter]
    ) -> list[SensitivityParameter]:
        """Check the parameters, their distributions and the number of samples.

        Raises:
            ValueError: see `Analysis`, `distributions` and `_samples`.
        """
        varied = super()._check(parameters)
        distributions(varied)
        self._samples()
        return varied


#: the quantiles of the statistics of a sampling: the median, the 5 % and 95 %
#: quantiles and the 2.5 % and 97.5 % quantiles of the uncertainty `U`
_QUANTILES: tuple[float, ...] = (0.5, 0.05, 0.95, 0.025, 0.975)


def _statistics(
    result: ScanResult, observables: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Get the statistics of sbmlsim 0.8.5 of the outputs of a sampling.

    `ScanResult.summary` over every point: `mean`, the sample standard
    deviation `std`, `cv = std / mean`, `min`, `max`, the `median`, the
    quantiles `q005` and `q095` and the uncertainty
    `U = (P97.5 - P2.5) / P50`; a failed point (`NaN`) is skipped.

    Args:
        result: the result of the sampling of a group.
        observables: the observables of the outputs.

    Returns:
        The statistics of every observable.
    """
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        # a zero mean or a single point gives an undefined cv or std
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = result.summary(
            statistics=("mean", "sd", "cv", "min", "max"), quantiles=_QUANTILES
        ).ds
        stats: dict[str, dict[str, float]] = {}
        for observable in observables:

            def value(label: str, observable: str = observable) -> float:
                return float(summary[observable].sel(statistic=label))

            median = value("q0.5")
            stats[observable] = {
                "mean": value("mean"),
                "median": median,
                "std": value("sd"),
                "cv": value("cv"),
                "min": value("min"),
                "q005": value("q0.05"),
                "q095": value("q0.95"),
                "max": value("max"),
                "U": float(
                    np.float64(value("q0.975") - value("q0.025")) / np.float64(median)
                ),
            }
    return stats


@dataclass(frozen=True, kw_only=True)
class SamplingAnalysis(_SampledAnalysis):
    """The uncertainty of the outputs over a Latin hypercube of the parameters.

    One design `sampling.lhs` of `sample_size(n)` points for every group; per
    group the parameters and outputs of every point, `samples.tsv` (sbmlsim
    0.8.5 kept them in its cache only). The tables and figures of sbmlsim 0.8.5
    compare the groups: `sampling_N<n>.tsv` (a row per group and output:
    `gid`, `gname`, `uid`, `name`, `N`, `mean`, `median`, `std`, `cv`, `min`,
    `q005`, `q095`, `max`, the uncertainty `U = (P97.5 - P2.5) / P50`, its
    `symbol` and `classification`, `unit`; the statistics of
    `ScanResult.summary`, `std` the sample standard deviation),
    `sampling_N<n>_compact.tsv` and its LaTeX table `sampling_N<n>.tex` (an
    output per row, `mean (cv in %)` per group), and the figures of the samples
    of the parameters and of the outputs per group,
    `sampling_N<n>_sensitivity_samples.png` and `..._outputs.png`.

    Attributes:
        n: the number of samples, see `sample_size`.
    """

    n: int = 1000

    @override
    def _run(
        self, context: _Context, varied: Sequence[SensitivityParameter], path: Path
    ) -> None:
        """Run the analysis, see the class."""
        n = self._samples()
        design = sampling.lhs(distributions(varied), n=n, seed=context.seed)
        observables = {o.observable: o.uid for o in context.outputs}
        results: dict[str, ScanResult] = {}
        for group in self.groups:
            group_path = self._group_path(path, group)
            res = context.scan(group, context.design(group, design, varied))
            results[group.uid] = res
            res.ds[list(observables)].to_dataframe().rename(columns=observables).to_csv(
                group_path / "samples.tsv", sep="\t"
            )
        self._write_085(context, varied, results, n, path)

    def _write_085(
        self,
        context: _Context,
        varied: Sequence[SensitivityParameter],
        results: Mapping[str, ScanResult],
        n: int,
        path: Path,
    ) -> None:
        """Write the tables and figures of sbmlsim 0.8.5, see the class."""
        prefix = f"sampling_N{n}"
        units = {
            o.uid: context.unit(results[self.groups[0].uid], o) for o in context.outputs
        }
        observables = [o.observable for o in context.outputs]
        rows = []
        stats: dict[tuple[str, str], dict[str, float]] = {}
        for group in self.groups:
            group_stats = _statistics(results[group.uid], observables)
            for output in context.outputs:
                stats[group.uid, output.uid] = group_stats[output.observable]
                symbol, classification = _classes(
                    stats[group.uid, output.uid]["U"],
                    uncertainty_classification,
                    uncertainty_classification_symbol,
                )
                rows.append(
                    {
                        "gid": group.uid,
                        "gname": group.name,
                        "uid": output.uid,
                        "name": output.name,
                        "N": n,
                        **stats[group.uid, output.uid],
                        "symbol": symbol,
                        "classification": classification,
                        "unit": units[output.uid],
                    }
                )
        pd.DataFrame(rows).to_csv(path / f"{prefix}.tsv", sep="\t", index=False)

        compact = pd.DataFrame(
            [
                {
                    "output": output.name,
                    **{
                        g.uid: (
                            f"{stats[g.uid, output.uid]['mean']:.3g} "
                            f"({stats[g.uid, output.uid]['cv'] * 100:.1f})"
                        )
                        for g in self.groups
                    },
                    "unit": units[output.uid],
                }
                for output in context.outputs
            ]
        )
        compact.to_csv(path / f"{prefix}_compact.tsv", sep="\t", index=False)
        latex = compact.to_latex(None, index=False)
        latex = latex.replace("∞", r"$\infty$").replace("*", r"$\cdot$")
        (path / f"{prefix}.tex").write_text(latex, encoding="utf-8")

        samples = [
            (
                f"{p.uid}: {p.name} [{p.unit if p.unit else 'AU'}]",
                [
                    np.asarray(results[g.uid].ds[p.uid].values, dtype=float)
                    for g in self.groups
                ],
            )
            for p in varied
        ]
        _plot_groups(
            samples,
            self.groups,
            path / f"{prefix}_sensitivity_samples.png",
            context.seed,
        )
        outputs = [
            (
                f"{o.name} [{units[o.uid] if units[o.uid] else 'AU'}]",
                [
                    np.asarray(results[g.uid].ds[o.observable].values, dtype=float)
                    for g in self.groups
                ],
            )
            for o in context.outputs
        ]
        _plot_groups(
            outputs,
            self.groups,
            path / f"{prefix}_sensitivity_outputs.png",
            context.seed,
        )


@dataclass(frozen=True, kw_only=True)
class _GlobalAnalysis(_SampledAnalysis):
    """A variance based analysis: the indices `S1` and `ST` of every group.

    One design for every group; per group the tables of `S1` and `ST` per
    output, `<output>.S1.tsv` and `<output>.ST.tsv` (sbmlsim 0.8.5 wrote no
    tables of a global analysis), and the figures of sbmlsim 0.8.5: the
    heatmaps of `S1` and `ST` (`<prefix>_sensitivity_<k>_<group>_S1.png`,
    `..._ST.png`, the parameters with their names, the outputs by name, cutoff
    0.05, no clustering, viridis from 0 to 1) and the indices of every output
    with their intervals, `<prefix>_sensitivity_<k>_<group>_<o>_<output>.png`
    (`plot_indices`).

    Attributes:
        n: the base number of samples, see `sample_size`.
    """

    @abstractmethod
    def _prefix(self, n: int) -> str:
        """Get the prefix of the files of sbmlsim 0.8.5."""

    @abstractmethod
    def _design(
        self, parameters: Sequence[SensitivityParameter], n: int, seed: int
    ) -> Dimension:
        """Get the design over the parameters the analysis varies."""

    @abstractmethod
    def _indices(self, result: ScanResult) -> SensitivityResult:
        """Get the indices of the result of the design."""

    @override
    def _run(
        self, context: _Context, varied: Sequence[SensitivityParameter], path: Path
    ) -> None:
        """Run the analysis, see the class."""
        n = self._samples()
        prefix = self._prefix(n)
        design = self._design(varied, n, context.seed)
        for kg, group in enumerate(self.groups):
            group_path = self._group_path(path, group)
            result = self._indices(
                context.scan(group, context.design(group, design, varied))
            )
            context.write_indices(result, ["S1", "ST"], group_path)
            figures = f"{prefix}_sensitivity_{kg:>02}_{group.uid}"
            for ko, output in enumerate(context.outputs):
                plots.plot_indices(
                    result,
                    output.observable,
                    title=f"{output.name} ({group.name})",
                    path=path / f"{figures}_{ko:>03}_{output.uid}.png",
                )
            labelled = _labelled(result, varied, context.outputs)
            for key in ["ST", "S1"]:
                _heatmap(
                    labelled,
                    key,
                    cutoff=CUTOFF_085,
                    cluster_rows=False,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    path=path / f"{figures}_{key}.png",
                )


@dataclass(frozen=True, kw_only=True)
class FastAnalysis(_GlobalAnalysis):
    """The indices of the extended FAST, `sampling.fast` and `sensitivity.fast`.

    The files of sbmlsim 0.8.5 have the prefix `fast_M4_N<n>`.

    Attributes:
        n: the number of samples per parameter, more than 64 (`4 m^2` of the
            `m = 4` harmonics of `sampling.fast`), see `sample_size`.
    """

    n: int = 1000

    @override
    def _check_size(self, n: int) -> None:
        """Check a number of samples, more than 64, as `sampling.fast` does.

        Raises:
            ValueError: if it is no int above 64.
        """
        super()._check_size(n)
        if n <= 64:
            raise ValueError(
                f"The number of samples of FastAnalysis is more than 64, not {n}."
            )

    @override
    def _prefix(self, n: int) -> str:
        """Get the prefix of the files of sbmlsim 0.8.5, `fast_M4_N<n>`."""
        return f"fast_M4_N{n}"

    @override
    def _design(
        self, parameters: Sequence[SensitivityParameter], n: int, seed: int
    ) -> Dimension:
        """Get the design over the parameters, see `sampling.fast`."""
        return sampling.fast(distributions(parameters), n=n, seed=seed)

    @override
    def _indices(self, result: ScanResult) -> SensitivityResult:
        """Get the indices, see `sensitivity.fast`."""
        return indices.fast(result)


@dataclass(frozen=True, kw_only=True)
class SobolAnalysis(_GlobalAnalysis):
    """The Sobol indices, `sampling.sobol` and `sensitivity.sobol`.

    The files of sbmlsim 0.8.5 have the prefix `sobol_N<n>`.

    Attributes:
        n: the base number of samples, a power of two of at least 2, see
            `sample_size`.
    """

    n: int = 4096

    @override
    def _check_size(self, n: int) -> None:
        """Check a number of samples, a power of two, as `sampling.sobol` does.

        Raises:
            ValueError: if it is no power of two of at least 2.
        """
        super()._check_size(n)
        if n < 2 or n & (n - 1):
            raise ValueError(
                "The number of samples of SobolAnalysis is a power of two of at "
                f"least 2, not {n}."
            )

    @override
    def _prefix(self, n: int) -> str:
        """Get the prefix of the files of sbmlsim 0.8.5, `sobol_N<n>`."""
        return f"sobol_N{n}"

    @override
    def _design(
        self, parameters: Sequence[SensitivityParameter], n: int, seed: int
    ) -> Dimension:
        """Get the design over the parameters, see `sampling.sobol`."""
        return sampling.sobol(distributions(parameters), n=n, seed=seed)

    @override
    def _indices(self, result: ScanResult) -> SensitivityResult:
        """Get the indices, see `sensitivity.sobol`."""
        return indices.sobol(result)


class CommandLine(NamedTuple):
    """The words of the command line of a runner.

    Attributes:
        analyses: the names of the analyses, the default analyses without one.
        choices: the words of the choices of the script, in their order, none
            if none is given.
        workers: the processes of `--workers`, `None` if not given or `0`
            (`n_workers=None`, sbmlsim chooses).
        out: the folder of `--out`, `None` if not given.
    """

    analyses: tuple[str, ...]
    choices: tuple[str, ...]
    workers: int | None = None
    out: Path | None = None


def _workers(text: str) -> int:
    """Read the value of `--workers`: `0` or a number of processes of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is no whole number") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is negative (0 or at least 1)")
    return value


def _report(items: Mapping[str, str], path: Path) -> None:
    """Print the report of a run."""
    display.section("Report", icon=display.ICON_REPORT)
    display.key_values(dict(items))
    display.link("results", path)


@dataclass(frozen=True, kw_only=True)
class SensitivityRunner:
    """The sensitivity analyses of a model of a project.

    The runner gives what is the model's own (the parameters, the outputs with
    their observables, the analyses with their groups) and runs them with
    `main` from a command line, with `run`, or as an entry of
    `Project.sensitivities` with the kind `sensitivity`. A run writes the table
    of the parameters (`write_parameters`) into its folder and the results of
    an analysis into `<analysis>` below it, or below a subfolder of it, see
    `Analysis`. A model with several simulations has a runner per simulation,
    see `run_subfolders`. What can be checked is checked when the runner is
    created, and the rest at the start of a run, before anything is written.

    Attributes:
        parameters: the function which gives the parameters of the analyses
            with their bounds; it reads the model, so it is called by a run.
        outputs: the outputs of the analyses, at least one, the ids, the names
            and the observables unique; stored as a tuple.
        observables: the observables of the simulation of a group, which give
            the observables of the outputs, or a function which builds them
            when a run starts, e.g. the non-compartmental analyses which a
            model package supplies; a sequence is stored as a tuple.
        analyses: the analyses by name, the name on the command line and the
            folder; at least one, stored read-only.
        default: the analyses which run without a name, at least one.
        seed: the seed of the designs.
        project: the project, whose variants the groups run and whose
            simulator settings the simulations use; `None` for a runner which
            a project lists, which binds it (`Project.sensitivity`).
        out: the folder of the results of a run without a folder of its own
            (`run(path=...)`, `--out`), e.g. the results directory of a model
            package; a run without either raises.
        description: what the analyses analyse, listed by `Project.analyses`.
    """

    parameters: Callable[[], Sequence[SensitivityParameter]]
    outputs: Sequence[SensitivityOutput]
    observables: Sequence[Observable] | Callable[[], Sequence[Observable]]
    analyses: Mapping[str, Analysis]
    default: Sequence[str] = ("local",)
    seed: int = SEED
    project: Project | None = None
    out: Path | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Check the outputs and the analyses.

        Raises:
            ValueError: for no output, two outputs of one id, name or
                observable, an output whose observable is none of the
                observables, no analysis, no default analysis, or a default
                analysis which is none of the analyses or is given twice, or a
                group of a variant which the project has not.
            TypeError: for a default which is a string.
        """
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if isinstance(self.observables, Sequence):
            object.__setattr__(self, "observables", tuple(self.observables))
        object.__setattr__(self, "analyses", MappingProxyType(dict(self.analyses)))
        object.__setattr__(
            self, "default", _sequence_of_names(self.default, "default analyses")
        )
        if not self.outputs:
            raise ValueError("A sensitivity analysis needs at least one output.")
        repeated = _repeated(o.uid for o in self.outputs)
        if repeated:
            raise ValueError(f"The outputs {repeated} are given twice.")
        repeated = _repeated(o.name for o in self.outputs)
        if repeated:
            raise ValueError(f"The names {repeated} of the outputs are given twice.")
        repeated = _repeated(o.observable for o in self.outputs)
        if repeated:
            raise ValueError(f"The observables {repeated} are outputs twice.")
        if isinstance(self.observables, Sequence):
            self._check_observables(self.observables)
        if not self.analyses:
            raise ValueError("A sensitivity analysis needs at least one analysis.")
        if not self.default:
            raise ValueError(
                "A sensitivity analysis needs at least one default analysis."
            )
        unknown = [name for name in self.default if name not in self.analyses]
        if unknown:
            raise ValueError(
                f"The default analyses {unknown} are none of {list(self.analyses)}."
            )
        repeated = _repeated(self.default)
        if repeated:
            raise ValueError(f"The default analyses {repeated} are given twice.")
        if self.project is not None:
            self._check_variants(self.project)

    def _check_observables(self, observables: Sequence[Observable]) -> None:
        """Check that every output names an observable, or a parameter of a `PK` one.

        Raises:
            ValueError: naming the observables which are missing.
        """
        ids = {observable.id for observable in observables}
        analyses = {o.id: o.parameters for o in observables if isinstance(o, PK)}

        def known(observable: str) -> bool:
            if observable in ids:
                return True
            pk, _, parameter = observable.partition(".")
            if pk not in analyses:
                return False
            parameters = analyses[pk]
            return parameters is None or parameter in parameters

        missing = [o.observable for o in self.outputs if not known(o.observable)]
        if missing:
            raise ValueError(f"The outputs name no observables {missing}.")

    def _check_variants(self, project: Project) -> None:
        """Check that the project has the variant of every group.

        Raises:
            ValueError: for a group of a variant the project has not, or a
                project without variants.
        """
        for name, analysis in self.analyses.items():
            for group in analysis.groups:
                try:
                    project.variant_path(group.variant)
                except (KeyError, ValueError) as err:
                    message = err.args[0] if err.args else str(err)
                    raise ValueError(
                        f"The group '{group.uid}' of the analysis '{name}' runs "
                        f"the variant {group.variant!r}: {message}"
                    ) from err

    def bound(self, project: Project) -> SensitivityRunner:
        """Get the runner of a project, which binds a runner without a project.

        Args:
            project: the project which lists the runner.

        Returns:
            The runner with the project.

        Raises:
            ValueError: for a runner of another project, or a group of a
                variant which the project has not.
        """
        if self.project is None:
            return dataclasses.replace(self, project=project)
        if self.project is not project and self.project != project:
            raise ValueError(
                f"The runner of the project '{project.name}' runs the project "
                f"'{self.project.name}'; give it no project, or the one it is "
                "listed by."
            )
        return self

    def _bound_project(self) -> Project:
        """Get the project of the runner.

        Raises:
            ValueError: for a runner without a project.
        """
        if self.project is None:
            raise ValueError(
                "The sensitivity runner has no project: give it one, or run it "
                "as an entry of Project.sensitivities."
            )
        return self.project

    def _built_observables(self) -> tuple[Observable, ...]:
        """Get the observables, built by the function of the runner if it has one.

        Raises:
            ValueError: if an output names no observable.
        """
        if isinstance(self.observables, Sequence):
            return tuple(self.observables)
        observables = tuple(self.observables())
        self._check_observables(observables)
        return observables

    def _names(self, names: Sequence[str] | None) -> tuple[str, ...]:
        """Get the names of the analyses of a run, `default` if `None`.

        Raises:
            TypeError: for names which are a string.
            ValueError: for a name which is no analysis or is given twice.
        """
        names = self.default if names is None else _sequence_of_names(names, "names")
        unknown = [name for name in names if name not in self.analyses]
        if unknown:
            raise ValueError(f"Unknown analyses {unknown}, use {list(self.analyses)}.")
        repeated = _repeated(names)
        if repeated:
            raise ValueError(f"The analyses {repeated} are given twice.")
        return tuple(names)

    def _plan(
        self,
        names: Sequence[str],
        parameters: Sequence[SensitivityParameter],
        where: str = "",
    ) -> dict[str, list[SensitivityParameter]]:
        """Check the named analyses and get the parameters each varies.

        Raises:
            ValueError: see `Analysis._check`, with a note naming the analysis.
        """
        varied: dict[str, list[SensitivityParameter]] = {}
        for name in names:
            try:
                varied[name] = self.analyses[name]._check(parameters)
            except ValueError as err:
                err.add_note(f"The analysis '{name}'{where} cannot run.")
                raise
        return varied

    def _variants(self, project: Project, names: Sequence[str]) -> set[str]:
        """Get the variants of the groups of the named analyses other than the default."""
        default = next(iter(project.variants))
        return {
            _variant(project, group)
            for name in names
            for group in self.analyses[name].groups
        } - {default}

    def _check_designs(
        self,
        project: Project,
        names: Sequence[str],
        varied: Mapping[str, Sequence[SensitivityParameter]],
        builds: Mapping[str, Mapping[str, float]],
        where: str = "",
    ) -> None:
        """Check that the global designs of the named analyses carry to every variant.

        Raises:
            ValueError: see `design_factors`, with a note naming the analysis
                and the group.
        """
        for name in names:
            analysis = self.analyses[name]
            if isinstance(analysis, LocalAnalysis):
                # a local design is drawn around the values of each variant
                continue
            for group in analysis.groups:
                variant = _variant(project, group)
                if variant not in builds:
                    continue
                try:
                    design_factors(varied[name], builds[variant])
                except ValueError as err:
                    err.add_note(
                        f"The group '{group.uid}' of the analysis '{name}'{where} "
                        f"runs the variant '{variant}'."
                    )
                    raise

    def _path(self, path: Path | None) -> Path:
        """Get the folder of a run, `out` without one.

        Raises:
            ValueError: if neither is given.
        """
        if path is not None:
            return Path(path)
        if self.out is None:
            raise ValueError(
                "The sensitivity analyses have no folder of their results: give "
                "the runner `out`, or the run a path (`--out`)."
            )
        return self.out

    def run(
        self,
        names: Sequence[str] | None = None,
        *,
        path: Path | None = None,
        subfolder: str | None = None,
        workers: int | None = None,
    ) -> Path:
        """Run analyses and write the table of the parameters.

        The names, the parameters (their distributions, whether an analysis
        varies one, whether a global design carries to the variant of every
        group, `design_factors`) and the numbers of samples of a smoke run are
        checked before anything is written. The values of the table are those
        of the default variant; with groups of another variant the table has
        their values as well (`value_<variant>`).

        Args:
            names: the names of the analyses, `default` if `None`.
            path: the folder of the results and of the table of the parameters,
                `out` if `None`.
            subfolder: the folder of the analyses below `path`; the table of
                the parameters stays in `path`. Several simulations of a model
                run with `run_subfolders`, which writes the table once.
            workers: the processes of the simulator, `None` lets sbmlsim
                choose (a pool for large designs; the script has the guard
                `if __name__ == "__main__":`).

        Returns:
            The folder of the results, `path`.

        Raises:
            TypeError: for names which are a string.
            ValueError: for a runner without a project, a run without a folder,
                a name which is no analysis or is given twice, a subfolder
                which does not lie below `path`, and see `distributions`,
                `sample_size` and `design_factors` (a global design of a group
                of another variant); before anything is written.
        """
        if subfolder is not None:
            _check_subfolder(subfolder)
        path = self._path(path)
        prepared = prepare_run(self, names)
        prepared.write_table(path, show=True)
        analyses_path = path if subfolder is None else path / subfolder
        simulator = prepared.simulator(workers)
        for unit in prepared.units:
            prepared.run_unit(unit, analyses_path / unit.folder, simulator)
        _report({"analyses": ", ".join(u.analysis for u in prepared.units)}, path)
        return path

    def parse_command_line(
        self, argv: Sequence[str] | None = None, *, choices: Sequence[str] = ()
    ) -> CommandLine:
        """Read the names of the analyses, and of choices of the script, from a command line.

        A script with a choice of its own, e.g. the simulation of a model with
        several, reads it here with the names of the analyses, in any order
        (`local sampling multi`), and runs the analyses itself.

        Args:
            argv: the arguments, those of the process if `None`.
            choices: the words of the choices of the script, none an analysis.

        Returns:
            The names of the analyses, the default analyses without one, the
            words of the choices, the `--workers` (`None` if absent or `0`),
            which a script passes as `workers`, and the folder of `--out`.

        Raises:
            TypeError: for arguments which are a string.
            ValueError: for a choice which is the name of an analysis.
            SystemExit: with 2 for a word which is neither an analysis nor a
                choice or which is given twice (argparse), with 0 for `--help`.
        """
        if argv is not None:
            argv = _sequence_of_names(argv, "argv")
        choices = _sequence_of_names(choices, "choices")
        clashes = sorted(set(choices) & set(self.analyses))
        if clashes:
            raise ValueError(f"The choices {clashes} are names of analyses.")
        words = ", ".join([*self.analyses, *choices])
        title = (
            f"of {self.project.title or self.project.name}"
            if self.project is not None
            else "of the model"
        )
        into = f", into {self.out}" if self.out is not None else ""
        parser = argparse.ArgumentParser(
            description=f"The sensitivity analyses {title}{into}.",
            epilog=(
                f"The environment variable {SAMPLE_SIZE} replaces the numbers of "
                "samples of the sampling and the global analyses for a smoke run."
            ),
        )
        # the words are checked here: argparse checks the default of a
        # positional `nargs="*"` with `choices` as one choice
        parser.add_argument(
            "words",
            nargs="*",
            metavar="ANALYSIS" if not choices else "WORD",
            help=(
                f"the analyses, of {', '.join(self.analyses)} "
                f"(default: {' '.join(self.default)})"
                + (
                    f"; the choices of the script, of {', '.join(choices)}"
                    if choices
                    else ""
                )
            ),
        )
        parser.add_argument(
            "--workers",
            type=_workers,
            default=0,
            metavar="N",
            help=(
                "the processes of the pool of a run, at least 1 (a pool needs "
                "about 300 MB per process for a large model); 0 or absent: "
                "sbmlsim chooses (every CPU for large designs)"
            ),
        )
        parser.add_argument(
            "--out",
            type=Path,
            default=None,
            metavar="DIR",
            help=(
                "the folder of the results"
                + (f" (default: {self.out})" if self.out is not None else "")
            ),
        )
        args = parser.parse_intermixed_args(argv)
        given: list[str] = args.words
        unknown = [w for w in given if w not in self.analyses and w not in choices]
        if unknown:
            parser.error(
                f"invalid choice: {', '.join(repr(w) for w in unknown)} "
                f"(choose from {words})"
            )
        repeated = _repeated(given)
        if repeated:
            parser.error(f"{', '.join(repr(w) for w in repeated)} given twice")
        analyses = tuple(w for w in given if w in self.analyses)
        return CommandLine(
            analyses=analyses or tuple(self.default),
            choices=tuple(w for w in given if w in choices),
            workers=args.workers or None,
            out=args.out,
        )

    def main(self, argv: Sequence[str] | None = None) -> None:
        """Run the analyses named on the command line, see `parse_command_line`, `run`.

        Args:
            argv: the arguments, those of the process if `None`: the names of
                the analyses, `default` without one; an unknown name exits with
                2; `--workers N` caps the processes of the pool, `--out DIR`
                is the folder of the results, `out` without.
        """
        words = self.parse_command_line(argv)
        self.run(words.analyses, path=words.out, workers=words.workers)


@dataclass(frozen=True)
class SensitivityUnit:
    """An analysis of a run, in a subfolder for a run of several runners.

    Attributes:
        subfolder: the subfolder of its runner, `None` for a single runner.
        analysis: the name of the analysis.
    """

    subfolder: str | None
    analysis: str

    @property
    def folder(self) -> Path:
        """The folder of the analysis relative to the folder of the run."""
        if self.subfolder is None:
            return Path(self.analysis)
        return Path(self.subfolder) / self.analysis


@dataclass(frozen=True)
class PreparedRun:
    """A checked run of sensitivity analyses: its table and its units.

    Created by `prepare_run`, which checks everything a run needs before
    anything is written; `write_table` writes the table of the parameters and
    `run_unit` runs one analysis into a folder.

    Attributes:
        project: the project of the runners.
        runners: the runners by their subfolder, `None` for a single runner.
        parameters: the parameters of the table.
        observables: the observables of every runner.
        plans: the names of the analyses of every runner and the parameters
            each varies.
        builds: the values of the parameters in the variants of the groups
            other than the default one, by variant.
    """

    project: Project
    runners: Mapping[str | None, SensitivityRunner]
    parameters: tuple[SensitivityParameter, ...]
    observables: Mapping[str | None, tuple[Observable, ...]]
    plans: Mapping[
        str | None, tuple[tuple[str, ...], dict[str, list[SensitivityParameter]]]
    ]
    builds: Mapping[str, Mapping[str, float]]
    _models: dict[Path, SBMLModel] = field(default_factory=dict, compare=False)

    @property
    def units(self) -> tuple[SensitivityUnit, ...]:
        """The analyses of the run, in the order of the runners and their names."""
        return tuple(
            SensitivityUnit(key, name)
            for key, (names, _) in self.plans.items()
            for name in names
        )

    def write_table(self, path: Path, *, show: bool = False) -> pd.DataFrame:
        """Write the table of the parameters into a folder.

        Args:
            path: the folder, created if it does not exist.
            show: print the table on the console of the package.

        Returns:
            The table.
        """
        path.mkdir(parents=True, exist_ok=True)
        df = write_parameters(self.parameters, path, self.builds)
        if show:
            with pd.option_context("display.float_format", "{:.5g}".format):
                console.print(df)
        return df

    def simulator(self, workers: int | None) -> Simulator:
        """Create the simulator of a run with the settings of the project.

        Args:
            workers: the processes, `None` lets sbmlsim choose.
        """
        return self.project.simulator.simulator(workers)

    def run_unit(
        self, unit: SensitivityUnit, folder: Path, simulator: Simulator
    ) -> None:
        """Run an analysis of the run into a folder.

        Args:
            unit: the analysis, one of `units`.
            folder: the folder of the analysis.
            simulator: the simulator, see `simulator`.
        """
        runner = self.runners[unit.subfolder]
        _, varied = self.plans[unit.subfolder]
        context = _Context(
            simulator=simulator,
            project=self.project,
            outputs=runner.outputs,
            observables=self.observables[unit.subfolder],
            seed=runner.seed,
            builds=self.builds,
            models=self._models,
        )
        runner.analyses[unit.analysis]._run(context, varied[unit.analysis], folder)


def bind(
    target: SensitivityRunner | Mapping[str, SensitivityRunner], project: Project
) -> SensitivityRunner | dict[str, SensitivityRunner]:
    """Bind a runner or the runners of subfolders to the project which lists them.

    Args:
        target: a runner or the runners by their subfolder.
        project: the project.

    Returns:
        The runners with the project, see `SensitivityRunner.bound`.

    Raises:
        TypeError: for an object which is neither.
        ValueError: see `SensitivityRunner.bound`.
    """
    if isinstance(target, SensitivityRunner):
        return target.bound(project)
    if isinstance(target, Mapping) and all(
        isinstance(runner, SensitivityRunner) for runner in target.values()
    ):
        return {key: runner.bound(project) for key, runner in target.items()}
    raise TypeError(
        "A sensitivity analysis of a project is a SensitivityRunner or a mapping "
        f"of them by subfolder, not a {type(target).__name__}."
    )


def prepare_run(
    target: SensitivityRunner | Mapping[str, SensitivityRunner],
    names: Sequence[str] | None = None,
    subfolders: Sequence[str] | None = None,
) -> PreparedRun:
    """Check a run of a runner or of the runners of subfolders, before anything is written.

    The names of the analyses, the subfolders, the project, the parameters
    (read once per function), their distributions, whether an analysis varies
    one, the numbers of samples of a smoke run and whether a global design
    carries to the variant of every group.

    Args:
        target: a runner, or the runners by the name of their subfolder.
        names: the names of the analyses, the default analyses of every runner
            if `None`.
        subfolders: the subfolders which run, every one if `None`; only for
            the runners of subfolders.

    Returns:
        The prepared run.

    Raises:
        TypeError: for names or subfolders which are a string.
        ValueError: for no runner, no subfolder, a subfolder which has no
            runner, is given twice or does not lie below the folder, runners
            of different projects or with different parameters (they share one
            table), subfolders of a single runner, and see
            `SensitivityRunner.run`.
    """
    runners: dict[str | None, SensitivityRunner]
    if isinstance(target, SensitivityRunner):
        if subfolders is not None:
            raise ValueError(
                "A single sensitivity runner has no subfolders, a mapping of "
                "runners by subfolder has."
            )
        runners = {None: target}
    elif isinstance(target, Mapping) and all(
        isinstance(runner, SensitivityRunner) for runner in target.values()
    ):
        runners = {}
        runners.update(_subfolder_runners(target, subfolders))
    else:
        raise TypeError(
            "A run of sensitivity analyses is of a SensitivityRunner or a mapping "
            f"of them by subfolder, not of a {type(target).__name__}."
        )
    keys = list(runners)
    first = runners[keys[0]]
    project = first._bound_project()
    others = [
        key
        for key in keys
        if runners[key].project is not project and runners[key].project != project
    ]
    if others:
        raise ValueError(
            f"The runners of the subfolders {others} have another project than the "
            f"one of '{keys[0]}'; the subfolders of a run share one project."
        )
    for runner in runners.values():
        runner._check_variants(project)

    # the parameters of every runner, read once per function
    read: dict[int, list[SensitivityParameter]] = {}
    parameters: dict[str | None, list[SensitivityParameter]] = {}
    for key in keys:
        function = runners[key].parameters
        if id(function) not in read:
            read[id(function)] = list(function())
        parameters[key] = read[id(function)]
    table = parameters_to_df(parameters[keys[0]])
    different = [
        key for key in keys if not parameters_to_df(parameters[key]).equals(table)
    ]
    if different:
        raise ValueError(
            f"The runners of the subfolders {different} give other parameters than "
            f"the one of '{keys[0]}'; the subfolders of a run share one table of "
            "the parameters."
        )

    plans: dict[
        str | None, tuple[tuple[str, ...], dict[str, list[SensitivityParameter]]]
    ] = {}
    for key in keys:
        runner = runners[key]
        where = "" if key is None else f" of the subfolder '{key}'"
        try:
            run_names = runner._names(names)
        except ValueError as err:
            if key is not None:
                err.add_note(f"The subfolder '{key}' cannot run.")
            raise
        plans[key] = (run_names, runner._plan(run_names, parameters[key], where))
    observables = {key: runners[key]._built_observables() for key in keys}

    variants = set().union(
        *(runners[key]._variants(project, plans[key][0]) for key in keys)
    )
    # the values of every parameter of every subfolder (the tables are equal,
    # the union keeps a parameter of any subfolder in the values)
    union = list({p.uid: p for key in keys for p in parameters[key]}.values())
    uids = [p.uid for p in union]
    builds = {
        variant: build_values(project.variant_path(variant), uids)
        for variant in project.variants
        if variant in variants
    }
    for key in keys:
        run_names, varied = plans[key]
        where = "" if key is None else f" of the subfolder '{key}'"
        runners[key]._check_designs(project, run_names, varied, builds, where)
    return PreparedRun(
        project=project,
        runners=runners,
        parameters=tuple(parameters[keys[0]]),
        observables=observables,
        plans=plans,
        builds=builds,
    )


def _subfolder_runners(
    runners: Mapping[str, SensitivityRunner], subfolders: Sequence[str] | None
) -> dict[str, SensitivityRunner]:
    """Get the runners of the subfolders which run, checked.

    Raises:
        TypeError: for subfolders which are a string.
        ValueError: for no runner, no subfolder, a subfolder which has no
            runner, is given twice or does not lie below the folder.
    """
    if not runners:
        raise ValueError(
            "A run of subfolders needs at least one runner, there is no runner."
        )
    keys = (
        tuple(runners)
        if subfolders is None
        else _sequence_of_names(subfolders, "subfolders")
    )
    if not keys:
        raise ValueError(
            f"A run of subfolders needs at least one subfolder, of {list(runners)}."
        )
    unknown = [key for key in keys if key not in runners]
    if unknown:
        raise ValueError(
            f"The subfolders {unknown} have no runner, use {list(runners)}."
        )
    repeated = _repeated(keys)
    if repeated:
        raise ValueError(f"The subfolders {repeated} are given twice.")
    for key in keys:
        _check_subfolder(key)
    return {key: runners[key] for key in keys}


def run_subfolders(
    runners: Mapping[str, SensitivityRunner],
    names: Sequence[str] | None = None,
    *,
    subfolders: Sequence[str] | None = None,
    path: Path | None = None,
    workers: int | None = None,
) -> Path:
    """Run the analyses of several simulations of a model, each into a subfolder.

    A model with several simulations (e.g. a single dose and weekly doses) has
    a runner per simulation, with its outputs and groups, by the name of its
    subfolder. The table of the parameters is written and printed once into
    `path`, the analyses of a runner run into
    `path / <subfolder> / <analysis> / <group>`. Everything is checked before
    anything is written, for every runner, see `prepare_run`.

    Args:
        runners: the runners by the name of their subfolder.
        names: the names of the analyses, the default analyses of every runner
            if `None`.
        subfolders: the subfolders which run, every one if `None`.
        path: the folder of the results, the `out` of the first runner which
            runs if `None`.
        workers: the processes of the simulator, see `SensitivityRunner.run`.

    Returns:
        The folder of the results, `path`.

    Raises:
        TypeError: for names or subfolders which are a string.
        ValueError: see `prepare_run` and `SensitivityRunner.run`; before
            anything is written.
    """
    selected = _subfolder_runners(runners, subfolders)
    path = next(iter(selected.values()))._path(path)
    prepared = prepare_run(selected, names)
    prepared.write_table(path, show=True)
    simulator = prepared.simulator(workers)
    for unit in prepared.units:
        prepared.run_unit(unit, path / unit.folder, simulator)
    _report(
        {
            "subfolders": ", ".join(str(key) for key in prepared.plans),
            "analyses": "; ".join(
                f"{key}: {', '.join(names)}"
                for key, (names, _) in prepared.plans.items()
            ),
        },
        path,
    )
    return path
