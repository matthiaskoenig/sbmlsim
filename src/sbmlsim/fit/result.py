"""Result of optimization."""

import datetime
import logging
import uuid
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from sbmlsim.console import console
from sbmlsim.fit.objects import FitParameter, describe_array
from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet, ParameterSets
from sbmlsim.serialization import ObjectJSONEncoder, from_json, to_json

logger = logging.getLogger(__name__)


def fit_id(name: str | None = None) -> str:
    """Create the unique id of a fit from the time and a short hash.

    The id is created when a fit starts and is the id of its optimization
    problem, of its result and of the directory of its report, so that
    everything a fit produces carries the same key and sorts by time.

    Args:
        name: name the id is prefixed with, e.g., the name of the problem.

    Returns:
        `<name>_<date>_<time>__<hash>`, e.g. `PK_20260908_144538__ea1ff`.
    """
    uid = f"{datetime.datetime.now():%Y%m%d_%H%M%S}__{uuid.uuid4().hex[:5]}"
    return f"{name}_{uid}" if name else uid


def bound_warnings(
    parameters: list[FitParameter],
    x: np.ndarray,
    scales: Sequence[ParameterScaleType],
    rtol: float = 0.05,
    groups: Mapping[str, Collection[str]] | None = None,
) -> list[str]:
    """Warn about optimal parameters which ended up on their bounds.

    A parameter on its bound means that the optimum is outside of the box in
    which the parameter was allowed to vary.

    The distance to a bound is relative to the interval of the parameter, in
    the scale the optimizer searched the parameter in.

    Args:
        parameters: fitted parameters with their bounds.
        x: optimal values of the parameters.
        scales: the scale of every parameter, see
            `OptimizationProblem.scales_initialized`.
        rtol: relative distance to a bound which is reported.
        groups: label of a group -> the ids of its elements, e.g. the arrays
            of a network with `ParameterGroup.ids`. A parameter belongs to
            the group which has the entity it writes (`FitParameter.entity_id`)
            as an element. The elements of a group are reported as one
            message which counts them, because a network has hundreds.

    Returns:
        Messages for the parameters which are within `rtol` of one of their
        bounds, and one message per group with such elements.
    """
    grouped: dict[str, str] = {
        sid: label for label, ids in (groups or {}).items() for sid in ids
    }
    messages: list[str] = []
    at_bound: dict[str, set[str]] = {}
    for k, (p, scale) in enumerate(zip(parameters, scales, strict=True)):
        lb, ub, value = p.lower_bound, p.upper_bound, x[k]
        if not np.isfinite(lb) or not np.isfinite(ub):
            # no relative distance exists on an infinite bound
            continue

        if scale.is_log and lb > 0.0 and ub > 0.0 and value > 0.0:
            # the optimization runs in logarithmic space, so does the distance
            lb, ub, value = np.log10(lb), np.log10(ub), np.log10(value)

        span = ub - lb
        if span <= 0.0:
            # the bounds are a single point
            continue

        for bound, name in [(lb, "lower"), (ub, "upper")]:
            if abs(value - bound) / span < rtol:
                if p.entity_id in grouped:
                    # a versioned element counts once
                    at_bound.setdefault(grouped[p.entity_id], set()).add(p.entity_id)
                    continue
                messages.append(
                    f"!Optimal parameter '{p.pid}' within {rtol:.0%} of {name} bound!"
                )
    for label, elements in at_bound.items():
        size = len((groups or {})[label])
        estimated = len(
            {p.entity_id for p in parameters if grouped.get(p.entity_id) == label}
        )
        counts = f"{size} element{'s' if size != 1 else ''}" + (
            f" ({estimated} estimated)" if estimated != size else ""
        )
        messages.append(
            f"!{len(elements)} of the {counts} of '{label}' within "
            f"{rtol:.0%} of a bound!"
        )
    return messages


class OptimizationResult(ObjectJSONEncoder):
    """Result of optimization problem."""

    def __init__(
        self,
        parameters: Iterable[FitParameter],
        fits: list[OptimizeResult],
        trajectories: list[list[float]],
        sid: str | None = None,
        opid: str | None = None,
        settings: FitSettings | dict[str, Any] | None = None,
    ):
        """Initialize optimization result.

        Provides access to the FitParameters, the individual fits, and
        the trajectories of the fits. The settings and the id of the problem
        are stored with the result, a report of the fit needs them.

        Args:
            parameters: fit parameters of the optimization problem.
            fits: results of the single optimizations.
            trajectories: cost of every step of the single optimizations.
            sid: identifier of the result, created from the time by default.
            opid: id of the optimization problem the result belongs to.
            settings: settings the fit was run with.
        """
        super().__init__()
        self.opid = opid
        if isinstance(settings, dict):
            settings = FitSettings.from_dict(settings)
        self.settings: FitSettings | None = settings
        # the id of the fit, which the runner creates before the fit starts
        self.sid = sid if sid else (opid if opid else fit_id())
        self.parameters: list[FitParameter] = []
        for p in parameters:
            if isinstance(p, dict):
                p = FitParameter(**p)
            self.parameters.append(p)

        self.fits: list[OptimizeResult] = []
        for fit in fits:
            if isinstance(fit, dict):
                fit = OptimizeResult(**fit)
            # JSON has no arrays, the parameter vectors are lists after a round trip
            for key in ["x", "x0"]:
                value = fit.get(key)
                if value is not None and not isinstance(value, np.ndarray):
                    fit[key] = np.asarray(value, dtype=float)
            self.fits.append(fit)

        # the cost of every step of a run, which is what the trace plot shows
        self.trajectories: list[list[float]] = [
            [float(cost) for cost in trajectory] for trajectory in trajectories
        ]

        # create data frame from results
        self.df_fits = OptimizationResult.process_fits(self.parameters, self.fits)
        self.df_traces = OptimizationResult.process_traces(self.trajectories)

    def run_result(self, k: int) -> "OptimizationResult":
        """Get the result of a single optimization run.

        The run is a result of its own, so it is stored while a fit runs and
        collected again afterwards, see `write_run` and `from_directory`.

        The runs are indexed in the order they ran, which pairs a fit with its
        trajectory; `parameter_set` indexes them by increasing cost instead.

        Args:
            k: index of the run in `fits`, i.e., in the order they ran.

        Returns:
            An `OptimizationResult` with this run only.
        """
        return OptimizationResult(
            parameters=self.parameters,
            fits=[self.fits[k]],
            trajectories=[self.trajectories[k]] if k < len(self.trajectories) else [[]],
            sid=f"{self.sid}_{k}",
            opid=self.opid,
            settings=self.settings,
        )

    @staticmethod
    def write_run(
        directory: Path,
        parameters: Iterable[FitParameter],
        fit: OptimizeResult,
        trajectory: list[float],
        sid: str,
        opid: str | None = None,
        settings: FitSettings | None = None,
    ) -> Path:
        """Store a single optimization run as JSON.

        The runs are stored while the fit runs, so a fit which is interrupted,
        times out or crashes leaves the runs which finished.

        Args:
            directory: directory of the runs, created if it does not exist.
            parameters: fit parameters of the problem.
            fit: result of the single optimization.
            trajectory: trajectory of the single optimization.
            sid: id of the run, the name of its file.
            opid: id of the optimization problem.
            settings: settings of the fit.

        Returns:
            Path of the file the run was written to.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sid}.json"
        OptimizationResult(
            parameters=parameters,
            fits=[fit],
            trajectories=[trajectory],
            sid=sid,
            opid=opid,
            settings=settings,
        ).to_json(path=path)
        return path

    @staticmethod
    def from_directory(directory: Path, sid: str | None = None) -> "OptimizationResult":
        """Collect the optimization runs of a directory.

        This is the counterpart of `write_run`: the runs a fit stored are read
        back and combined, which recovers the results of a fit which did not
        finish.

        Args:
            directory: directory with the JSON files of the runs.
            sid: id of the combined result, the name of the directory by default.

        Returns:
            The combined result of all runs in the directory.

        Raises:
            ValueError: if the directory holds no run.
        """
        paths = sorted(directory.glob("*.json"))
        results = [OptimizationResult.from_json(path) for path in paths]
        if not results:
            raise ValueError(f"No optimization run in '{directory}'.")

        combined = OptimizationResult.combine(results)
        combined.sid = sid if sid else directory.name
        return combined

    def to_tsv(self, path: Path) -> None:
        """Store fit results as TSV, one line per run.

        The columns named by the parameter ids are the fitted values, the columns
        `x0.<pid>` the start values of the run (`.` is no character of an id, so
        the names cannot collide). The vectors `x` and `x0` of `df_fits` are not
        written as such, their text would span several lines and be rounded.
        """
        df = self.df_fits.drop(columns=["x", "x0"])
        pids = [p.pid for p in self.parameters]
        x0 = pd.DataFrame(
            [
                np.full(len(pids), np.nan) if x is None else np.asarray(x, dtype=float)
                for x in self.df_fits.x0
            ],
            columns=pd.Index([f"x0.{pid}" for pid in pids]),
        )
        df = pd.concat([df, x0], axis=1)
        # a message of an error can span several lines
        df["message"] = [
            m if not isinstance(m, str) else " ".join(m.split()) for m in df.message
        ]
        df.to_csv(path, sep="\t", index=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {}
        for key in ["sid", "opid", "parameters", "fits", "trajectories"]:
            d[key] = self.__dict__[key]
        d["settings"] = self.settings.to_dict() if self.settings else None
        return d

    def to_json(self, path: Path | None = None) -> str | Path:
        """Store OptimizationResult as json.

        Uses the to_dict method.
        """
        return to_json(object=self, path=path)

    @staticmethod
    def from_json(json_info: str | Path) -> "OptimizationResult":
        """Load OptimizationResult from Path or str.

        :param json_info:
        :return:
        """
        d = from_json(json_info)
        return OptimizationResult(**d)

    def __str__(self) -> str:
        """Get string representation."""
        return f"<OptimizationResult: n={self.size}>"

    @property
    def settings_stored(self) -> FitSettings:
        """Settings the fit was run with.

        Raises:
            ValueError: if the result carries no settings, e.g., because it was
                created by hand.
        """
        if self.settings is None:
            raise ValueError(
                f"OptimizationResult '{self.sid}' carries no FitSettings, they are "
                f"required to report the fit."
            )
        return self.settings

    def parameter_set(self, k: int = 0, sid: str | None = None) -> ParameterSet:
        """Get the parameters of a single optimization run.

        The runs are ordered by increasing cost, so `k=0` is the best fit.

        Args:
            k: index of the run in the results ordered by cost.
            sid: identifier of the set, `<result id>_<k>` by default.

        Returns:
            The parameter set of the run.

        Raises:
            IndexError: if the result has no run `k`.
        """
        if k >= self.size:
            raise IndexError(
                f"OptimizationResult '{self.sid}' has '{self.size}' runs, no run '{k}'."
            )
        row = self.df_fits.iloc[k]
        return ParameterSet.from_fit_parameters(
            parameters=self.parameters,
            x=row.x,
            sid=sid if sid else f"{self.sid}_{k}",
            cost=float(row.cost),
            provenance=f"optimization '{self.opid}', run '{int(row.run)}'",
        )

    def parameter_sets(self, size: int = 1) -> ParameterSets:
        """Get the parameters of the best optimization runs.

        Args:
            size: number of runs, ordered by increasing cost.

        Returns:
            The parameter sets of the best runs.
        """
        return ParameterSets(
            [self.parameter_set(k=k) for k in range(min(size, self.size))]
        )

    @staticmethod
    def combine(opt_results: list["OptimizationResult"]) -> "OptimizationResult":
        """Combine results from multiple parameter fitting experiments.

        Raises:
            ValueError: if no results are given.
        """
        # FIXME: check that the parameters are fitting
        if not opt_results:
            raise ValueError("No OptimizationResults to combine.")
        parameters = opt_results[0].parameters
        pids = {p.pid for p in parameters}

        fits = []
        trajectories = []
        for opt_res in opt_results:
            pids_next = {p.pid for p in opt_res.parameters}
            if pids != pids_next:
                logger.error(
                    "Parameters of OptimizationResults do not match: %s != %s",
                    pids,
                    pids_next,
                )

            fits.extend(opt_res.fits)
            trajectories.extend(opt_res.trajectories)
        return OptimizationResult(
            parameters=parameters,
            fits=fits,
            trajectories=trajectories,
            opid=opt_results[0].opid,
            settings=opt_results[0].settings,
        )

    @property
    def size(self) -> int:
        """Get number of optimization runs in result."""
        return len(self.df_fits)

    @property
    def xopt(self) -> np.ndarray:
        """Numerical values of optimal parameters."""
        values: np.ndarray = self.df_fits.x.iloc[0]
        return values

    @property
    def xopt_fit_parameters(self) -> list[FitParameter]:
        """Optimal parameters as Fit parameters."""
        return self._x_as_fit_parameters(x=self.xopt)

    def _x_as_fit_parameters(self, x: np.ndarray) -> list[FitParameter]:
        """Convert numerical parameter vector to fit parameters."""
        fit_pars = []
        for k, p in enumerate(self.parameters):
            fit_pars.append(
                FitParameter(
                    pid=p.pid,
                    start_value=x[k],
                    lower_bound=p.lower_bound,
                    upper_bound=p.upper_bound,
                    unit=p.unit,
                    target=p.target,
                    scale=p.scale,
                    mappings=p.mappings,
                )
            )
        return fit_pars

    @property
    def scales(self) -> list[ParameterScaleType]:
        """Get the scale the optimizer searched every parameter in.

        It is the `FitParameter.scale` of a parameter and the
        `parameter_scale` of the settings, of the default settings for a
        result without them, for a parameter without one.
        """
        settings = self.settings if self.settings is not None else FitSettings()
        return [
            settings.parameter_scale if p.scale is None else p.scale
            for p in self.parameters
        ]

    @staticmethod
    def process_traces(trajectories: list[list[float]]) -> pd.DataFrame:
        """Process the trajectories of the optimizations.

        A trajectory is the cost of every step of a run, which is what the
        trace plot of a report shows.

        Args:
            trajectories: cost of every step, per run.

        Returns:
            DataFrame with the columns `run`, `step` and `cost`.
        """
        return pd.DataFrame(
            [
                {"run": kt, "step": step, "cost": cost}
                for kt, trajectory in enumerate(trajectories)
                for step, cost in enumerate(trajectory)
            ],
            columns=pd.Index(["run", "step", "cost"]),
        )

    @staticmethod
    def process_fits(
        parameters: list[FitParameter], fits: list[OptimizeResult]
    ) -> pd.DataFrame:
        """Process the optimization results, sorted by increasing cost."""
        results = []
        pids = [p.pid for p in parameters]
        for kf, fit in enumerate(fits):
            res = {
                "run": kf,
                # 'status': fit.status,
                "success": fit.success,
                "duration": fit.duration,
                "cost": fit.cost,
                # 'optimality': fit.optimality,
            }
            # add parameter columns; a run which failed before it started
            # has no parameters
            for k, pid in enumerate(pids):
                res[pid] = np.nan if fit.x is None else fit.x[k]
            res["message"] = fit.message if hasattr(fit, "message") else None
            res["x"] = fit.x
            res["x0"] = fit.x0

            results.append(res)
        df = pd.DataFrame(results)
        return df.sort_values(by=["cost"]).reset_index(drop=True)

    def report(
        self,
        path: Path | None = None,
        print_output: bool = True,
        groups: Mapping[str, Collection[str]] | None = None,
    ) -> str:
        """Report of optimization.

        Args:
            path: file the report is written to, none by default.
            print_output: print the report.
            groups: the groups of elements which are reported as one, e.g. the
                arrays of a network, see `bound_warnings`. The optimal
                parameters are listed as one line per group.
        """
        # the elements of a group are listed as one line per group, in the
        # table of the runs as well
        groups = groups or {}
        label_of: dict[str, str] = {
            sid: label for label, ids in groups.items() for sid in ids
        }
        by_label: dict[str, list[int]] = {label: [] for label in groups}
        for k, p in enumerate(self.parameters):
            if p.entity_id in label_of:
                by_label[label_of[p.entity_id]].append(k)
        grouped = {self.parameters[k].pid for ks in by_label.values() for k in ks}
        fits = self.df_fits
        if grouped:
            fits = fits.drop(columns=[*grouped, "x", "x0"], errors="ignore")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)
        info = [
            "\n",
            "-" * 80,
            "-" * 80,
            f"Optimization results: {self.sid}",
            "-" * 80,
            str(fits),
            "-" * 80,
            "Optimal parameters:",
        ]
        pd.reset_option("display.max_columns")
        pd.reset_option("display.expand_frame_repr")

        xopt = self.xopt
        for msg in bound_warnings(self.parameters, xopt, self.scales, groups=groups):
            logger.error(msg)
            info.append(f"\t>>> {msg} <<<")

        for k, p in enumerate(self.parameters):
            if p.pid not in grouped:
                info.append(
                    f"\t'{p.pid}': Q({xopt[k]}, '{p.unit}'),  "
                    f"# [{p.lower_bound} - {p.upper_bound}]"
                )
        for label, ks in by_label.items():
            info.append(
                "\t"
                + describe_array(
                    label,
                    len(groups[label]),
                    [self.parameters[k] for k in ks],
                    [xopt[k] for k in ks],
                )
            )
        info.append("-" * 80)
        info_str: str = "\n".join(info)

        if print_output:
            console.print(info_str)

        if path:
            with open(path, "w", encoding="utf-8") as f_out:
                f_out.write(info_str)

        return info_str
