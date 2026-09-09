"""Result of optimization."""

import datetime
import logging
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from sbmlsim.console import console
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.options import FitSettings
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
    parameters: list[FitParameter], x: np.ndarray, rtol: float = 0.05
) -> list[str]:
    """Warn about optimal parameters which ended up on their bounds.

    A parameter on its bound means that the optimum is outside of the box in
    which the parameter was allowed to vary.

    The distance to a bound is relative to the interval of the parameter. The
    optimization runs in logarithmic parameter space, so the distance is measured
    there as well whenever the bounds and the value are positive.

    Args:
        parameters: fitted parameters with their bounds.
        x: optimal values of the parameters.
        rtol: relative distance to a bound which is reported.

    Returns:
        Messages for the parameters which are within `rtol` of one of their bounds.
    """
    messages: list[str] = []
    for k, p in enumerate(parameters):
        lb, ub, value = p.lower_bound, p.upper_bound, x[k]
        if not np.isfinite(lb) or not np.isfinite(ub):
            # no relative distance exists on an infinite bound
            continue

        if lb > 0.0 and ub > 0.0 and value > 0.0:
            # the optimization runs in logarithmic space, so does the distance
            lb, ub, value = np.log10(lb), np.log10(ub), np.log10(value)

        span = ub - lb
        if span <= 0.0:
            # the bounds are a single point
            continue

        for bound, name in [(lb, "lower"), (ub, "upper")]:
            if abs(value - bound) / span < rtol:
                messages.append(
                    f"!Optimal parameter '{p.pid}' within {rtol:.0%} of {name} bound!"
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
        """Store fit results as TSV."""
        self.df_fits.to_csv(path, sep="\t", index=False)

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
                )
            )
        return fit_pars

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

    def report(self, path: Path | None = None, print_output: bool = True) -> str:
        """Report of optimization."""
        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)
        info = [
            "\n",
            "-" * 80,
            "-" * 80,
            f"Optimization results: {self.sid}",
            "-" * 80,
            str(self.df_fits),
            "-" * 80,
            "Optimal parameters:",
        ]
        pd.reset_option("display.max_columns")
        pd.reset_option("display.expand_frame_repr")

        xopt = self.xopt
        for msg in bound_warnings(self.parameters, xopt):
            logger.error(msg)
            info.append(f"\t>>> {msg} <<<")

        fitted_pars = {
            p.pid: (xopt[k], p.unit, p.lower_bound, p.upper_bound)
            for k, p in enumerate(self.parameters)
        }

        for key, value in fitted_pars.items():
            info.append(
                f"\t'{key}': Q_({value[0]}, '{value[1]}'),  # [{value[2]} - {value[3]}]"
            )
        info.append("-" * 80)
        info_str: str = "\n".join(info)

        if print_output:
            console.print(info_str)

        if path:
            with open(path, "w", encoding="utf-8") as f_out:
                f_out.write(info_str)

        return info_str
