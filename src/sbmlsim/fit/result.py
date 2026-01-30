"""Result of optimization."""

import datetime
import uuid
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from pymetadata import log
from pymetadata.console import console
from scipy.optimize import OptimizeResult

from sbmlsim.fit.objects import FitParameter
from sbmlsim.serialization import ObjectJSONEncoder, from_json, to_json


logger = log.get_logger(__name__)


class OptimizationResult(ObjectJSONEncoder):
    """Result of optimization problem."""

    def __init__(
        self,
        parameters: Iterable[FitParameter],
        fits: list[OptimizeResult],
        trajectories: list,
        sid: str = None,
    ):
        """Initialize optimization result.

        Provides access to the FitParameters, the individual fits, and
        the trajectories of the fits.

        # FIXME: store for which problem

        :param parameters:
        :param fits:
        :param trajectories:
        """
        super(OptimizationResult, self).__init__()
        if sid:
            self.sid = sid
        else:
            uuid_str = str(uuid.uuid4())
            self.sid = (
                "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now()) + f"__{uuid_str[:5]}"
            )
        self.parameters: list[FitParameter] = []
        for p in parameters:
            if isinstance(p, dict):
                p = FitParameter(**p)
            self.parameters.append(p)

        self.fits: list[OptimizeResult] = []
        for fit in fits:
            if isinstance(fit, dict):
                fit = OptimizeResult(**fit)
            self.fits.append(fit)

        self.trajectories = trajectories

        # create data frame from results
        self.df_fits = OptimizationResult.process_fits(self.parameters, self.fits)
        self.df_traces = OptimizationResult.process_traces(
            self.parameters, self.trajectories
        )

    def to_tsv(self, path: Path):
        """Store fit results as TSV."""
        self.df_fits.to_csv(path, sep="\t", index=False)

    def to_dict(self):
        """Convert to dictionary."""
        d = dict()
        for key in ["sid", "parameters", "fits", "trajectories"]:
            d[key] = self.__dict__[key]
        return d

    def to_json(self, path: Optional[Path] = None) -> Union[str, Path]:
        """Store OptimizationResult as json.

        Uses the to_dict method.
        """
        return to_json(object=self, path=path)

    @staticmethod
    def from_json(json_info: Union[str, Path]) -> "OptimizationResult":
        """Load OptimizationResult from Path or str.

        :param json_info:
        :return:
        """
        d = from_json(json_info)
        return OptimizationResult(**d)

    def __str__(self) -> str:
        """Get string representation."""
        info = f"<OptimizationResult: n={self.size}>"
        return info

    @staticmethod
    def combine(opt_results: list["OptimizationResult"]) -> "OptimizationResult":
        """Combine results from multiple parameter fitting experiments."""
        # FIXME: check that the parameters are fitting
        parameters = opt_results[0].parameters
        pids = {p.pid for p in parameters}

        fits = []
        trajectories = []
        for opt_res in opt_results:
            pids_next = {p.pid for p in opt_res.parameters}
            if pids != pids_next:
                logger.error(
                    f"Parameters of OptimizationResults do not match: "
                    f"{pids} != {pids_next}"
                )

            fits.extend(opt_res.fits)
            trajectories.extend(opt_res.trajectories)
        return OptimizationResult(
            parameters=parameters, fits=fits, trajectories=trajectories
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

    def _x_as_fit_parameters(self, x) -> list[FitParameter]:
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
    def process_traces(parameters: list[FitParameter], trajectories):
        """Process the optimization results."""
        results = []
        pids = [p.pid for p in parameters]
        for kt, trajectory in enumerate(trajectories):
            for step in trajectory:
                res = {
                    "run": kt,
                    "cost": step[1],
                }
                # add parameter columns
                for k, pid in enumerate(pids):
                    res[pid] = step[0][k]
                results.append(res)
        df = pd.DataFrame(results)
        return df

    @staticmethod
    def process_fits(parameters: list[FitParameter], fits: list[OptimizeResult]):
        """Process the optimization results."""
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
            # add parameter columns
            for k, pid in enumerate(pids):
                res[pid] = fit.x[k]
            res["message"] = fit.message if hasattr(fit, "message") else None
            res["x"] = fit.x
            res["x0"] = fit.x0

            results.append(res)
        df = pd.DataFrame(results)
        df.sort_values(by=["cost"], inplace=True)
        # reindex
        df.index = range(len(df))

        return df

    def report(self, path: Optional[Path] = None, print_output: bool = True) -> str:
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
        fitted_pars = {}
        for k, p in enumerate(self.parameters):
            opt_value = xopt[k]
            if abs(opt_value - p.lower_bound) / p.lower_bound < 0.05:
                msg = f"!Optimal parameter '{p.pid}' within 5% of lower bound!"
                logger.error(msg)
                info.append(f"\t>>> {msg} <<<")

            if abs(opt_value - p.upper_bound) / p.upper_bound < 0.05:
                msg = f"!Optimal parameter '{p.pid}' within 5% of upper bound!"
                logger.error(msg)
                info.append(f"\t>>> {msg} <<<")
            fitted_pars[p.pid] = (opt_value, p.unit, p.lower_bound, p.upper_bound)

        for key, value in fitted_pars.items():
            info.append(
                "\t'{}': Q_({}, '{}'),  # [{} - {}]".format(
                    key, value[0], value[1], value[2], value[3]
                )
            )
        info.append("-" * 80)
        info_str: str = "\n".join(info)

        if print_output:
            console.print(info_str)

        if path:
            with open(path, "w") as f_out:
                f_out.write(info_str)

        return info_str
