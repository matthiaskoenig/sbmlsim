"""The records of the adopted fits of a project: the fitted values and their uncertainty.

A fit becomes part of a model in two steps:

1. **Adopt** the fit: the `fit` kind of `sbmlsim.run` with `adopt` copies the
   record of the finished fit (its parameters, the profile likelihood and the
   Fisher information, its report) into the directory `fits/<fit id>` of its
   output and writes `adopted.json` there, the adopted fits of the project
   with the fit adopted for its subset (`adopt_fit`). A person commits both
   into the project (`Project.fits_dir`, `Project.adopted_file`).
2. **Use** it in the factory: a model file takes the value of a fitted
   parameter from the record, together with its uncertainty, see
   `AdoptedRecords.parameter`:

   ```python
   # not executed
   FITS = PROJECT.fit_records()
   Parameter("k", **FITS.parameter("k", unit=U.per_min), name="rate")
   ```

   The uncertainty is a distrib `Uncertainty` on the SBML parameter: the
   confidence interval of the profile likelihood (the Fisher information when
   the fit has no profile), the standard error of the Fisher information, the
   distribution the draws of the parameter come from and the provenance in its
   notes, see `ParameterUncertainty`. The distributions are read back from the
   SBML by `sbmlsim.simulation.sampling.parameter_distributions`.

distrib states one distribution per parameter, the dependence of the
parameters of a fit stays in its record: `AdoptedRecord.correlation` is the
correlation of the normal scores of its parameters, of the Laplace posterior
(`posterior.json`) or else of a Fisher information of full rank, see
`parameter_correlation`, which the draws of the experiments
(`sbmlsim.experiment.draws.UncertaintyRun`) and of the credibility
(`sbmlsim.credibility.uncertainty`) take as the correlation of a Gaussian
copula of the distributions of the SBML.

The records of one fit which its report is built from are
`sbmlsim.fit.cli.FitRecords` (`load_fit_records`); an `AdoptedRecord` is the
fitted parameters of an adopted fit with their uncertainty.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import libsbml
import numpy as np
from scipy import stats

from sbmlsim.fit.fisher import FisherInformation
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.posterior import POSTERIOR_FILE, ParameterPosterior
from sbmlsim.units import ureg

if TYPE_CHECKING:
    from sbmlutils.factory import Uncertainty
    from sbmlutils.factory.units import UnitType

__all__ = [
    "RECORD_FILES",
    "REPORT_FIGURE",
    "REPORT_SOURCES",
    "REQUIRED_FILES",
    "UNCERTAINTY_FILES",
    "AdoptedRecord",
    "AdoptedRecords",
    "ParameterCorrelation",
    "ParameterUncertainty",
    "UncertaintyMethod",
    "adopt_fit",
    "convert_value",
    "parameter_correlation",
]

logger = logging.getLogger(__name__)

#: the files of a fit which an adopted fit keeps; the first ones are required
RECORD_FILES: tuple[str, ...] = (
    "parameters.json",
    "optimization_result.json",
    "identifiability.json",
    "identifiability.tsv",
    "fisher.json",
    "fisher.tsv",
    "posterior.json",
    "posterior.tsv",
    "metrics.tsv",
    "metrics_mappings.tsv",
    "report.txt",
    "report.pdf",
    "report.typ",
    "references.bib",
    "references.csl",
)
#: the figures of the plots which the Typst source of a report shows
REPORT_FIGURE = re.compile(r"plots/[\w.\-]+\.(?:png|svg|pdf)")
#: the source of the report which refers to figures; the LaTeX sources and the
#: PNG copies of the figures they show stay in the results of the fit
REPORT_SOURCES: tuple[str, ...] = ("report.typ",)
#: the files without which a fit cannot be adopted
REQUIRED_FILES: tuple[str, ...] = ("parameters.json", "optimization_result.json")
#: one of these files gives the uncertainty of the parameters
UNCERTAINTY_FILES: tuple[str, ...] = ("identifiability.json", "fisher.json")


class UncertaintyMethod(StrEnum):
    """Where the uncertainty of a fitted parameter comes from."""

    PROFILE = "profile likelihood"
    FISHER = "Fisher information"


# ----------------------------------------------------------------------------
# adopting a fit
# ----------------------------------------------------------------------------
def adopt_fit(
    fit_dir: Path, fits_dir: Path, *, force: bool = False, fit_id: str | None = None
) -> Path:
    """Copy the record of a finished fit into the folder of the fits of a model.

    The record is the part of the results of a fit which the model needs: the
    parameters, the settings and the problem (`optimization_result.json`), the
    profile likelihood and the Fisher information with their tables, the
    metrics, the text report and the methods report as PDF and Typst source
    with its references (`RECORD_FILES`) and the figures the Typst source
    shows, in `plots/`, so that `report.typ` compiles in the record. The report
    is optional, a record of an older fit without it is adopted as well. The
    runs, the other figures, the LaTeX sources with the PNG copies of their
    figures and the HTML report stay in the results of the fit.

    The record is copied into a directory of a unique name next to its place,
    created with the permissions of the umask, and renamed into it, so a
    reader sees no record or a complete one; a record it replaces is moved
    aside first and restored if the rename fails.

    Args:
        fit_dir: the directory of the results of the fit, named by its id.
        fits_dir: the folder of the adopted fits of the model.
        force: replace a record of the same id.
        fit_id: the id of the fit, the name of `fit_dir` without.

    Returns:
        The directory of the record.

    Raises:
        FileNotFoundError: if the fit does not exist, lacks a required file or
            has neither a profile likelihood nor a Fisher information.
        FileExistsError: if the record exists and `force` is not set.
    """
    if not fit_dir.is_dir():
        raise FileNotFoundError(f"The fit '{fit_dir}' does not exist.")
    missing = [name for name in REQUIRED_FILES if not (fit_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"The fit '{fit_dir.name}' has no {missing}.")
    if not any((fit_dir / name).exists() for name in UNCERTAINTY_FILES):
        raise FileNotFoundError(
            f"The fit '{fit_dir.name}' has neither a profile likelihood nor a "
            f"Fisher information ({list(UNCERTAINTY_FILES)}): run the fit with "
            f"its identifiability."
        )
    fit_id = fit_dir.name if fit_id is None else fit_id
    target = fits_dir / fit_id
    if target.exists() and not force:
        raise FileExistsError(f"The fit '{fit_id}' is adopted already: {target}.")
    fits_dir.mkdir(parents=True, exist_ok=True)
    # a directory of a unique name, created with the permissions of the umask
    temporary = fits_dir / f".{fit_id}.{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for name in RECORD_FILES:
            source = fit_dir / name
            if source.exists():
                shutil.copy2(source, temporary / name)
        for figure in _report_figures(fit_dir):
            (temporary / figure).parent.mkdir(exist_ok=True)
            shutil.copy2(fit_dir / figure, temporary / figure)
        if target.exists():
            # the old record is moved aside first, a rename does not replace a
            # directory which is not empty; it comes back if the new one fails
            old = fits_dir / f".{fit_id}.old.{uuid.uuid4().hex}"
            target.replace(old)
            try:
                temporary.replace(target)
            except OSError:
                old.replace(target)
                raise
            shutil.rmtree(old)
        else:
            temporary.replace(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def _report_figures(fit_dir: Path) -> list[str]:
    """Get the figures, relative to the fit, which the sources of the report show.

    Args:
        fit_dir: the directory of the results of the fit.

    Returns:
        The paths (`plots/<name>.svg`) which are referenced by `report.typ`
        and exist, sorted.
    """
    found: set[str] = set()
    for name in REPORT_SOURCES:
        source = fit_dir / name
        if source.exists():
            found.update(REPORT_FIGURE.findall(source.read_text(encoding="utf-8")))
    return sorted(figure for figure in found if (fit_dir / figure).is_file())


# ----------------------------------------------------------------------------
# the uncertainty of a parameter
# ----------------------------------------------------------------------------
def _number(value: float) -> str:
    """Get a number as the shortest text which reads back as the same float."""
    return repr(float(value))


@dataclass(frozen=True, kw_only=True)
class ParameterUncertainty:
    """The fitted value of a parameter and its uncertainty, from the record of a fit.

    All numbers are in the unit of the model (`unit`); the scale is the space
    the fit searched the parameter in.

    Attributes:
        pid: the id of the parameter in the flat model.
        value: the fitted value.
        unit: the unit of the model, as the fit wrote it.
        lower_bound: the lower bound of the fit.
        upper_bound: the upper bound of the fit.
        scale: the space the fit searched the parameter in.
        alpha: the confidence level of the intervals.
        z: the quantile of the standard normal of the confidence level of the
            profile, `sqrt(chi2(alpha, df))`.
        method: where the confidence interval and the distribution come from.
        ci_lower: the lower end of the confidence interval, `None` if it is
            open (the profile reaches the bound).
        ci_upper: the upper end of the confidence interval, `None` if open.
        identifiability: the class of the profile likelihood, `None` without.
        standard_error: the standard error of the Fisher information in the
            space of the fit (`scale`), `None` without.
        fisher_ci: the confidence interval of the Fisher information.
        fit_id: the id of the fit.
        cost: the cost of the fit.
        n_data: the number of data points of the fit, `None` if unknown.
        record: the path of the record, relative to the folder of the fits.
    """

    pid: str
    value: float
    unit: str | None
    lower_bound: float
    upper_bound: float
    scale: ParameterScaleType
    alpha: float
    z: float
    method: UncertaintyMethod
    ci_lower: float | None
    ci_upper: float | None
    identifiability: str | None = None
    standard_error: float | None = None
    fisher_ci: tuple[float, float] | None = None
    fit_id: str
    cost: float
    n_data: int | None = None
    record: str

    def _to_scale(self, x: float) -> float:
        """Get a value in the space of the fit."""
        return float(self.scale.to_scale(x))

    def sigma(self) -> float | None:
        """Get the standard deviation of the distribution in the space of the fit.

        From the profile it is the mean of the closed half widths of the
        confidence interval divided by `z`; from the Fisher information its
        standard error. `None` if neither side of the profile is closed.
        """
        if self.method == UncertaintyMethod.FISHER:
            return self.standard_error
        center = self._to_scale(self.value)
        widths = [
            abs(center - self._to_scale(bound))
            for bound in (self.ci_lower, self.ci_upper)
            if bound is not None
        ]
        widths = [w for w in widths if w > 0.0]
        if not widths:
            return None
        return float(np.mean(widths)) / self.z

    def interval(self) -> tuple[float, float]:
        """Get the confidence interval, an open side at the bound of the fit."""
        lower = self.lower_bound if self.ci_lower is None else self.ci_lower
        upper = self.upper_bound if self.ci_upper is None else self.ci_upper
        return lower, upper

    def standard_error_model(self) -> float | None:
        """Get the standard error in the unit of the model (delta method)."""
        if self.standard_error is None:
            return None
        if self.scale is ParameterScaleType.LOG10:
            return self.value * math.log(10) * self.standard_error
        if self.scale is ParameterScaleType.LOG:
            return self.value * self.standard_error
        return self.standard_error

    def distribution(self) -> str:
        """Get the distribution of the draws as a formula of distrib.

        A standard deviation which covers the bounds of the fit says that the
        data do not constrain the parameter: the truncated distribution is flat
        over the bounds, so it is written as the uniform it is. The Fisher
        information of a rank deficient problem gives such a standard error for
        the directions it does not constrain, and the lognormal of one is not a
        number any sampler can draw from.

        Returns:
            On a logarithmic scale `lognormal(ln(value), sigma, lb, ub)` with
            sigma in natural logarithms, on the linear scale `normal(value,
            sigma, lb, ub)`, both truncated to the bounds of the fit; without a
            closed side of the profile, and for a standard deviation which
            covers the bounds, `uniform(lb, ub)`.
        """
        lb, ub = _number(self.lower_bound), _number(self.upper_bound)
        sigma = self.sigma()
        if sigma is None or not sigma > 0.0:
            return f"uniform({lb}, {ub})"
        if self.scale.is_log:
            sigma_ln = (
                sigma * math.log(10)
                if self.scale is ParameterScaleType.LOG10
                else sigma
            )
            span = (
                math.log(self.upper_bound / self.lower_bound)
                if self.lower_bound > 0.0 and self.upper_bound > 0.0
                else math.inf
            )
            if sigma_ln >= span:
                return f"uniform({lb}, {ub})"
            mu = math.log(self.value)
            return f"lognormal({_number(mu)}, {_number(sigma_ln)}, {lb}, {ub})"
        if sigma >= self.upper_bound - self.lower_bound:
            return f"uniform({lb}, {ub})"
        return f"normal({_number(self.value)}, {_number(sigma)}, {lb}, {ub})"

    def notes(self) -> str:
        """Get the provenance of the value and of its uncertainty as markdown."""
        lower, upper = self.interval()
        rows = [
            ("fit", f"`{self.fit_id}`"),
            ("record", f"`{self.record}`"),
            ("cost", f"{self.cost:.6g}"),
            ("data points", "" if self.n_data is None else str(self.n_data)),
            ("value", f"{self.value:.6g} {self.unit or ''}".strip()),
            ("bounds", f"[{self.lower_bound:.6g}, {self.upper_bound:.6g}]"),
            ("scale", self.scale.name),
            ("method", self.method.value),
            ("identifiability", self.identifiability or ""),
            (
                f"{100 * self.alpha:g} % confidence interval",
                f"[{lower:.6g}, {upper:.6g}]"
                + (
                    " (open sides at the bounds)"
                    if self.ci_lower is None or self.ci_upper is None
                    else ""
                ),
            ),
        ]
        if self.fisher_ci is not None:
            rows.append(
                (
                    "Fisher confidence interval",
                    f"[{self.fisher_ci[0]:.6g}, {self.fisher_ci[1]:.6g}]",
                )
            )
        rows.append(("distribution of the draws", f"`{self.distribution()}`"))
        table = "\n".join(f"| {key} | {value} |" for key, value in rows)
        return (
            f"The value of `{self.pid}` is the result of the fit `{self.fit_id}`, "
            f"its uncertainty is the {self.method.value} of the fit.\n\n"
            f"| | |\n|---|---|\n{table}\n"
        )

    def uncertainty(self, unit: UnitType = None) -> Uncertainty:
        """Get the distrib uncertainty of the parameter.

        Args:
            unit: the unit of the values, the unit of the parameter in the
                model definition (`sbmlutils.factory` unit).

        Returns:
            The uncertainty: the confidence interval, the standard error of
            the Fisher information if there is one and the distribution of the
            draws, the provenance in its notes.
        """
        from sbmlutils.factory import Uncertainty, UncertParameter, UncertSpan

        lower, upper = self.interval()
        children: list[UncertParameter | UncertSpan] = [
            UncertSpan(
                type=libsbml.DISTRIB_UNCERTTYPE_CONFIDENCEINTERVAL,
                valueLower=lower,
                valueUpper=upper,
                unit=unit,
                name=f"{100 * self.alpha:g} % confidence interval ({self.method.value})",
            )
        ]
        se = self.standard_error_model()
        if se is not None and math.isfinite(se):
            children.append(
                UncertParameter(
                    type=libsbml.DISTRIB_UNCERTTYPE_STANDARDERROR,
                    value=se,
                    unit=unit,
                    name="standard error (Fisher information)",
                )
            )
        children.append(
            UncertParameter(
                type=libsbml.DISTRIB_UNCERTTYPE_DISTRIBUTION,
                math=self.distribution(),
                name="distribution of the draws",
            )
        )
        return Uncertainty(
            uncertParameters=children,
            name=f"fit {self.fit_id}",
            notes=self.notes(),
        )


# ----------------------------------------------------------------------------
# the records of the fits
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterCorrelation:
    """The dependence of the fitted parameters of a fit.

    Attributes:
        pids: the parameters, in the order of the rows of `matrix`.
        matrix: the correlation of their normal scores in the space of the
            fit, `None` for independent parameters.
        source: where the correlation comes from, or why there is none.
    """

    pids: tuple[str, ...]
    matrix: np.ndarray | None = field(repr=False)
    source: str

    @property
    def correlated(self) -> bool:
        """Whether the parameters have a correlation."""
        return self.matrix is not None

    def rank_correlation(self) -> np.ndarray:
        """Get the Spearman rank correlation of the parameters.

        The rank correlation of the normal scores, `6 / π asin(r / 2)`, which
        the designs `lhs` and `random` of `sbmlsim.simulation.sampling` take as
        `correlation`; the identity for independent parameters.
        """
        if self.matrix is None:
            return np.eye(len(self.pids))
        rank = 6.0 / np.pi * np.arcsin(np.clip(self.matrix, -1.0, 1.0) / 2.0)
        np.fill_diagonal(rank, 1.0)
        return (rank + rank.T) / 2.0


def parameter_correlation(
    pids: Sequence[str],
    *,
    posterior: ParameterPosterior | None = None,
    fisher: FisherInformation | None = None,
) -> ParameterCorrelation:
    """Get the correlation of the normal scores of fitted parameters.

    The correlation of the Laplace posterior, which constrains every
    direction (`sbmlsim.fit.posterior.posterior_laplace` raises otherwise),
    else of a Fisher information of full rank, in the space of the fit; for a
    normal or log-normal marginal in that space it is the correlation of the
    multivariate normal of the analysis. A Fisher information which is not of
    full rank, an analysis of other parameters or none gives independent
    parameters.

    Args:
        pids: the parameters.
        posterior: the Laplace posterior of the fit, `None` without.
        fisher: the Fisher information of the fit, `None` without.

    Returns:
        The correlation, with where it comes from.
    """
    pids = tuple(pids)
    if len(pids) < 2:
        return ParameterCorrelation(pids, None, "nothing to correlate")
    if posterior is not None and set(pids) <= set(posterior.pids):
        matrix = posterior.correlation.loc[list(pids), list(pids)]
        source = "the correlation of the Laplace posterior in the space of the fit"
    elif fisher is None:
        return ParameterCorrelation(
            pids,
            None,
            "independent, the fit has no posterior and no Fisher information",
        )
    elif not fisher.is_identifiable:
        return ParameterCorrelation(
            pids,
            None,
            f"independent, the Fisher information has rank {fisher.rank} of {fisher.k}",
        )
    elif not set(pids) <= set(fisher.pids):
        return ParameterCorrelation(
            pids, None, "independent, the Fisher information has other parameters"
        )
    else:
        matrix = fisher.correlation.loc[list(pids), list(pids)]
        source = "the correlation of the Fisher information in the space of the fit"
    values = matrix.to_numpy(dtype=float)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 1.0)
    return ParameterCorrelation(pids, values, source)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def convert_value(value: float, unit: str | None, target: str | None) -> float:
    """Convert a number from one unit into another, both units of pint.

    Args:
        value: the number in `unit`.
        unit: the unit of the number, `None` for the unit of the target.
        target: the unit to convert into, `None` for the unit of the number.

    Returns:
        The number in `target`, the number itself if a unit is missing or
        both are the same.
    """
    if unit is None or target is None or unit == target:
        return float(value)
    return float(ureg.Quantity(value, unit).to(target).magnitude)


def _convert_optional(
    value: float | None, unit: str | None, target: str | None
) -> float | None:
    """Convert a number which may be missing, see `convert_value`."""
    return None if value is None else convert_value(value, unit, target)


def _scale(name: str | None, default: ParameterScaleType) -> ParameterScaleType:
    if name is None:
        return default
    return ParameterScaleType[str(name)]


@dataclass(frozen=True)
class AdoptedRecord:
    """The record of an adopted fit: the fitted parameters with their uncertainty.

    Attributes:
        directory: the directory of the record, named by the id of the fit.
        parameters: the uncertainty of every fitted parameter by pid.
        fisher: the Fisher information of the fit, `fisher.json`.
        posterior: the Laplace posterior of the fit, `posterior.json`.
    """

    directory: Path
    parameters: Mapping[str, ParameterUncertainty] = field(repr=False)
    fisher: FisherInformation | None = field(default=None, repr=False)
    posterior: ParameterPosterior | None = field(default=None, repr=False)

    def correlation(self, pids: Sequence[str] | None = None) -> ParameterCorrelation:
        """Get the correlation of the normal scores of fitted parameters.

        Args:
            pids: the parameters, every fitted parameter without.

        Returns:
            The correlation of the posterior or of the Fisher information,
            see `parameter_correlation`.

        Raises:
            KeyError: for a parameter which the fit does not fit.
        """
        pids = tuple(self.parameters) if pids is None else tuple(pids)
        unknown = [pid for pid in pids if pid not in self.parameters]
        if unknown:
            raise KeyError(
                f"The fit '{self.fit_id}' does not fit {unknown}, it fits "
                f"{sorted(self.parameters)}."
            )
        return parameter_correlation(pids, posterior=self.posterior, fisher=self.fisher)

    @property
    def fit_id(self) -> str:
        """The id of the fit."""
        return self.directory.name

    @classmethod
    def load(cls, directory: Path) -> AdoptedRecord:
        """Read the record of an adopted fit.

        The value of a parameter is the best parameter set of the fit
        (`parameters.json`), the bounds and the scale are those of the problem
        (`optimization_result.json`); the confidence interval is the one of the
        profile likelihood (`identifiability.json`), the Fisher information
        (`fisher.json`) gives the standard error and is the fallback for a
        parameter without a profile. The Laplace posterior (`posterior.json`)
        gives the correlation of the parameters, see `correlation`.

        Args:
            directory: the directory of the record.

        Returns:
            The record.

        Raises:
            FileNotFoundError: if the record lacks a required file or has
                neither a profile nor a Fisher information.
            ValueError: if a fitted parameter has neither a profile nor a
                Fisher information, or the posterior is not the one of the
                fitted parameters.
        """
        fit_id = directory.name
        missing = [n for n in REQUIRED_FILES if not (directory / n).exists()]
        if missing:
            raise FileNotFoundError(f"The record '{directory}' has no {missing}.")
        best = _read_json(directory / "parameters.json")["sets"][0]
        values: dict[str, float] = {k: float(v) for k, v in best["values"].items()}
        units: dict[str, str | None] = dict(best.get("units") or {})
        result = _read_json(directory / "optimization_result.json")
        fit_parameters = {p["pid"]: p for p in result["parameters"]}
        default_scale = _scale(
            result.get("settings", {}).get("parameter_scale"),
            ParameterScaleType.LINEAR,
        )

        profiles: dict[str, Mapping[str, Any]] = {}
        alpha, df = 0.95, 1
        if (directory / "identifiability.json").exists():
            identifiability = _read_json(directory / "identifiability.json")
            profiles = identifiability.get("profiles") or {}
            settings = identifiability.get("settings") or {}
            alpha = float(settings.get("alpha", alpha))
            df = int(settings.get("degrees_of_freedom", df))
        fisher: FisherInformation | None = None
        if (directory / "fisher.json").exists():
            fisher = FisherInformation.from_dict(_read_json(directory / "fisher.json"))
        if not profiles and fisher is None:
            raise FileNotFoundError(
                f"The record '{directory}' has neither a profile likelihood nor a "
                f"Fisher information."
            )
        z = math.sqrt(float(stats.chi2.ppf(alpha, df)))

        fisher_se: dict[str, float] = {}
        fisher_ci: dict[str, tuple[float, float]] = {}
        fisher_alpha = alpha
        if fisher is not None:
            fisher_alpha = fisher.alpha
            errors = fisher.standard_errors
            lower, upper = fisher.confidence_intervals()
            for k, pid in enumerate(fisher.pids):
                if math.isfinite(float(errors[k])):
                    fisher_se[pid] = float(errors[k])
                    fisher_ci[pid] = (float(lower[k]), float(upper[k]))

        parameters: dict[str, ParameterUncertainty] = {}
        for pid, value in values.items():
            fp = fit_parameters[pid]
            unit = units.get(pid)
            fp_unit = fp.get("unit")
            scale = _scale(fp.get("scale"), default_scale)
            lower_bound = convert_value(float(fp["lower_bound"]), fp_unit, unit)
            upper_bound = convert_value(float(fp["upper_bound"]), fp_unit, unit)
            profile = profiles.get(pid)
            if profile is not None:
                method = UncertaintyMethod.PROFILE
                ci_lower = _convert_optional(profile.get("ci_lower"), fp_unit, unit)
                ci_upper = _convert_optional(profile.get("ci_upper"), fp_unit, unit)
                identifiability_class = profile.get("identifiability")
                level = alpha
            elif pid in fisher_se:
                method = UncertaintyMethod.FISHER
                ci_lower, ci_upper = fisher_ci[pid]
                identifiability_class = None
                level = fisher_alpha
            else:
                raise ValueError(
                    f"The parameter '{pid}' of the fit '{fit_id}' has neither a "
                    f"profile likelihood nor a Fisher information."
                )
            parameters[pid] = ParameterUncertainty(
                pid=pid,
                value=value,
                unit=unit,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                scale=scale,
                alpha=level,
                z=z,
                method=method,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                identifiability=identifiability_class,
                standard_error=fisher_se.get(pid),
                fisher_ci=fisher_ci.get(pid),
                fit_id=fit_id,
                cost=float(best["cost"]),
                n_data=None if fisher is None else fisher.n,
                record=f"{fit_id}/",
            )
        posterior: ParameterPosterior | None = None
        if (directory / POSTERIOR_FILE).exists():
            posterior = ParameterPosterior.read(directory / POSTERIOR_FILE)
            _check_posterior(posterior, values, units, directory)
        return cls(
            directory=directory,
            parameters=parameters,
            fisher=fisher,
            posterior=posterior,
        )


def _check_posterior(
    posterior: ParameterPosterior,
    values: Mapping[str, float],
    units: Mapping[str, str | None],
    directory: Path,
) -> None:
    """Check that the posterior of a record is the one of its fitted parameters.

    Raises:
        ValueError: if the posterior has other parameters or other values.
    """
    if set(posterior.pids) != set(values):
        raise ValueError(
            f"The posterior of the record '{directory}' has the parameters "
            f"{sorted(posterior.pids)}, the fit {sorted(values)}."
        )
    for pid, value, unit in zip(
        posterior.pids, posterior.values, posterior.units, strict=True
    ):
        expected = convert_value(values[pid], units.get(pid), unit)
        if not math.isclose(float(value), expected, rel_tol=1e-9):
            raise ValueError(
                f"The posterior of the record '{directory}' is at {pid} = "
                f"{float(value)!r}, the fit at {expected!r}: it is the posterior "
                f"of another parameter set."
            )


class AdoptedRecords:
    """The adopted fits of a project: the fitted values and their uncertainty.

    A model file of the factory takes the value of a fitted parameter from the
    record of its fit, so the value, its uncertainty and its provenance come
    from one place, see `sbmlsim.project.Project.fit_records`:

    ```python
    # not executed
    FITS = PROJECT.fit_records()
    Parameter("HCTZEX_f", **FITS.parameter("KI__HCTZEX_f", unit=U.dimensionless))
    ```
    """

    def __init__(self, directory: Path, fit_ids: Sequence[str]) -> None:
        """Read the records of the adopted fits.

        Args:
            directory: the folder of the adopted fits.
            fit_ids: the ids of the adopted fits.

        Raises:
            FileNotFoundError: if a record does not exist.
            ValueError: if a parameter is in two fits.
        """
        self.directory = directory
        self.records: dict[str, AdoptedRecord] = {}
        for fit_id in fit_ids:
            path = directory / fit_id
            if not path.is_dir():
                raise FileNotFoundError(
                    f"The fit '{fit_id}' is not adopted: {path} does not exist; "
                    f"a fit is adopted by the fit kind with `adopt`, which writes "
                    f"its record into the output, fits/{fit_id}."
                )
            self.records[fit_id] = AdoptedRecord.load(path)
        self.parameters: dict[str, ParameterUncertainty] = {}
        for record in self.records.values():
            for pid, parameter in record.parameters.items():
                if pid in self.parameters:
                    raise ValueError(
                        f"The parameter '{pid}' is a result of the fits "
                        f"'{self.parameters[pid].fit_id}' and '{record.fit_id}'; "
                        f"a parameter takes its value from one adopted fit."
                    )
                self.parameters[pid] = parameter

    def __getitem__(self, pid: str) -> ParameterUncertainty:
        """Get the uncertainty of a fitted parameter.

        Raises:
            KeyError: if no adopted fit has the parameter.
        """
        try:
            return self.parameters[pid]
        except KeyError:
            raise KeyError(
                f"No adopted fit has the parameter '{pid}', the fitted parameters "
                f"are {sorted(self.parameters)}."
            ) from None

    def parameter(self, pid: str, unit: UnitType = None) -> dict[str, Any]:
        """Get the keyword arguments of a `sbmlutils.factory.Parameter` of a fitted parameter.

        Args:
            pid: the id of the parameter in the flat model, as the fit names it.
            unit: the unit of the parameter in the model definition, set on the
                parameter and its uncertainty.

        Returns:
            `value` and `uncertainties`, and `unit` if given.

        Raises:
            KeyError: if no adopted fit has the parameter.
        """
        parameter = self[pid]
        kwargs: dict[str, Any] = {
            "value": parameter.value,
            "uncertainties": [parameter.uncertainty(unit)],
        }
        if unit is not None:
            kwargs["unit"] = unit
        return kwargs
