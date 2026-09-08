"""Parameter sets, the artifact which connects a fit to its report.

A parameter fit stores its results as `ParameterSets`: named sets of parameter
values with their units. A report is created from the definition of the
optimization problem, the settings of the fit and one or more of these sets, so
that reporting is separate from optimizing and several sets can be compared in
a single report.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sbmlsim.fit.objects import FitParameter

logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """A named set of parameter values.

    Attributes:
        sid: identifier of the set, used as its label in a report.
        values: value of every parameter by parameter id.
        units: unit of every parameter by parameter id, the model unit if `None`.
        cost: cost of the set, if it comes from an optimization.
        provenance: where the set comes from, e.g. the id of an optimization.
    """

    sid: str
    values: dict[str, float]
    units: dict[str, str | None] = field(default_factory=dict)
    cost: float | None = None
    provenance: str | None = None

    def __post_init__(self) -> None:
        """Normalize the values and complete the units."""
        self.values = {pid: float(value) for pid, value in self.values.items()}
        self.units = {pid: self.units.get(pid) for pid in self.values}

    def __len__(self) -> int:
        """Get the number of parameters."""
        return len(self.values)

    def __str__(self) -> str:
        """Get string representation."""
        values = ", ".join(f"{pid}={value:.5g}" for pid, value in self.values.items())
        return f"{self.__class__.__name__}<{self.sid}: {values}>"

    def x(self, pids: Sequence[str]) -> np.ndarray:
        """Get the values as a vector in the order of the given parameter ids.

        Args:
            pids: parameter ids of the optimization problem.

        Returns:
            Vector of the values.

        Raises:
            KeyError: if the set does not contain one of the parameters.
        """
        missing = [pid for pid in pids if pid not in self.values]
        if missing:
            raise KeyError(
                f"ParameterSet '{self.sid}' does not contain the parameters "
                f"'{missing}', it has '{sorted(self.values)}'."
            )
        return np.array([self.values[pid] for pid in pids], dtype=float)

    @staticmethod
    def from_fit_parameters(
        parameters: Iterable[FitParameter],
        x: Sequence[float] | np.ndarray,
        sid: str,
        cost: float | None = None,
        provenance: str | None = None,
    ) -> ParameterSet:
        """Create a parameter set from the parameters of a problem and a vector.

        Args:
            parameters: fit parameters, they provide the ids and the units.
            x: values in the order of the parameters.
            sid: identifier of the set.
            cost: cost of the set.
            provenance: where the set comes from.

        Returns:
            The parameter set.

        Raises:
            ValueError: if the number of values does not match the parameters.
        """
        parameters = list(parameters)
        if len(parameters) != len(x):
            raise ValueError(
                f"'{sid}': '{len(x)}' values for '{len(parameters)}' parameters."
            )
        return ParameterSet(
            sid=sid,
            values={
                p.pid: float(value) for p, value in zip(parameters, x, strict=True)
            },
            units={p.pid: p.unit for p in parameters},
            cost=cost,
            provenance=provenance,
        )

    @staticmethod
    def from_model(
        parameters: Iterable[FitParameter], x: np.ndarray, sid: str = "model"
    ) -> ParameterSet:
        """Create the parameter set of the initial values of the model."""
        return ParameterSet.from_fit_parameters(
            parameters=parameters,
            x=x,
            sid=sid,
            provenance="initial values of the model",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary of JSON serializable values."""
        return {
            "sid": self.sid,
            "values": self.values,
            "units": self.units,
            "cost": self.cost,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ParameterSet:
        """Create a parameter set from a dictionary."""
        return ParameterSet(
            sid=d["sid"],
            values=d["values"],
            units=d.get("units", {}),
            cost=d.get("cost"),
            provenance=d.get("provenance"),
        )


@dataclass
class ParameterSets:
    """One or more named parameter sets.

    This is what a fit writes and a report reads; the sets of a report are
    compared with each other, e.g., the fitted parameters against the initial
    values of the model.
    """

    sets: list[ParameterSet] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check that the identifiers of the sets are unique.

        Raises:
            ValueError: for duplicate identifiers.
        """
        sids = [pset.sid for pset in self.sets]
        if len(sids) > len(set(sids)):
            raise ValueError(f"Parameter sets need unique ids, but got '{sids}'.")

    def __len__(self) -> int:
        """Get the number of sets."""
        return len(self.sets)

    def __iter__(self) -> Iterator[ParameterSet]:
        """Iterate over the sets."""
        return iter(self.sets)

    def __getitem__(self, key: int | str) -> ParameterSet:
        """Get a set by index or by identifier.

        Raises:
            KeyError: if no set has the given identifier.
        """
        if isinstance(key, int):
            return self.sets[key]
        for pset in self.sets:
            if pset.sid == key:
                return pset
        raise KeyError(f"No parameter set '{key}' in '{[p.sid for p in self.sets]}'.")

    def __str__(self) -> str:
        """Get string representation."""
        return f"{self.__class__.__name__}<{[pset.sid for pset in self.sets]}>"

    @staticmethod
    def of(
        parameter_sets: ParameterSets | Iterable[ParameterSet] | ParameterSet,
    ) -> ParameterSets:
        """Accept a single set, an iterable of sets or `ParameterSets`.

        Args:
            parameter_sets: the sets in any of the supported forms.

        Returns:
            The sets as `ParameterSets`.

        Raises:
            ValueError: if no set is given.
        """
        if isinstance(parameter_sets, ParameterSets):
            psets = parameter_sets
        elif isinstance(parameter_sets, ParameterSet):
            psets = ParameterSets([parameter_sets])
        else:
            psets = ParameterSets(list(parameter_sets))

        if not psets.sets:
            raise ValueError("At least one ParameterSet is required.")
        return psets

    def to_df(self) -> pd.DataFrame:
        """Get a DataFrame with one row per parameter and one column per set."""
        pids: list[str] = []
        for pset in self.sets:
            pids.extend(pid for pid in pset.values if pid not in pids)

        data: list[dict[str, Any]] = []
        for pid in pids:
            row: dict[str, Any] = {"parameter": pid}
            units = {pset.units.get(pid) for pset in self.sets if pid in pset.values}
            row["unit"] = (
                next(iter(units))
                if len(units) == 1
                else "|".join(str(unit) for unit in sorted(units, key=str))
            )
            for pset in self.sets:
                row[pset.sid] = pset.values.get(pid, np.nan)
            data.append(row)

        return pd.DataFrame(data)

    def to_json(self, path: Path | None = None) -> str | Path:
        """Store the sets as JSON.

        Args:
            path: file to write, the JSON string is returned if it is `None`.

        Returns:
            The path or the JSON string.
        """
        info = {"sets": [pset.to_dict() for pset in self.sets]}
        if path is None:
            return json.dumps(info, indent=2)
        with open(path, "w", encoding="utf-8") as f_json:
            json.dump(info, f_json, indent=2)
        return path

    @staticmethod
    def from_json(json_info: str | Path) -> ParameterSets:
        """Load the sets from a JSON file or string.

        Args:
            json_info: path of the file or the JSON string.

        Returns:
            The parameter sets.
        """
        if isinstance(json_info, Path):
            with open(json_info, encoding="utf-8") as f_json:
                d = json.load(f_json)
        else:
            d = json.loads(json_info)
        return ParameterSets([ParameterSet.from_dict(s) for s in d["sets"]])
