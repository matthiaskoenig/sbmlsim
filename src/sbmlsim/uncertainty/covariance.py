"""Covariances of uncertain values: the dependence between them.

A `Covariance` is the dependence of a set of uncertain values (the inputs of a
simulation or the parameters of a model), given as a covariance matrix or as a
correlation: a full matrix or the coefficients of pairs. The same object is
used by every step of an uncertainty analysis, see `sbmlsim.uncertainty`.

A covariance is in the units of its targets (`units`, the entry `ij` in the
unit of `i` times the unit of `j`), on the linear scale or on the log scale
(`scale="log"`, e.g. the omega matrix of a population analysis of log-normal
parameters). A correlation is dimensionless; the standard deviations come from
the marginals of its targets.

Every matrix is validated: square, symmetric, finite, a positive diagonal
(unit for a correlation, `|rho| <= 1`) and positive semi-definite. A
correlation which is not positive semi-definite, e.g. assembled from the
coefficients of several sources, can be replaced by the nearest correlation
matrix (`nearest=True`, Higham 2002).

Higham NJ (2002). Computing the nearest correlation matrix: a problem from
finance. IMA J Numer Anal 22:329-343.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from sbmlsim.engine import import_core

logger = logging.getLogger(__name__)

#: the relative tolerance of the smallest eigenvalue of a positive
#: semi-definite matrix, relative to the largest one, `PSD_RTOL` of
#: `sbmlsim_core.stats`
PSD_RTOL: float = 1e-10


def _read_only(matrix: np.ndarray) -> np.ndarray:
    """Get a read-only copy of a matrix."""
    copy = np.array(matrix, dtype=float)
    copy.setflags(write=False)
    return copy


def nearest_correlation(
    matrix: ArrayLike, max_iterations: int = 100, tol: float = 1e-10
) -> np.ndarray:
    """Get the nearest correlation matrix in the Frobenius norm.

    The alternating projections method with the correction of Dykstra of
    Higham (2002): the projection onto the positive semi-definite matrices
    and onto the matrices with a unit diagonal, alternately, until the
    iterates change less than `tol`.

    Args:
        matrix: a symmetric matrix with a unit diagonal.
        max_iterations: the most iterations.
        tol: the tolerance of the change of an iteration in the Frobenius norm.

    Returns:
        The nearest correlation matrix.
    """
    import_core()
    from sbmlsim_core import stats

    return stats.nearest_correlation(matrix, max_iterations, tol)


@dataclass(frozen=True, eq=False)
class Covariance:
    """The dependence of uncertain values, see the module.

    Attributes:
        targets: the targets, the names of the uncertain values.
        matrix: the read-only matrix, a covariance or a correlation.
        kind: `covariance` or `correlation`.
        scale: the scale of a covariance, `linear` or `log`.
        units: the unit of every target of a covariance, `None` without.
    """

    targets: tuple[str, ...]
    matrix: np.ndarray
    kind: Literal["covariance", "correlation"] = "covariance"
    scale: Literal["linear", "log"] = "linear"
    units: tuple[str | None, ...] | None = None

    def __init__(
        self,
        targets: Sequence[str],
        matrix: ArrayLike,
        units: Sequence[str | None] | None = None,
        scale: Literal["linear", "log"] = "linear",
        kind: Literal["covariance", "correlation"] = "covariance",
    ) -> None:
        """Create a covariance (or a correlation, see `correlation`).

        Args:
            targets: the targets in the order of the rows.
            matrix: the matrix.
            units: the unit of every target, for a covariance.
            scale: `linear` or `log`, for a covariance.
            kind: `covariance` or `correlation`.

        Raises:
            ValueError: if the matrix is not valid, see the module.
        """
        object.__setattr__(self, "targets", tuple(str(t) for t in targets))
        object.__setattr__(self, "matrix", _read_only(np.asarray(matrix, dtype=float)))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "units", None if units is None else tuple(units))
        self._validate()

    @classmethod
    def correlation(
        cls,
        pairs: Mapping[tuple[str, str], float] | None = None,
        *,
        matrix: ArrayLike | None = None,
        targets: Sequence[str] | None = None,
        nearest: bool = False,
    ) -> Covariance:
        """Create a correlation, from the coefficients of pairs or a matrix.

        Args:
            pairs: `(a, b) -> rho`, the other pairs are uncorrelated; the
                targets in the order they appear.
            matrix: a full correlation matrix, with `targets`.
            targets: the targets of the matrix.
            nearest: replace a matrix which is not positive semi-definite by
                the nearest correlation matrix, with a warning.

        Returns:
            The correlation.

        Raises:
            ValueError: for pairs and a matrix together or neither, a matrix
                without targets, or a matrix which is not valid.
        """
        if (pairs is None) == (matrix is None):
            raise ValueError("A correlation takes either pairs or a matrix.")
        if pairs is not None:
            names = list(dict.fromkeys(t for pair in pairs for t in pair))
            full = np.eye(len(names))
            for (a, b), rho in pairs.items():
                if a == b:
                    raise ValueError(
                        f"The pair ('{a}', '{b}') correlates a target with itself."
                    )
                i, j = names.index(a), names.index(b)
                full[i, j] = full[j, i] = float(rho)
            targets, matrix = names, full
        if targets is None:
            raise ValueError("A correlation matrix needs its targets.")
        array = np.asarray(matrix, dtype=float)
        if nearest and array.ndim == 2 and array.shape[0] == array.shape[1]:
            symmetric = np.allclose(array, array.T, rtol=0.0, atol=1e-12)
            if symmetric and not _is_psd(array):
                logger.warning(
                    "The correlation of %s is not positive semi-definite, the "
                    "nearest correlation matrix is used.",
                    list(targets),
                )
                array = nearest_correlation(array)
        return cls(targets, array, kind="correlation")

    def _validate(self) -> None:
        """Check the matrix, see the module.

        Raises:
            ValueError: if it is not valid.
        """
        import_core()
        from sbmlsim_core import stats

        stats.validate_covariance(
            self.targets, self.matrix, kind=self.kind, units=self.units
        )

    def sds(self) -> np.ndarray | None:
        """Get the standard deviations of a covariance, `None` for a correlation."""
        if self.kind == "correlation":
            return None
        return np.sqrt(np.diag(self.matrix))

    def correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix."""
        if self.kind == "correlation":
            return np.array(self.matrix)
        d = np.sqrt(np.diag(self.matrix))
        return self.matrix / np.outer(d, d)

    def to_dict(self) -> dict[str, Any]:
        """Get the covariance as a dictionary of JSON types."""
        return {
            "targets": list(self.targets),
            "matrix": self.matrix.tolist(),
            "kind": self.kind,
            "scale": self.scale,
            "units": None if self.units is None else list(self.units),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Covariance:
        """Create a covariance from `to_dict`."""
        return cls(
            d["targets"],
            d["matrix"],
            units=d.get("units"),
            scale=d.get("scale", "linear"),
            kind=d.get("kind", "covariance"),
        )

    def __eq__(self, other: object) -> bool:
        """Compare the targets, the matrices and the fields."""
        if not isinstance(other, Covariance):
            return NotImplemented
        return (
            self.targets == other.targets
            and self.kind == other.kind
            and self.scale == other.scale
            and self.units == other.units
            and np.array_equal(self.matrix, other.matrix)
        )

    def __hash__(self) -> int:
        """Hash the targets and the bytes of the matrix."""
        return hash((self.targets, self.kind, self.scale, self.matrix.tobytes()))


def _is_psd(matrix: np.ndarray) -> bool:
    """Check whether a symmetric matrix is positive semi-definite."""
    import_core()
    from sbmlsim_core import stats

    return stats.is_psd(matrix, PSD_RTOL)
