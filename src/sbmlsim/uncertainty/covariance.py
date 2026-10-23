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

logger = logging.getLogger(__name__)

#: the relative tolerance of the smallest eigenvalue of a positive
#: semi-definite matrix, relative to the largest one
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
    y = np.array(matrix, dtype=float)
    correction = np.zeros_like(y)
    x = y
    for _ in range(max_iterations):
        r = y - correction
        values, vectors = np.linalg.eigh((r + r.T) / 2.0)
        x = (vectors * np.clip(values, 0.0, None)) @ vectors.T
        correction = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        if np.linalg.norm(y_new - y) <= tol * max(np.linalg.norm(y), 1.0):
            y = y_new
            break
        y = y_new
    # the last projection onto the positive semi-definite matrices with the
    # unit diagonal restored, so the result is both within the rounding
    values, vectors = np.linalg.eigh((y + y.T) / 2.0)
    y = (vectors * np.clip(values, 0.0, None)) @ vectors.T
    d = np.sqrt(np.clip(np.diag(y), 1e-300, None))
    y = y / np.outer(d, d)
    np.fill_diagonal(y, 1.0)
    return (y + y.T) / 2.0


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
        m, n = len(self.targets), self.matrix
        if len(set(self.targets)) != m:
            twice = sorted({t for t in self.targets if self.targets.count(t) > 1})
            raise ValueError(f"The targets {twice} are given twice.")
        if n.ndim != 2 or n.shape != (m, m):
            raise ValueError(
                f"The matrix of {list(self.targets)} must be square of size {m}, "
                f"not of the shape {n.shape}."
            )
        if not np.all(np.isfinite(n)):
            raise ValueError(f"The matrix of {list(self.targets)} is not finite.")
        scale = max(float(np.max(np.abs(n))), 1e-300)
        if not np.allclose(n, n.T, rtol=0.0, atol=1e-12 * scale):
            raise ValueError(f"The matrix of {list(self.targets)} is not symmetric.")
        if self.kind == "correlation":
            if not np.allclose(np.diag(n), 1.0, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"The diagonal of the correlation of {list(self.targets)} "
                    "must be 1."
                )
            if np.any(np.abs(n) > 1.0 + 1e-12):
                raise ValueError(
                    f"The correlation of {list(self.targets)} has a coefficient "
                    "with |rho| > 1."
                )
        else:
            if np.any(np.diag(n) <= 0.0):
                raise ValueError(
                    f"The covariance of {list(self.targets)} must have a positive "
                    "variance of every target."
                )
            if self.units is not None and len(self.units) != m:
                raise ValueError(
                    f"The covariance of {list(self.targets)} needs one unit per target."
                )
        if not _is_psd(n):
            smallest = float(np.linalg.eigvalsh(n).min())
            corr = self.correlation_matrix()
            off = np.abs(corr - np.eye(m))
            i, j = np.unravel_index(int(np.argmax(off)), off.shape)
            raise ValueError(
                f"The {self.kind} of {list(self.targets)} is not positive "
                f"semi-definite: its smallest eigenvalue is {smallest:.3g}; the "
                f"largest correlation is {corr[i, j]:.3g} of "
                f"('{self.targets[i]}', '{self.targets[j]}')."
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
    values = np.linalg.eigvalsh(matrix)
    return bool(values.min() >= -PSD_RTOL * max(float(values.max()), 1e-300))
