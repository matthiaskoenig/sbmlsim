"""DataGenerator."""

import numpy as np
import xarray as xr

from sbmlsim.data import DataSet
from sbmlsim.result import XResult


class DataGeneratorFunction:
    """DataGeneratorFunction."""

    def __call__(
        self, xresults: dict[str, XResult], dsets: dict[str, DataSet] | None = None
    ) -> dict[str, XResult]:
        """Call the function.

        Args:
            xresults: Results to process.
            dsets: Datasets to process.

        Returns:
            Processed results.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class DataGeneratorIndexingFunction(DataGeneratorFunction):
    """DataGeneratorIndexingFunction."""

    def __init__(self, index: int, dimension: str = "_time"):
        """Initialize DataGeneratorIndexingFunction.

        The index of the time of a result whose simulations keep their own
        time points (`_point`) is the index of the point of every simulation,
        a negative index counts from its last point.

        Args:
            index: Index to select on the dimension.
            dimension: Dimension to reduce, the time by default.
        """
        self.index = index
        self.dimension = dimension

    def __call__(
        self, xresults: dict[str, XResult], dsets: dict[str, DataSet] | None = None
    ) -> dict[str, XResult]:
        """Reduce a dimension, by default the time, at the index.

        Args:
            xresults: Results to process.
            dsets: Datasets to process (unused).

        Returns:
            Reduced results.
        """
        results = {}
        for key, xres in xresults.items():
            if self.dimension == "_time" and "_point" in xres.xds.dims:
                xds_new = _point_of_every_simulation(xres.xds, self.index)
            else:
                xds_new = xres.xds.isel({self.dimension: self.index})
            xres_new = XResult(xdataset=xds_new, uinfo=xres.uinfo)
            results[key] = xres_new

        return results


class DataGenerator:
    """DataGenerator.

    DataGenerators allow to postprocess existing data. This can be a variety of
    operations.

    - Slicing: reduce the dimension of a given XResult, by slicing a subset on a
      given dimension
    - Cumulative processing: mean, sd, ...
    - Complex processing, such as pharmacokinetics calculation.
    """

    def __init__(
        self,
        f: DataGeneratorFunction,
        xresults: dict[str, XResult],
        dsets: dict[str, DataSet] | None = None,
    ):
        """Initialize DataGenerator.

        Args:
            f: Function applied to the data.
            xresults: Results to process.
            dsets: Datasets to process.
        """
        self.xresults = xresults
        self.dsets = dsets
        self.f = f

    def process(self) -> dict[str, XResult]:
        """Process the data generator.

        Returns:
            Processed results.
        """
        return self.f(xresults=self.xresults, dsets=self.dsets)


def _point_of_every_simulation(xds: xr.Dataset, index: int) -> xr.Dataset:
    """Select a point of every simulation of a result, without its padding.

    Args:
        xds: the dataset with the dimension `_point` first.
        index: the index of the point, negative from the last point.

    Returns:
        The dataset without the dimension `_point`.
    """
    time = np.asarray(xds["time"].values, dtype=float)
    n_points = np.sum(np.isfinite(time), axis=0)
    rows = n_points + index if index < 0 else np.full_like(n_points, index)
    variables = {}
    for key in xds.data_vars:
        values = np.asarray(xds[key].values)
        picked = np.take_along_axis(values, rows[np.newaxis, ...], axis=0)[0]
        dims = [d for d in xds[key].dims if d != "_point"]
        variables[key] = xr.DataArray(
            picked,
            dims=dims,
            coords={d: xds.coords[d] for d in dims if d in xds.coords},
            attrs=xds[key].attrs,
        )
    return xr.Dataset(variables)
