"""The netCDF files of the results: compressed netCDF4, the attributes as JSON.

A result is written as netCDF4 (HDF5) through h5netcdf, the one file format
which xarray, netCDF-C, R, MATLAB and Julia read with the dimensions,
coordinates and attributes of the result. The attributes of a result are
nested, they are written as one JSON attribute, `NETCDF_ATTRS`.

The numeric variables are compressed (#412): the results of simulations
compress well, the HCTZ model with its 191 selections over 5000 points from
1551 MB to 136 MB. `zstd` (the default) is the HDF5 filter 32015 of
`hdf5plugin` with the shuffle filter of HDF5, which writes about twice as long
as no compression and reads about as fast; netCDF-C 4.9 reads it, a python
reader outside of sbmlsim imports `hdf5plugin` first. `gzip` is the filter
every HDF5 reads, at about five times the time of writing. A variable is
stored in chunks of whole timecourses (its last dimension) of about
`CHUNK_BYTES`, which compress better than the chunks h5netcdf would choose.

Arrow, Zarr and npz were measured as well (#412). Arrow IPC writes and reads
fastest, but it is a table: the dimensions, coordinates and attributes would be
an encoding of our own which no other tool reads as such. Zarr is
N-dimensional and native to xarray, but a directory writes a file per chunk,
a zip file duplicate entries, and it adds its dependencies. npz writes slowest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import xarray as xr

#: the attribute of a netCDF file which holds the attributes of the result
NETCDF_ATTRS = "sbmlsim"

#: a compression of the variables, `None` for none
Compression = Literal["zstd", "gzip"] | None

#: the compressions of `Compression`
COMPRESSIONS: tuple[str, ...] = get_args(get_args(Compression)[0])

#: the level of zstd: higher levels compress results a few percent better at
#: several times the time of writing
ZSTD_LEVEL = 1

#: the level of gzip, see `ZSTD_LEVEL`
GZIP_LEVEL = 1

#: the size of a chunk of a compressed variable in bytes
CHUNK_BYTES = 2**20


def write_netcdf(
    ds: xr.Dataset, path: str | Path, compression: Compression = "zstd"
) -> None:
    """Write a dataset as netCDF4, its attributes as JSON.

    Args:
        ds: the dataset, whose attributes must be JSON serializable; numpy
            scalars and arrays are written as numbers and lists.
        path: the file.
        compression: the compression of the numeric variables, see the module.

    Raises:
        ValueError: for an unknown compression.
    """
    if compression is not None and compression not in COMPRESSIONS:
        raise ValueError(
            f"The compression is one of {', '.join(COMPRESSIONS)} or None, not "
            f"'{compression}'."
        )
    # the encoding of a dataset which was read, e.g. its compression, is not kept
    written = ds.drop_encoding()
    written.attrs = {NETCDF_ATTRS: json.dumps(ds.attrs, default=_json_default)}
    encoding = {
        name: _encoding(variable, compression)
        for name, variable in written.data_vars.items()
        if compression is not None and _compressible(variable)
    }
    written.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def read_netcdf(path: str | Path) -> xr.Dataset:
    """Read a dataset of `write_netcdf` into memory, with its attributes.

    Raises:
        ValueError: if the file has no attributes of a result of sbmlsim.
    """
    # registers the filters of the compressed variables with HDF5
    import hdf5plugin  # noqa: F401

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        loaded = ds.load()
    if NETCDF_ATTRS not in loaded.attrs:
        raise ValueError(
            f"'{path}' is no result of sbmlsim, it has no '{NETCDF_ATTRS}' attribute."
        )
    loaded.attrs = json.loads(loaded.attrs[NETCDF_ATTRS])
    return loaded


def _compressible(variable: xr.DataArray) -> bool:
    """Check whether a variable is numeric with values, which HDF5 chunks."""
    return (
        variable.ndim > 0
        and variable.size > 0
        and np.issubdtype(variable.dtype, np.number)
    )


def _encoding(variable: xr.DataArray, compression: str) -> dict[str, Any]:
    """Get the encoding of h5netcdf of a compressed variable."""
    if compression == "zstd":
        import hdf5plugin

        encoding: dict[str, Any] = dict(hdf5plugin.Zstd(clevel=ZSTD_LEVEL))
    else:
        encoding = {"compression": "gzip", "compression_opts": GZIP_LEVEL}
    encoding["shuffle"] = True
    encoding["chunksizes"] = chunk_shape(variable.shape, variable.dtype.itemsize)
    return encoding


def chunk_shape(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """Get the chunks of a variable: whole rows of its last dimension.

    A chunk holds about `CHUNK_BYTES`: the whole last dimension (a timecourse)
    and as many of the dimensions before it as fit, from the last of them on.
    A last dimension larger than a chunk is cut.

    Args:
        shape: the shape of the variable, every size at least one.
        itemsize: the bytes of a value.
    """
    room = max(1, CHUNK_BYTES // itemsize)
    chunks = [1] * len(shape)
    for axis in reversed(range(len(shape))):
        chunks[axis] = max(1, min(shape[axis], room))
        room = max(1, room // chunks[axis])
        if room == 1:
            break
    return tuple(chunks)


def json_attributes(attrs: Mapping[str, Any]) -> Any:
    """Get attributes as the JSON types which `read_netcdf` reads them as.

    Attributes in memory and attributes read from a file compare equal in
    this form, e.g. a tuple and a numpy array are lists.
    """
    return json.loads(json.dumps(attrs, default=_json_default))


def _json_default(value: Any) -> Any:
    """Convert a numpy value of the attributes for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable.")
