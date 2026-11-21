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

The variables of one shape are written as one array and not as one variable
each (#607): HDF5 and h5netcdf cost about 2.5 ms to write and 3.4 ms to read
per variable of a file, whatever its size, so the 190 selections of the HCTZ
model over 200 points took 474 ms to write and 559 ms to read as 190 variables
against 62 ms and 30 ms as one array of them, and a result of the large model
(3301 variables) would take about 8 s and 11 s. The cost is the variable and
not the compression, which the same measurement without it shows (366 ms and
422 ms against 15 ms and 10 ms), and the file is the same size either way,
since a chunk of the array holds what a chunk of a variable held.

So `write_netcdf` gathers the numeric variables of equal dimensions and dtype,
from `STACK_MIN` of them on, into one array over a dimension of their names
(`NETCDF_STACKS` records what it gathered, `read_netcdf` hands the variables
back in the order they were written, with their attributes) and leaves a
result of a few variables as it is, which keeps a small file readable as
itself. An array over a dimension of names is ordinary netCDF, so every
reader still reads a gathered file; what it does not read is one variable per
selection by name.

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

#: the attribute of a netCDF file which records the arrays the variables were
#: gathered into, `None` in a file whose variables are written one by one
NETCDF_STACKS = "sbmlsim_stacks"

#: the fewest variables of equal dimensions and dtype which are gathered into
#: one array: below it the variables of a file cost little and a file which is
#: read by a tool outside of sbmlsim is more useful with its variables named
STACK_MIN = 8

#: the name of the array which holds the variables of one shape and of the
#: dimension of their names, with the position of the group appended
STACK_PREFIX = "_sbmlsim_stack"
STACK_DIMENSION = "_sbmlsim_variable"

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
    written, stacks = _stacked(ds.drop_encoding())
    written.attrs = {NETCDF_ATTRS: json.dumps(ds.attrs, default=_json_default)}
    if stacks is not None:
        written.attrs[NETCDF_STACKS] = json.dumps(stacks)
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
    stacks = loaded.attrs.get(NETCDF_STACKS)
    attrs = json.loads(loaded.attrs[NETCDF_ATTRS])
    if stacks is not None:
        loaded = _unstacked(loaded, json.loads(stacks))
    loaded.attrs = attrs
    return loaded


def _stacked(ds: xr.Dataset) -> tuple[xr.Dataset, dict[str, Any] | None]:
    """Gather the variables of equal dimensions and dtype into one array each.

    The variables of one group become one array over a leading dimension of
    their names, in the order they have in the dataset; a group of fewer than
    `STACK_MIN` variables is left as it is, and so is every variable which is
    not numeric, has no dimension or is empty. See the module for what the
    gathering is for.

    Args:
        ds: the dataset to write.

    Returns:
        The dataset to write and what was gathered (`None` when nothing was,
        which writes the dataset as it is): the names of the variables of the
        dataset in their order and, per array, its dimension of the names, the
        variables it holds and their attributes.
    """
    groups: dict[tuple[Any, ...], list[str]] = {}
    for name, variable in ds.data_vars.items():
        if _compressible(variable):
            groups.setdefault((variable.dims, variable.dtype.str), []).append(str(name))
    gathered = [
        (key, names) for key, names in groups.items() if len(names) >= STACK_MIN
    ]
    if not gathered:
        return ds, None
    stacks: list[dict[str, Any]] = []
    written = ds
    for position, ((dims, _), names) in enumerate(gathered):
        stack = f"{STACK_PREFIX}{position}"
        dimension = f"{STACK_DIMENSION}{position}"
        if stack in ds.variables or dimension in ds.dims:
            # a dataset of our own never carries these names
            raise ValueError(
                f"The dataset has a variable or dimension of the name '{stack}' or "
                f"'{dimension}', which is the name of an array of gathered variables."
            )
        values = np.stack([ds[name].values for name in names])
        written = written.drop_vars(names).assign({stack: ((dimension, *dims), values)})
        written = written.assign_coords({dimension: np.array(names, dtype=object)})
        stacks.append(
            {
                "stack": stack,
                "dimension": dimension,
                "variables": names,
                "attrs": [json_attributes(ds[name].attrs) for name in names],
            }
        )
    return written, {"order": [str(name) for name in ds.data_vars], "stacks": stacks}


def _unstacked(ds: xr.Dataset, stacks: Mapping[str, Any]) -> xr.Dataset:
    """Hand the variables of the arrays of `_stacked` back, in their order.

    Args:
        ds: the dataset as it was read, with the arrays of the gathering.
        stacks: what `_stacked` recorded.

    Returns:
        The dataset with the variables of every array, their attributes and
        the order they were written in, without the arrays and their
        dimensions.

    Raises:
        ValueError: if the file misses an array its attribute records.
    """
    # the arrays of the gathering are no variables of the result, the variables
    # they hold are; a variable the gathering left alone is one as it is
    drop: list[str] = [group["stack"] for group in stacks["stacks"]]
    drop += [group["dimension"] for group in stacks["stacks"]]
    variables: dict[str, xr.DataArray] = {
        str(name): variable
        for name, variable in ds.data_vars.items()
        if str(name) not in drop
    }
    for group in stacks["stacks"]:
        stack = group["stack"]
        if stack not in ds:
            raise ValueError(
                f"The file records the gathered variables {group['variables']} in "
                f"'{stack}', which it does not have."
            )
        array = ds[stack]
        dimension = group["dimension"]
        dims = tuple(d for d in array.dims if d != dimension)
        values = array.values
        for position, name in enumerate(group["variables"]):
            variables[name] = xr.DataArray(
                values[position], dims=dims, attrs=group["attrs"][position]
            )
    coords = {
        str(name): coordinate
        for name, coordinate in ds.coords.items()
        if str(name) not in drop
    }
    ordered = {name: variables[name] for name in stacks["order"] if name in variables}
    # a variable the gathering left alone and which the order does not name
    ordered.update({n: v for n, v in variables.items() if n not in ordered})
    return xr.Dataset(ordered, coords=coords)


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
