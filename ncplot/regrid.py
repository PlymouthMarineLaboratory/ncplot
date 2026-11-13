"""Curvilinear grids and unstructured meshes as regular longitude-latitude ones.

``view`` plots regular grids well, and anything else badly: the seam of a
global ocean grid (where the longitudes jump by 360 degrees), its tripolar
north fold (two rows of cells in the same places), cells piled up at one
point, and FVCOM's triangular mesh. ``regrid_dataset`` gives such variables a
regular grid to be drawn on instead, using the engine ported from oceanval
(``ncplot.grid`` for curvilinear and unstructured cells, ``ncplot.mesh`` for FVCOM).

The result is lazy, and needs no dask: a frame (one time, one level) is read
and put on the regular grid only when it is asked for, so a 6 GB FVCOM file
is opened at once and each time step shown costs one slice.
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from ncplot import grid as ncgrid
from ncplot import mesh as ncmesh

_LON_UNITS = ("degrees_east", "degree_east", "degrees_e", "degree_e", "degreee")
_LAT_UNITS = ("degrees_north", "degree_north", "degrees_n", "degree_n", "degreen")

# variables that describe a grid, not the ocean
_BOUNDS_SUFFIXES = ("_bnds", "_bounds", "_vertices")

# a grid whose longitudes and latitudes leave out more of the cells with values than this is not regridded
MIN_PLACED = 0.9


# ---- what a variable is on ----


def _attr(variable, name):
    value = variable.attrs.get(name, "")
    return str(value) if value is not None else ""


def _bounds_names(ds):
    names = set()
    for variable in ds.variables.values():
        for key in ("bounds", "climatology", "edges"):
            if key in variable.attrs:
                names.add(str(variable.attrs[key]))
    return names


def find_lonlat(ds, name):
    """The names of the longitude and latitude of variable ``name`` of ``ds``,
    or Nones: its coordinates with a standard_name, units or name that says
    so (``nav_lon`` and ``nav_lat`` too). A variable no coordinates name (an
    unstructured grid's, whose longitudes and latitudes are variables
    nothing points at) takes any variable on its dimensions that says it is
    one. Attributes are read through ``ds.variables``, which is cheap."""
    variables = ds.variables
    variable = variables[name]
    dims = set(variable.dims)
    candidates = list(ds[name].coords)
    for word in _attr(variable, "coordinates").split():
        if word in variables and word not in candidates:
            candidates.append(word)
    candidates += [dim for dim in variable.dims if dim in variables and dim not in candidates]

    def find(keys, standard, units, words):
        for key in keys:
            if _attr(variables[key], "standard_name") == standard:
                return key
        for key in keys:
            if _attr(variables[key], "units").lower() in units:
                return key
        for key in keys:
            if any(key.lower().startswith(word) or key.lower().endswith(word) for word in words):
                return key
        return None

    lon = find(candidates, "longitude", _LON_UNITS, ("lon", "nav_lon", "glamt"))
    lat = find(candidates, "latitude", _LAT_UNITS, ("lat", "nav_lat", "gphit"))
    if lon is None or lat is None:
        others = [key for key, other in variables.items() if key != name and other.ndim and set(other.dims) <= dims]
        lon = lon or find(others, "longitude", _LON_UNITS, ())
        lat = lat or find(others, "latitude", _LAT_UNITS, ())
    # coordinates that do not span the variable's own dimensions cannot place it
    for key in (lon, lat):
        if key is not None and not set(variables[key].dims) <= dims:
            return None, None
    return lon, lat


def _repeated_coordinates(lons, lats):
    """Whether cells share a longitude and latitude: the 0s (or fill values)
    a model writes at land cells, which put those cells on top of one another."""
    pairs = pd.DataFrame({"lon": np.asarray(lons).ravel(), "lat": np.asarray(lats).ravel()})
    return bool(pairs.duplicated().any())


def needs_raster(lons, lats):
    """Whether a grid at ``lons``, ``lats`` is better drawn on a regular grid
    than as it is: an unstructured grid (1-D, cells along one dimension), or
    a curvilinear one with a seam, cells in the same places, cells turned
    over, or cells with no longitude or latitude."""
    if lons.ndim == 1 and lats.ndim == 1:
        return lons.shape == lats.shape
    if lons.ndim != 2 or lons.shape != lats.shape:
        return False
    if ncgrid.has_seam(lons) or _repeated_coordinates(lons, lats) or ncgrid.has_folds(lons, lats):
        return True
    return bool((~np.isfinite(lons) | ~np.isfinite(lats)).any())


class Source:
    """What one variable is drawn from: its ``kind`` ("mesh", "cells" or
    "grid"), its horizontal dimensions ``dims`` (in the order of its
    longitudes and latitudes), and where its values are on an FVCOM mesh
    (``location``: "node" or "nele")."""

    def __init__(self, kind, dims, lon=None, lat=None, location=None):
        self.kind, self.dims, self.lon, self.lat, self.location = kind, tuple(dims), lon, lat, location

    @property
    def key(self):
        return (self.kind, self.dims, self.lon, self.lat, self.location)


def find_source(ds, name):
    """The Source of variable ``name`` if it needs regridding, else None."""
    variable = ds.variables[name]
    if variable.dtype.kind not in "fiu" or not variable.ndim:
        return None
    if ncmesh.looks_like_fvcom(ds):
        location = ncmesh.mesh_location(variable.dims)
        if name in ncmesh.MESH_VARIABLES or location is None or variable.dtype.kind != "f":
            return None
        return Source("mesh", [location], location=location)
    if name.endswith(_BOUNDS_SUFFIXES):
        return None
    lon, lat = find_lonlat(ds, name)
    if lon is None or lat is None:
        return None
    lon_dims, lat_dims = ds.variables[lon].dims, ds.variables[lat].dims
    if len(lon_dims) == 1 and lon_dims == lat_dims and lon_dims[0] in variable.dims:
        # cells along one dimension, each with its longitude and latitude
        source = Source("cells", lon_dims, lon, lat)
    elif len(lon_dims) == 2 and lon_dims == lat_dims and set(lon_dims) <= set(variable.dims):
        source = Source("grid", lon_dims, lon, lat)
    else:
        return None
    lons, lats = np.asarray(ds.variables[lon].values, dtype="float64"), np.asarray(ds.variables[lat].values, dtype="float64")
    return source if needs_raster(lons, lats) else None


# ---- the regular grid ----


class RasterArray(BackendArray):
    """A variable with its horizontal dimensions on a regular grid: the
    ``source`` Variable's values put on ``raster`` when asked for. Its shape
    is that of the source with its horizontal dimensions (``dims``) replaced by
    the raster's two, last; only the frames (the other dimensions) that are
    indexed are read."""

    def __init__(self, source, dims, raster, location=None):
        self.source, self.dims, self.raster, self.location = source, tuple(dims), raster, location
        self.lead = [dim for dim in source.dims if dim not in self.dims]
        self.shape = tuple(source.sizes[dim] for dim in self.lead) + tuple(raster.shape)
        self.dtype = np.dtype("float32")

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.OUTER, self._get)

    def _image(self, values):
        if self.location is not None:
            return self.raster.image(values, self.location)
        return self.raster.image(values)

    def _get(self, key):
        lead_key, (key_y, key_x) = key[: len(self.lead)], key[len(self.lead):]
        part = self.source.isel({dim: k for dim, k in zip(self.lead, lead_key)})
        kept = [dim for dim in self.lead if dim in part.dims]
        # the frames, each with its horizontal dimensions in the order of the grid's coordinates
        frames = part.transpose(*kept, *self.dims).values.reshape((-1,) + tuple(part.sizes[dim] for dim in self.dims))
        images = np.stack([self._image(np.asarray(frame, dtype="float64")) for frame in frames])
        images = images.reshape(tuple(part.sizes[dim] for dim in kept) + tuple(self.raster.shape))
        return images[..., key_y, :][..., key_x].astype("float32")


def _level_coordinate(ds, dim):
    """The coordinate of a dimension that comes through regridding: its own
    if it has a 1-D one, else the numbers 1 to n (FVCOM's sigma layers have
    values at each node, not one for the layer: 1 is the surface)."""
    if dim in ds.variables and ds.variables[dim].dims == (dim,):
        return ds.variables[dim]
    return xr.Variable((dim,), np.arange(1, ds.sizes[dim] + 1), {"long_name": f"{dim} number"})


def regrid_dataset(ds, names=None):
    """``ds`` with the variables that need it (see ``needs_raster``; ``names``:
    of these only) on a regular longitude-latitude grid, as ``(Dataset, notes)``:
    only those variables, with dimensions ``lat`` and ``lon``, lazy. ``(None,
    [])`` if no variable needs it, so that the dataset is plotted as it is.

    A grid whose longitudes and latitudes cannot place its cells is left out
    (it is plotted by its indices, as it was)."""
    if names is not None:
        names = [names] if isinstance(names, str) else list(names)
    wanted = list(ds.data_vars) if names is None else names
    skip = _bounds_names(ds)
    sources = {}
    for name in wanted:
        if name in skip or name not in ds.variables:
            continue
        source = find_source(ds, name)
        if source is not None:
            sources[name] = source
    if not sources:
        return None, []

    notes = []
    rasters = {}
    mesh = None
    variables = {}
    axes = None
    for name, source in sources.items():
        if source.key not in rasters:
            if source.kind == "mesh":
                if mesh is None:
                    mesh = ncmesh.Mesh.from_dataset(ds)
                rasters[source.key] = mesh.raster()
            else:
                lons = np.asarray(ds.variables[source.lon].values, dtype="float64")
                lats = np.asarray(ds.variables[source.lat].values, dtype="float64")
                try:
                    raster = ncgrid.raster_for(lons, lats)
                except ValueError:
                    raster = None
                if raster is not None:
                    # a grid whose coordinates leave out many of its cells with values is plotted by its indices
                    sample = ds[name].isel({dim: 0 for dim in ds[name].dims if dim not in source.dims})
                    if raster.placed_share(sample.transpose(*source.dims).values) < MIN_PLACED:
                        raster = None
                rasters[source.key] = raster
        raster = rasters[source.key]
        if raster is None:
            notes.append(f"{name}: the grid's longitudes and latitudes cannot place its cells.")
            continue
        if axes is not None and not (np.array_equal(axes[0], raster.x_axis) and np.array_equal(axes[1], raster.y_axis)):
            notes.append(f"{name}: it is on another grid than the first variable, so it is left out.")
            continue
        axes = (raster.x_axis, raster.y_axis)
        array = RasterArray(ds.variables[name], source.dims, raster, source.location)
        attrs = {k: v for k, v in ds.variables[name].attrs.items() if k in ("long_name", "standard_name", "units")}
        variables[name] = xr.Variable(array.lead + ["lat", "lon"], indexing.LazilyIndexedArray(array), attrs)
    if not variables:
        return None, notes

    coords = {}
    for name, variable in variables.items():
        for dim in variable.dims[:-2]:
            if dim not in coords:
                coords[dim] = _level_coordinate(ds, dim)
    coords["lat"] = xr.Variable(("lat",), axes[1], {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"})
    coords["lon"] = xr.Variable(("lon",), axes[0], {"standard_name": "longitude", "units": "degrees_east", "axis": "X"})
    result = xr.Dataset(variables, coords=coords, attrs=dict(ds.attrs))
    dropped = [name for name in wanted if name not in variables and name not in skip and name in ds.data_vars]
    if names is not None and dropped:
        notes.append("Not on a grid that is regridded, so left out: " + ", ".join(dropped))
    return result, notes


def native_range(ds, name):
    """The smallest and largest value of variable ``name`` of ``ds``, from the
    values as they are on their own grid (cheaper than on the regular one, and
    one range for every time and level). ``(low, high)``, or None if there
    are no values."""
    values = ds[name]
    low, high = float(values.min(skipna=True)), float(values.max(skipna=True))
    return (low, high) if np.isfinite(low) and np.isfinite(high) else None


def warn_notes(notes):
    for note in notes:
        warnings.warn(note)
