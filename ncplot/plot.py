import sys
import warnings
from threading import Thread

import glob
import time
import holoviews as hv
import panel as pn

try:
    from cartopy import crs
    import cartopy.crs as ccrs

    projection = ccrs.PlateCarree()
except:
    warnings.warn(
        "Unable to import cartopy. For better plots install cartopy or check cartopy installation!"
    )
    projection = None

import pandas as pd
import hvplot.pandas
import hvplot.xarray
import hvplot
from bokeh.plotting import show
import xarray as xr
import copy
import numpy as np
import os

from ncplot.utils import get_dims, check_lon, check_lat, normalise_lon

hv.extension("bokeh")
hv.Store.renderers

def as_raster():
    """
    Simple hack to rasterize when datashader is available.
    This is to get around Python 3.11 datashader nonavailability
    """
    try:
        import datashader
        return True
    except:
        return False


def get_coastline(ds, lon_name, lat_name):
    """
    A function to work out an appropriate coastline resolution
    """
    lon_max = ds[lon_name].values.max()
    lon_min = ds[lon_name].values.min()

    lat_max = ds[lat_name].values.max()
    lat_min = ds[lat_name].values.min()

    if ((lon_max - lon_min) * (lat_max - lat_min)) / (360**2) < 0.0016:
        return "10m"

    if ((lon_max - lon_min) * (lat_max - lat_min)) / (360**2) < 0.25:
        return "50m"

    return "110m"


def change_coords(dx):
    """
    Some model output will have repeated zeros as lon/lat. This will mess up quadmesh. Replace lon/lat with indices
    """

    ds = dx.copy()
    df_dims = get_dims(ds)

    lon_name = df_dims.longitude[0]
    lat_name = df_dims.latitude[0]

    lat_df = (
        pd.DataFrame(ds[lat_name].values)
        .melt()
        .groupby("value")
        .size()
        .sort_values(ascending=False)
        .reset_index()
    )
    lat_df.columns = ["value", "number"]

    if lat_df.number[0] / lat_df.number[1] < 10:
        return ds

    if sum(ds[lon_name].values[0] == ds[lon_name].values[1]) == len(ds[lon_name][0]):
        for i in range(0, len(ds[lon_name])):
            ds[lon_name].values[i] = range(0, len(ds[lon_name][0]))
    else:
        for i in range(0, len(ds[lat_name])):
            ds[lon_name].values[i] = np.repeat(i, len(ds[lon_name][i]))

    if sum(ds[lat_name].values[0] == ds[lat_name].values[1]) == len(ds[lon_name][0]):
        for i in range(0, len(ds[lat_name])):
            ds[lat_name].values[i] = range(0, len(ds[lat_name][i]))
    else:
        for i in range(0, len(ds[lat_name])):
            ds[lat_name].values[i] = np.repeat(i, len(ds[lat_name][i]))

    if "x" in list(ds.dims.keys()):
        ds.coords[lon_name].attrs["long_name"] = "x"
        ds.coords[lon_name].attrs["units"] = "x"
        ds = ds.rename({lon_name: "x_coord"})
    else:
        ds = ds.rename({lon_name: "x"})

    if "y" in list(ds.dims.keys()):
        ds.coords[lat_name].attrs["long_name"] = "y"
        ds.coords[lat_name].attrs["units"] = "y"
        ds = ds.rename({lat_name: "y_coord"})
    else:
        ds = ds.rename({lat_name: "y"})

    if "x_coord" in ds.coords:
        ds["x_coord"] = ds["x"]
        ds["y_coord"] = ds["y"]
        ds = ds.rename({"x_coord": "x"}).rename({"y_coord": "y"})

    return ds


def is_curvilinear(ds):
    """
    A function originally in nctoolkit, which used CDO to figure out if a grid was curvilinear. This should
    be rename to something like is_quadmesh
    """

    ds_dims = get_dims(ds)
    if ds_dims.longitude.values[0] is None or ds_dims.latitude.values[0] is None:
        return False

    if len(ds.coords[ds_dims.longitude.values[0]].dims) > 1:
        return True

    return False


def ctrc():
    """
    A basic control to send a message to the terminal
    """
    time.sleep(1)
    print("Press Ctrl+C to stop plotting server")


def in_notebook(out=None):
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    import sys

    if out is not None:
        return True
    if "ipykernel" in sys.modules:
        return True

    if "spyder" in sys.modules:
        return False

    return "ipykernel" in sys.modules


def _looks_like_time_coord(coord, coord_name):
    """
    Check whether a coordinate is likely to represent time.
    """

    if coord_name is None:
        return False

    if coord_name == "time" or coord_name.startswith("time"):
        return True

    if "time" in coord_name:
        return True

    if coord.attrs.get("standard_name") == "time":
        return True

    if coord.attrs.get("axis") == "T":
        return True

    if np.issubdtype(coord.dtype, np.datetime64):
        return True

    values = coord.values
    if len(values) > 0:
        first_value = values.ravel()[0]
        if hasattr(first_value, "year") and hasattr(first_value, "month"):
            return True

    return False


def _find_time_coord(ds):
    """
    Identify the most likely time coordinate name for a dataset.
    """

    df_dims = get_dims(ds)
    time_name = df_dims.time[0]

    if time_name is not None:
        return time_name

    time_candidates = []
    for coord_name in list(ds.coords):
        coord = ds[coord_name]
        if _looks_like_time_coord(coord, coord_name):
            time_candidates.append(coord_name)

    if len(time_candidates) == 1:
        return time_candidates[0]

    return None


def _combine_datasets(datasets):
    """
    Combine multiple datasets by time or by variables.
    """

    if len(datasets) == 1:
        return datasets[0]

    normalized = []
    for ds in datasets:
        time_name = _find_time_coord(ds)
        if time_name is not None and time_name != "time":
            ds = ds.rename({time_name: "time"})
        normalized.append(ds)

    try:
        return xr.combine_by_coords(normalized)
    except Exception:
        try:
            return xr.merge(normalized, compat="override")
        except Exception:
            return xr.concat(normalized, dim="time")


def _normalize_time_coord(ds):
    """
    Normalize detected time coordinate names to ``time``.
    """

    time_name = _find_time_coord(ds)
    if time_name is not None and time_name != "time":
        return ds.rename({time_name: "time"})

    return ds


def _open_multiple_files(files):
    """
    Open multiple files into one dataset.

    This uses ``open_mfdataset`` first, then falls back to manual per-file
    opening and combining if xarray cannot infer a valid combine strategy.
    """

    try:
        return xr.open_mfdataset(
            files,
            combine="by_coords",
            preprocess=_normalize_time_coord,
        )
    except Exception:
        datasets = [xr.open_dataset(ff) for ff in files]
        return _combine_datasets(datasets)


def _load_input_as_dataset(x):
    """
    Load a path, path glob, or sequence of paths into a single dataset.
    """

    if isinstance(x, (list, tuple, set)):
        files = [os.fspath(ff) for ff in x]
        return _open_multiple_files(files)

    if isinstance(x, str) and glob.has_magic(x):
        files = sorted(glob.glob(x))
        if len(files) == 0:
            raise FileNotFoundError(f"No files matched pattern {x}")
        return _open_multiple_files(files)

    return None


def _get_color_limits(ds, vars, kwargs):
    """
    Work out the value range to use for a plot.

    For time-varying plots, this is evaluated across the full dataset so the
    color scale stays consistent between time steps.
    """

    if "clim" in kwargs:
        clim_min, clim_max = kwargs["clim"]
        if clim_min < 0 < clim_max:
            v_max = float(max(-clim_min, clim_max))
            return (-v_max, v_max)

        return (float(clim_min), float(clim_max))

    data = ds[vars]
    data_min = float(data.min(skipna=True).values)
    data_max = float(data.max(skipna=True).values)

    if data_min < 0 < data_max:
        v_max = float(max(-data_min, data_max))
        return (-v_max, v_max)

    return (data_min, data_max)


def _apply_color_limits(plot, ds, vars, autoscale, kwargs):
    """
    Apply a stable color range to a plot when autoscaling is enabled.
    """

    if autoscale:
        return plot.redim.range(**{vars: _get_color_limits(ds, vars, kwargs)})

    return plot


def view(x, vars=None, autoscale=True, out=None, **kwargs):
    """
    Plot the contents of a NetCDF out

    Parameters
    -------------
    x : object or str
        xarray object or file  path
    vars : list or str
        Variables you want to plot. Everything will be plotted if this is not supplied
    autoscale : bool
        Set to False if you do not want the colour scales to vary. Defaults to True, which sets
        scales based on minimum and maximum values.

    out : str
        Name of file if you want to store output

    """

    if "clim" in kwargs:
        ignore = False

        if kwargs["clim"][0] < 0:
            if kwargs["clim"][1] > 0:
                ignore = True

        if not ignore:
            autoscale = False

    coastline = False

    if out is not None:
        if out.endswith(".html") is False:
            raise ValueError("out name must end with html")

    for kk in kwargs.keys():
        if kk == "coast":
            coastline = kwargs[kk]

    if "coast" in kwargs.keys():
        kwargs.pop("coast")
    if type(x) is xr.core.dataarray.DataArray:
        x = x.to_dataset()

    loaded_ds = _load_input_as_dataset(x)
    if loaded_ds is not None:
        x = loaded_ds

    if type(x) is xr.core.dataset.Dataset:
        xr_file = True
    else:
        xr_file = False

    nc_vars = None
    if xr_file is False:
        try:
            try:
                import nctoolkit as nc

                if os.path.exists(x):
                    ds = nc.open_data(x)
                    nc_vars = ds.variables
                    ds = ds.to_xarray()
                else:
                    ds = nc.open_thredds(x)
                    nc_vars = ds.data_vars
                    ds = ds.to_xarray()
            except:
                ds = xr.open_dataset(x)
        except:
            ds = xr.open_dataset(x, decode_times=False)
            warnings.warn("Warning: xarray could not decode times!")
    else:
        ds = x

    if vars is None:
        if nc_vars is not None:
            vars = [x for x in nc_vars]
        else:
            vars = [x for x in list(ds.data_vars)]

    if type(vars) is list:
        ds = ds[vars]

    coord_list = list(ds.coords)


    for cc in coord_list:
        new_coords = list(ds.coords)
        if cc not in new_coords:
            continue
        if len(ds[cc].values.ravel()) <= 1:
            if cc in list(ds.dims):
                ds = ds.squeeze(cc, drop=True)

    coord_list = list(ds.dims)

    for cc in coord_list:
        if len(ds[cc].values.ravel()) <= 1:
            if cc in list(ds.dims):
                ds = ds.squeeze(cc, drop=True)

    quadmesh = False

    if len(list(ds.dims)) + len(list(ds.coords)) == 0:
        raise ValueError("There are no dimensions or coordinates in the dataset!")

    df_dims = get_dims(ds)
    # figure out number of points
    lon_name = df_dims.longitude[0]
    lat_name = df_dims.latitude[0]

    if lon_name is not None:
        if "long_name" in ds[lon_name].attrs:
            if "rotate" in ds[lon_name].long_name:
                coastline = False

            if "pole" in ds[lon_name].long_name:
                coastline = False

    bad = []



    if True:
        dims = list(ds.dims)
        coords = ds.coords
        coords_list = list(coords)
        selected = []
        for x in coords_list:
            selected += list(coords[x].dims)
        selected = list(set(selected))
        add_these = [x for x in dims if x not in selected]

        if len(add_these) > 0:
            for x in add_these:
                if x not in coords_list and "vert" not in x:
                    ds[x] = ds[x]
            df_dims = get_dims(ds)
            lon_name = df_dims.longitude[0]
            lat_name = df_dims.latitude[0]

            if lon_name is not None:
                if "long_name" in ds[lon_name].attrs:
                    if "rotate" in ds[lon_name].long_name:
                        coastline = False

                    if "pole" in ds[lon_name].long_name:
                        coastline = False

    if len([x for x in ds.coords if "lon" in x]) > len(
        [x for x in ds.dims if "lon" in x]
    ):
        coastline = False

    if quadmesh is False and lat_name is not None:
        lats = ds[lat_name].values
        if len(lats) > 1:

            max_step = np.max([np.abs(x) for x in (lats[0:-2] - lats[1:-1])])
            min_step = np.min([np.abs(x) for x in (lats[0:-2] - lats[1:-1])])
            if min_step == 0:
                quadmesh = True
            else:
                if max_step / min_step > 1.001:
                    quadmesh = True

    switch_coords = False

    try:
        #import datashader
        rasterize = True
    except:
        rasterize = False

    if quadmesh:
        switch_coords = True
        if len(np.shape(ds[lon_name].values)) == 1:
            switch_coords = False

        if switch_coords:
            if lon_name != check_lon(lon_name, ds):
                lon_name = check_lon(lon_name, ds)
                rasterize = False
                switch_coords = False
                quadmesh = False
                coastline = False
            if lat_name != check_lat(lat_name, ds):
                lat_name = check_lat(lat_name, ds)
                rasterize = False
                quadmesh = False

    if switch_coords:
        orig_coords = list(ds.coords)
        ds = change_coords(ds)
        new_coords = list(ds.coords)
        quadmesh = False
        df_dims = get_dims(ds)
        # figure out number of points
        lon_name = df_dims.longitude[0]
        lat_name = df_dims.latitude[0]
        if orig_coords != new_coords:
            coastline = False
            rasterize = False

    if lon_name is not None:
        if len(ds[lon_name].dims) > 1:
            quadmesh = True

    n_points = 1
    if lon_name is not None:
        if len(ds[lon_name].values) > 1:
            n_points += len(ds[lon_name].values)

    if lat_name is not None:
        if len(ds[lat_name].values) > 1:
            n_points += len(ds[lat_name].values)

    # figure out number of times

    time_name = df_dims.time[0]

    # time name maybe cannot be parsed. If "time" is among the coords, use that

    if time_name is None:
        candidates = list()
        n_candidates = 0
        for x in list(ds.coords):
            if x.startswith("time"):
                candidates.append(x)
                n_candidates += 0
        if len(candidates) == 1:
            time_name = candidates[0]

    if time_name is None:
        n_times = 0
    else:
        n_times = len(ds[time_name].values)

    if lat_name is not None:
        if len(ds[lat_name].values) > 1:
            n_points += len(ds[lat_name].values)

    if lon_name is not None:
        if "lon" not in lon_name:
            coastline = False

    # coastlines are drawn on a -180-180 map, so longitudes must follow that convention
    if coastline:
        ds = normalise_lon(ds, lon_name)

    ff_dims = list(ds.coords)

    possible_others = [x for x in ff_dims if x not in df_dims.columns]

    if len(possible_others) == 0:
        n_levels = 1
    else:
        if ds[possible_others[0]].values.ndim == 0:
            n_levels = 0
        else:
            n_levels = len(ds[possible_others[0]].values)


        # also must have all of the coordinates...

    # code below figures out what is a variable, not a coordinate. Could be improved...

    coord_list = list(ds.coords)

    for cc in coord_list:
        if len(ds[cc].values.ravel()) <= 1:
            if cc in list(ds.dims):
                ds = ds.squeeze(cc, drop=True)

    if type(vars) is list:
        new_vars = []
        for vv in vars:
            set_coords = set(
                [x for x in list(ds.coords) if "vertice" not in x and "bnds" not in x]
            )
            if set(list(ds[vv].coords)) == set_coords:
                new_vars.append(vv)
        vars = new_vars

    if type(vars) is list:
        if len(vars) == 1:
            vars = vars[0]

    if type(vars) is list:
        for vv in vars:
            if vv not in list(ds.data_vars):
                raise ValueError(f"{vv} is not a valid variable")
    else:
        if vars not in list(ds.data_vars):
            raise ValueError(f"{vars} is not a valid variable")

    if (lon_name is not None) and (lat_name is not None) and type(vars) is list:
        new_vars = []
        for x in vars:

            dims_required = []
            for i in list(ds.coords):
                dims_required += ds.coords[i].dims
            dims_required = set(dims_required)

            if set(ds[x].dims) == dims_required:
                new_vars.append(x)
        vars = new_vars

    if len(vars) == 1:
        vars = vars[0]

    if (lon_name is None) or (lat_name is None):
        rasterize = False

    # Case when all you can plot is a time series

    # heat map 1

    # get rid of coordinates without multiple values

    coord_list = list(ds.coords)

    
    for cc in coord_list:
        new_coords = list(ds.coords)
        if cc not in new_coords:
            continue
        if len(ds[cc].values.ravel()) <= 1:
            if cc in list(ds.dims):
                ds = ds.squeeze(cc, drop=True)
                
    coord_list = list(ds.coords)

    coord_list = [x for x in coord_list if x in list(ds.dims)]

    coord_df = pd.DataFrame(
        {"coord": coord_list, "coord_length": [len(ds.coords[x].values) for x in coord_list]}
    )

    # It's possible there are still 2 time variables in the dimensions which could cause problems...

    for tt in [x for x in ds.coords if "time" in x and x != time_name]:
        ds = ds.drop_vars(tt)

    # work out if it should be a spatial map

    spatial_map = False

    # line plot

    if lon_name is not None and lat_name is not None:

        if lon_name in list(ds.coords) and lat_name in list(ds.coords):
            if (len(ds[lon_name].values) > 1) and (len(ds[lat_name].values) > 1):
                spatial_map = True

    if len([x for x in coord_df.coord_length if x > 1]) == 1 and spatial_map is False:

        df = ds.to_dataframe()
        if nc_vars is not None:
            cols = [x for x in list(df.columns) if x in nc_vars]
            df = df.loc[:, cols]

        df = df.reset_index().drop_duplicates().reset_index()

        if type(vars) is str:
            vars = [vars]

        for x in list(ds.coords):
            if len(ds.coords[x].values) > 1:
                x_var = x
                break

        selection = [x for x in df.columns if x in vars or x == x_var]

        df = df.loc[:, selection].melt(x_var).drop_duplicates().set_index(x_var)

        intplot = df.hvplot(
            groupby="variable",
            dynamic=True,
            #responsive=(in_notebook() is False),
            responsive = False,
            **kwargs,
        )
        if in_notebook(out):
            if out is None:
                return intplot
            else:
                hvplot.save(intplot, out)
                return None

        t = Thread(target=ctrc)
        t.start()

        return None

    # heat map where 2 coords have more than 1 value, not a spatial map
    if len([x for x in coord_df.coord_length if x > 1]) == 2 and spatial_map is False:

        df = ds.to_dataframe().reset_index()
        x_var = coord_df.query("coord_length > 1").reset_index().coord[0]
        y_var = coord_df.query("coord_length > 1").reset_index().coord[1]
        selection = [x for x in df.columns if x in vars or x == x_var or x == y_var]

        df = df.loc[:, selection].melt([x_var, y_var]).drop_duplicates()

        # we need to make sure this is not 2d geographic data
        case1 = 0
        if lon_name is not None and lon_name in list(ds.coords):
            if len(ds.coords[lon_name].values) > 1:
                case1 += 1

        if lat_name is not None and lat_name in list(ds.coords):
            if len(ds.coords[lat_name].values) > 1:
                case1 += 1

        if case1 <= 1:

            if type(vars) is list:
                # figure out if it needs to be quadmesh
                if x_var == time_name:
                    quadmesh = True
                x_vals = ds[x_var].values

                if len(x_vals) > 2:

                    if (
                        np.nanmax(x_vals[0:-2] - x_vals[1:-1]).astype("float")
                        - np.nanmin(x_vals[0:-2] - x_vals[1:-1]).astype("float")
                        > 0
                    ):
                        quadmesh = True
                y_vals = ds[y_var].values
                if len(y_vals) > 2:
                    if (
                        np.nanmax(y_vals[0:-2] - y_vals[1:-1]).astype("float")
                        - np.nanmin(y_vals[0:-2] - y_vals[1:-1]).astype("float")
                        > 0
                    ):
                        quadmesh = True

                if quadmesh:
                    intplot = ds.hvplot.quadmesh(
                        x_var, y_var, vars, cmap="viridis", **kwargs
                    )
                else:
                    intplot = ds.hvplot.image(
                        x_var, y_var, vars, cmap="viridis", **kwargs
                    )

            else:

                if autoscale:

                    self_max = ds.rename({vars: "x"}).x.max()
                    self_min = ds.rename({vars: "x"}).x.min()
                    v_max = float(max(self_max.values, -self_min.values))
                    if "clim" in kwargs:
                        v_max = float(max(-kwargs["clim"][0], kwargs["clim"][1]))

                else:
                    self_max = 1
                    self_min = 1

                # figure out if it needs to be quadmesh
                if x_var == time_name:
                    quadmesh = True
                if not quadmesh:
                    x_vals = ds[x_var].values

                    if len(x_vals) > 2:
                        if (
                            np.nanmax(x_vals[0:-2] - x_vals[1:-1]).astype("float")
                            - np.nanmin(x_vals[0:-2] - x_vals[1:-1]).astype("float")
                            > 0
                        ):
                            quadmesh = True
                if not quadmesh:
                    y_vals = ds[y_var].values

                    if len(y_vals) > 2:
                        if (
                            np.nanmax(y_vals[0:-2] - y_vals[1:-1]).astype("float")
                            - np.nanmin(y_vals[0:-2] - y_vals[1:-1]).astype("float")
                            > 0
                        ):
                            quadmesh = True

                if self_max > 0 and self_min < 0:
                    if quadmesh:
                        intplot = ds.hvplot.quadmesh(
                            x_var,
                            y_var,
                            vars,
                            cmap="RdBu_r",
                            #responsive=(in_notebook() is False),
                            responsive = False,
                            **kwargs,
                        ).redim.range(**{vars: (-v_max, v_max)})
                    else:
                        intplot = ds.hvplot.image(
                            x_var,
                            y_var,
                            vars,
                            cmap="RdBu_r",
                            rasterize=False,
                            #responsive=(in_notebook() is False),
                            responsive = False,
                            **kwargs,
                        ).redim.range(**{vars: (-v_max, v_max)})

                else:
                    if quadmesh:
                        intplot = ds.hvplot.quadmesh(
                            x_var, y_var, vars, cmap="viridis", **kwargs
                        )
                    else:
                        intplot = ds.hvplot.image(
                            x_var, y_var, vars, cmap="viridis", **kwargs
                        )

            if in_notebook(out):
                if out is None:
                    return intplot
                else:
                    hvplot.save(intplot, out)
                    return None
            if out is not None:
                hvplot.save(intplot, out)
                return None

            t = Thread(target=ctrc)
            t.start()
            return None

    # heat map where 3 coords have more than 1 value, and one of them is time. Not a spatial map though
    if len([x for x in coord_df.coord_length if x > 1]) == 3:

        non_map = True

        if lon_name is not None and lat_name is not None:
            if (lon_name is not None) and (lon_name in list(ds.coords)):
                lons = int(coord_df.query("coord == @lon_name").coord_length.values[0])
            else:
                lons = 0

            if (lat_name) is not None and (lat_name in list(ds.coords)):
                lats = int(coord_df.query("coord == @lat_name").coord_length.values[0])
            else:
                lats = 0

            if lons > 1 and lats > 1:
                non_map = False

        time_in = False

        possible = 0

        for x in coord_list:
            if "time" in x:
                time_in = True
                possible += 1

        if time_name in coord_list and time_in and non_map:

            if coord_df.query("coord == @time_name").coord_length.values[0] > 1:

                df = ds.to_dataframe().reset_index()
                for x in list(ds.coords):
                    if (len(ds.coords[x].values) > 1) and (x != time_name):
                        x_var = x
                        break

                for x in list(ds.coords):
                    if (
                        (len(ds.coords[x].values) > 1)
                        and (x is not x_var)
                        and (x != time_name)
                    ):
                        y_var = x
                        break

                selection = [
                    x
                    for x in df.columns
                    if x in vars or x == x_var or x == y_var or x == time_name
                ]

                df = df.loc[:, selection]
                df = df.melt([x_var, y_var, time_name]).drop_duplicates()
                case1 = 0
                if lon_name is not None and lon_name in list(ds.coords):
                    if len(ds.coords[lon_name].values) > 1:
                        case1 += 1

                if lat_name is not None and lat_name in list(ds.coords):
                    if len(ds.coords[lat_name].values) > 1:
                        case1 += 1

                if case1 <= 1:

                    # this could also do with a blue to red option

                    # figure out if it should be quadmesh
                    quadmesh = False
                    x_vals = ds[x_var].values
                    if len(x_vals) > 2:
                        if (
                            np.nanmax(x_vals[0:-2] - x_vals[1:-1]).astype("float")
                            - np.nanmin(x_vals[0:-2] - x_vals[1:-1]).astype("float")
                            > 0
                        ):
                            quadmesh = True

                    y_vals = ds[y_var].values
                    if len(y_vals) > 2:
                        if (
                            np.nanmax(y_vals[0:-2] - y_vals[1:-1]).astype("float")
                            - np.nanmin(y_vals[0:-2] - y_vals[1:-1]).astype("float")
                            > 0
                        ):
                            quadmesh = True

                    if quadmesh:
                        intplot = ds.hvplot.quadmesh(
                            x_var, y_var, vars, cmap="viridis", rasterize=rasterize, **kwargs
                        )
                    else:
                        intplot = ds.hvplot.image(
                            x_var,
                            y_var,
                            vars,
                            cmap="viridis",
                            rasterize=False,
                            **kwargs,
                        )

                    if in_notebook(out):
                        if out is None:
                            return intplot
                        else:
                            hvplot.save(intplot, out)
                            return None

                    t = Thread(target=ctrc)
                    t.start()
                    return None

    if (n_times > 1) and (n_points < 2) and (n_levels <= 1):

        if type(vars) is str:
            vars = [vars]

        df = ds

        dim_dict = dict(df.coords)
        to_go = []
        for kk in dim_dict.keys():
            if dim_dict[kk] == 1:
                df = df.squeeze(kk)
                to_go.append(kk)

        df = df.to_dataframe()
        keep = vars
        df = df.reset_index()

        to_go = [x for x in df.columns if (x not in [time_name] and x not in keep)]

        df = df.drop(columns=to_go).drop_duplicates()

        intplot = (
            df.reset_index()
            .set_index(time_name)
            .loc[:, vars]
            .reset_index()
            .melt(time_name)
            .set_index(time_name)
            .hvplot(
                groupby="variable",
                dynamic=True,
                #responsive=(in_notebook() is False),
                responsive = False,
                **kwargs,
            )
        )
        if in_notebook(out):
            if out is None:
                return intplot
            else:
                hvplot.save(intplot, out)
                return None

        t = Thread(target=ctrc)
        t.start()

        return None

    if (n_points > 1) and (n_levels >= 1) and (type(vars) is list):

        if quadmesh:

            if coastline:
                coastline = get_coastline(ds, lon_name, lat_name)
                try:
                    projection = ccrs.PlateCarree()
                except:
                    projection = None
                    coastline = False

            else:
                projection = None

            intplot = ds.hvplot.quadmesh(
                lon_name,
                lat_name,
                vars,
                dynamic=True,
                cmap="viridis",
                coastline=coastline,
                projection=projection,
                rasterize=rasterize,
                #responsive=in_notebook() is False,
                responsive = False,

                **kwargs,
            )
        else:

            if coastline:
                coastline = get_coastline(ds, lon_name, lat_name)
                try:
                    projection = ccrs.PlateCarree()
                except:
                    projection = None
                    coastline = False
            else:
                projection = None

            intplot = ds.hvplot.image(
                lon_name,
                lat_name,
                vars,
                dynamic=True,
                cmap="viridis",
                coastline=coastline,
                rasterize=False,
                projection=projection,
                #responsive=in_notebook() is False,
                responsive = False,
                **kwargs,
            )

        if in_notebook(out):
            if out is None:
                return intplot
            else:
                hvplot.save(intplot, out)
                return None

        t = Thread(target=ctrc)
        t.start()
        return None

    if n_points > 1:

        if type(vars) is list:
            warnings.warn(message="Warning: Only the first variable is mapped")
            vars = vars[0]

        if autoscale:
            self_max = ds.rename({vars: "x_dim12345"}).x_dim12345.max()
            self_min = ds.rename({vars: "x_dim12345"}).x_dim12345.min()
            v_max = max(self_max.values, -self_min.values)
            if "clim" in kwargs:
                v_max = float(max(-kwargs["clim"][0], kwargs["clim"][1]))
        else:
            self_max = 1
            self_min = 1

        if (self_max > 0) and (self_min < 0):
            if quadmesh:
                if coastline:
                    coastline = get_coastline(ds, lon_name, lat_name)
                    try:
                        projection = ccrs.PlateCarree()
                    except:
                        projection = None
                        coastline = False
                else:
                    projection = None

                if autoscale:
                    intplot = ds.hvplot.quadmesh(
                        lon_name,
                        lat_name,
                        vars,
                        dynamic=True,
                        cmap="RdBu_r",
                        coastline=coastline,
                        projection=projection,
                        rasterize=rasterize,
                        #responsive=(in_notebook() is False),
                        responsive = False,
                        **kwargs,
                    )
                    intplot = _apply_color_limits(intplot, ds, vars, autoscale, kwargs)
                else:
                    intplot = ds.hvplot.quadmesh(
                        lon_name,
                        lat_name,
                        vars,
                        dynamic=True,
                        cmap="RdBu_r",
                        coastline=coastline,
                        projection=projection,
                        rasterize=rasterize,
                        #responsive=(in_notebook() is False),
                        responsive = False,
                        **kwargs,
                    )
                    intplot = _apply_color_limits(intplot, ds, vars, autoscale, kwargs)
            else:
                if coastline:
                    coastline = get_coastline(ds, lon_name, lat_name)
                    try:
                        projection = ccrs.PlateCarree()
                    except:
                        projection = None
                        coastline = False
                else:
                    projection = None

                if autoscale:
                    intplot = ds.hvplot.image(
                        lon_name,
                        lat_name,
                        vars,
                        dynamic=True,
                        coastline=coastline,
                        projection=projection,
                        rasterize=False,
                        cmap="RdBu_r",
                        #responsive=(in_notebook() is False),
                        responsive = False,
                        **kwargs,
                    )
                    intplot = _apply_color_limits(intplot, ds, vars, autoscale, kwargs)
                else:
                    intplot = ds.hvplot.image(
                        lon_name,
                        lat_name,
                        vars,
                        dynamic=True,
                        coastline=coastline,
                        projection=projection,
                        rasterize=False,
                        cmap="RdBu_r",
                        #responsive=(in_notebook() is False),
                        responsive = False,
                        **kwargs,
                    )

            if in_notebook(out):
                if out is None:
                    return intplot
                else:
                    hvplot.save(intplot, out)
                    return None

            t = Thread(target=ctrc)
            t.start()
            return None

        else:
            if quadmesh:
                if coastline:
                    coastline = get_coastline(ds, lon_name, lat_name)
                    try:
                        projection = ccrs.PlateCarree()
                    except:
                        projection = None
                        coastline = False
                else:
                    projection = None

                if autoscale:
                    intplot = ds.hvplot.quadmesh(
                        lon_name,
                        lat_name,
                        vars,
                        dynamic=True,
                        cmap="viridis",
                        coastline=coastline,
                        projection=projection,
                        rasterize=rasterize,
                        #responsive=(in_notebook() is False),
                        responsive = False,
                        **kwargs,
                    )
                    intplot = _apply_color_limits(intplot, ds, vars, autoscale, kwargs)
                else:
                    intplot = ds.hvplot.quadmesh(
                        lon_name,
                        lat_name,
                        vars,
                        dynamic=True,
                        cmap="viridis",
                        coastline=coastline,
                        projection=projection,
                        rasterize=rasterize,
                        #responsive=(in_notebook() is False),
                        responsive = False,
                        **kwargs,
                    )
                    intplot = _apply_color_limits(intplot, ds, vars, autoscale, kwargs)

            else:

                if coastline:
                    coastline = get_coastline(ds, lon_name, lat_name)
                    try:
                        projection = ccrs.PlateCarree()
                    except:
                        projection = None
                        coastline = False
                else:
                    projection = None

                if autoscale:
                    intplot = ds.hvplot.image(
                        lon_name,
                        lat_name,
                        vars,
                        dynamic=True,
                        cmap="viridis",
                        coastline=coastline,
                        rasterize=False,
                        projection=projection,
                        #responsive=(in_notebook() is False),
                        responsive = False,
                        **kwargs,
                    )
                    intplot = _apply_color_limits(intplot, ds, vars, autoscale, kwargs)
                else:
                    intplot = ds.hvplot.image(
                        lon_name,
                        lat_name,
                        vars,
                        dynamic=True,
                        cmap="viridis",
                        coastline=coastline,
                        rasterize=False,
                        projection=projection,
                        #responsive=(in_notebook() is False),
                        responsive = False,
                        **kwargs,
                    )
                    intplot = _apply_color_limits(intplot, ds, vars, autoscale, kwargs)

            if in_notebook(out):
                if out is None:
                    return intplot
                else:
                    hvplot.save(intplot, out)
                    return None

            t = Thread(target=ctrc)
            t.start()

            return None

    # it's possible that the data has not spatial or temporal coordinates. Do something about it....

    # Throw an error if case has not plotting method available yet
    # right now this only seems to be when you have lon/lat/time/levels and multiple variables
    # maybe needs an appropriate

    raise ValueError("Autoplotting method for this type of data is not yet available!")
