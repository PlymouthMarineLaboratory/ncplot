"""Curvilinear grids and FVCOM's mesh, put on regular longitude-latitude grids."""

import numpy as np
import pytest
import xarray as xr

from ncplot import regrid as regrid_module
from ncplot.grid import raster_for
from ncplot.loading import open_one
from ncplot.mesh import Mesh
from ncplot.regrid import find_lonlat, native_range, regrid_dataset

from conftest import (
    ALL_FILES, CURVILINEAR_FILES, FVCOM_FILE, REGULAR_FILES, file_id, variable_of,
)

REGRIDDED = CURVILINEAR_FILES + [FVCOM_FILE]


def frame(ds, path, **select):
    """The first time step and level of the variable, as it is on its own grid."""
    name = variable_of(path)
    select = {"time": 0, **({"siglay": 0} if path == FVCOM_FILE else {}), **select}
    return ds[name].isel(select)


@pytest.mark.parametrize("path", REGULAR_FILES, ids=file_id)
def test_regular_grids_are_left_alone(path):
    assert regrid_dataset(open_one(path)) == (None, [])


def test_other_variables_are_left_alone():
    ds = xr.Dataset({"sst": ("time", np.arange(5.0))})
    assert regrid_dataset(ds) == (None, [])


@pytest.mark.parametrize("path", REGRIDDED, ids=file_id)
def test_the_result_is_on_a_regular_grid(path):
    out, notes = regrid_dataset(open_one(path))
    assert notes == []
    variable = out[variable_of(path)]
    assert variable.dims[-2:] == ("lat", "lon")
    assert out["lat"].ndim == out["lon"].ndim == 1
    assert out["lat"].attrs["standard_name"] == "latitude" and out["lon"].attrs["units"] == "degrees_east"
    assert np.all(np.diff(out["lat"].values) > 0) and np.all(np.diff(out["lon"].values) > 0)
    assert variable.dtype == np.float32
    assert out.sizes["time"] == 2
    # the units and names come through
    assert variable.attrs.get("units")


@pytest.mark.parametrize("path", CURVILINEAR_FILES, ids=file_id)
def test_a_curvilinear_frame_is_the_grid_on_the_regular_grid(path):
    ds = open_one(path)
    out, _ = regrid_dataset(ds)
    native = frame(ds, path, time=1)
    lon_name, lat_name = find_lonlat(ds, "tos")
    raster = raster_for(ds[lon_name].values, ds[lat_name].values)
    expected = raster.image(native.transpose(*ds[lon_name].dims).values).astype("float32")
    got = out["tos"].isel(time=1).values
    assert got.shape == expected.shape
    np.testing.assert_array_equal(np.nan_to_num(got, nan=-999), np.nan_to_num(expected, nan=-999))
    # the map covers the globe, and nothing is made up
    assert out["lat"].min() < -60 and out["lat"].max() > 60 and float(out["lon"].max() - out["lon"].min()) > 330
    assert np.nanmin(got) >= float(native.min()) - 1e-4 and np.nanmax(got) <= float(native.max()) + 1e-4


def test_fvcom_nodes_and_elements():
    ds = open_one(FVCOM_FILE)
    out, _ = regrid_dataset(ds)
    raster = Mesh.from_dataset(ds).raster()
    for name, location in (("temp", "node"), ("salinity", "node"), ("zeta", "node"), ("u", "nele"), ("v", "nele")):
        select = {"time": 1, "siglay": 2} if "siglay" in ds[name].dims else {"time": 1}
        expected = raster.image(ds[name].isel(select).values, location).astype("float32")
        got = out[name].isel(select).values
        np.testing.assert_array_equal(np.nan_to_num(got, nan=-999), np.nan_to_num(expected, nan=-999))
    # what describes the mesh is not data
    assert not {"nv", "lon", "lat", "Itime", "Itime2", "siglay"} & set(out.data_vars)


def test_a_dimension_without_a_coordinate_gets_one_and_time_comes_through():
    ds = open_one(FVCOM_FILE)
    out, _ = regrid_dataset(ds, "temp")
    assert list(out["siglay"].values) == [1, 2, 3]
    assert np.issubdtype(out["time"].dtype, np.datetime64)
    assert out["time"].values[0] == ds["time"].values[0]


@pytest.mark.parametrize("path", REGRIDDED, ids=file_id)
def test_nothing_is_read_until_a_frame_is_asked_for(path, monkeypatch):
    reads = []
    original = regrid_module.RasterArray._get
    monkeypatch.setattr(regrid_module.RasterArray, "_get", lambda self, key: reads.append(key) or original(self, key))
    out, _ = regrid_dataset(open_one(path))
    variable = out[variable_of(path)]
    assert reads == []
    select = {"time": 1, **({"siglay": 2} if path == FVCOM_FILE else {})}
    chosen = variable.isel(select)
    assert reads == []
    values = chosen.values
    # one read, of that frame alone
    assert len(reads) == 1 and values.shape == variable.shape[-2:]


def test_the_ways_of_indexing_a_regridded_variable():
    ds = open_one(FVCOM_FILE)
    out, _ = regrid_dataset(ds, "temp")
    whole = out["temp"].values
    assert whole.shape == (2, 3, *out["temp"].shape[-2:])
    np.testing.assert_array_equal(whole[1, 2], out["temp"].isel(time=1, siglay=2).values)
    np.testing.assert_array_equal(whole[:, 1], out["temp"].isel(siglay=1).values)
    np.testing.assert_array_equal(whole[0, :, 10:20, 5:50], out["temp"].isel(time=0, lat=slice(10, 20), lon=slice(5, 50)).values)
    np.testing.assert_array_equal(whole[[1, 0], 2], out["temp"].isel(time=[1, 0], siglay=2).values)
    np.testing.assert_array_equal(whole[0, 0, [3, 7], 5], out["temp"].isel(time=0, siglay=0, lat=[3, 7], lon=5).values)
    assert out["temp"].mean().notnull()


def test_variables_can_be_chosen():
    ds = open_one(FVCOM_FILE)
    out, notes = regrid_dataset(ds, "u")
    assert list(out.data_vars) == ["u"] and notes == []
    out, notes = regrid_dataset(ds, ["temp", "h"])
    assert set(out.data_vars) == {"temp", "h"} and "siglay" in out["temp"].dims and out["h"].dims == ("lat", "lon")
    out, notes = regrid_dataset(ds, ["temp", "nv"])
    assert list(out.data_vars) == ["temp"] and any("nv" in note for note in notes)


def test_a_grids_bounds_are_not_data():
    ds = open_one(CURVILINEAR_FILES[0])
    ds["tos_bnds"] = ((*ds["tos"].dims[1:], "nvertex"), np.zeros((*ds["tos"].shape[1:], 4)))
    ds["lat_vertices"] = ds["tos_bnds"]
    out, _ = regrid_dataset(ds)
    assert list(out.data_vars) == ["tos"]


def test_a_grid_that_cannot_place_its_cells_is_left_as_it_is():
    # every cell at 0, 0 (a model that does not say where its cells are)
    ds = xr.Dataset(
        {"tos": (("y", "x"), np.random.rand(6, 7))},
        coords={"lon": (("y", "x"), np.zeros((6, 7))), "lat": (("y", "x"), np.zeros((6, 7)))},
    )
    out, notes = regrid_dataset(ds)
    assert out is None


def test_the_range_of_the_values_is_found_from_the_grid_they_are_on():
    ds = open_one(FVCOM_FILE)
    low, high = native_range(ds, "temp")
    assert low == pytest.approx(float(ds["temp"].min())) and high == pytest.approx(float(ds["temp"].max()))
    assert native_range(xr.Dataset({"x": ("a", [np.nan, np.nan])}), "x") is None
