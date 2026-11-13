"""Make the small NetCDF files in ``tests/data`` from full-size model output.

This is a developer tool: it needs the original (large) files and is NOT run
by the tests or by CI -- the files it writes are committed to the repository.

    python tests/make_test_data.py [--cmip6-dir DIR] [--fvcom-file FILE] [--out DIR]

CMIP6 -- one ``tos`` file for each distinct horizontal-grid layout found in
the CMIP6 test collection.  Every ``STRIDE``-th point is kept in each
horizontal dimension, so the whole globe (including the seam of a 0-360 grid
and the tripolar fold of an ocean grid) is still there, just coarser.

FVCOM -- the nodes inside a lon/lat box, the elements whose three nodes are
all in the box (``nv`` renumbered), two times and three sigma layers.
"""

import argparse
import os

import numpy as np
import xarray as xr

CMIP6_DIR = "/data/proteus1/scratch/rwi/nctoolkit_testing/cmip6_plotting/data/cmip6"
FVCOM_FILE = (
    "/data/thaumus2/scratch/maz/Models/FVCOM/run/aqua_v16_trawl/output/aqua_v16_trawl/"
    "tide_wind_nest_ersem/2012_2014_val/aqua_v16_trawl_avg_0001_2012_01.nc"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

STRIDE = 4

# name of the test file -> the source file, one for each grid layout
CMIP6_FILES = {
    # 1-D lat/lon
    "cmip6_latlon_1d.nc": "tos_Amon_EC-Earth3-CC_historical_r1i1p1f1_gr_185001-185012.nc",
    "cmip6_latitude_longitude_1d.nc": "tos_Omon_MCM-UA-1-0_ssp126_r1i1p1f2_gn_201501-210012.nc",
    # 2-D lon/lat on different dimension names
    "cmip6_ji.nc": "tos_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc",
    "cmip6_latlon_2d.nc": "tos_Omon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc",
    "cmip6_nlat_nlon.nc": "tos_Omon_CESM2_ssp126_r10i1p1f1_gn_201501-206412.nc",
    "cmip6_ij.nc": "tos_Omon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc",
    "cmip6_yx.nc": "tos_Omon_CNRM-CM6-1_historical_r10i1p1f2_gn_185001-201412.nc",
    # 2-D coordinates called nav_lon / nav_lat
    "cmip6_nav_lonlat.nc": "tos_Omon_IPSL-CM6A-LR_ssp126_r14i1p1f1_gn_201501-210012.nc",
}

FVCOM_BOX = (-5.0, -3.0, 50.0, 51.0)  # lon0, lon1, lat0, lat1
FVCOM_TIMES = 2
FVCOM_LAYERS = 3
FVCOM_NODE_VARS = ["temp", "salinity", "zeta"]
FVCOM_ELEMENT_VARS = ["u", "v"]

COMPRESSION = dict(zlib=True, complevel=6)


def _encoding(ds):
    return {name: dict(COMPRESSION) for name, var in ds.data_vars.items() if var.dtype.kind in "fi" and var.ndim}


def make_cmip6(source, target):
    ds = xr.open_dataset(source, decode_times=False, mask_and_scale=False)
    name = next(v for v in ds.data_vars if v == "tos")
    horizontal = ds[name].dims[1:]
    ds = ds.isel({dim: slice(None, None, STRIDE) for dim in horizontal})
    # bounds describe the full-size grid; they no longer match
    drop = [v for v in ds.variables if "bnds" in v or "vertices" in v or v == "area"]
    ds = ds.drop_vars(drop)
    for var in ds.variables.values():
        var.attrs.pop("bounds", None)
        var.encoding.pop("chunksizes", None)
        var.encoding.pop("contiguous", None)
    ds = ds[[name]].assign_coords({c: ds[c] for c in ds.coords})
    ds.attrs = {"title": "ncplot test data (subsampled CMIP6)", "source_file": os.path.basename(source)}
    ds.to_netcdf(target, encoding=_encoding(ds), unlimited_dims=["time"])
    return ds


def make_fvcom(source, target, box=FVCOM_BOX):
    ds = xr.open_dataset(source, decode_times=False)
    lon, lat = ds["lon"].values, ds["lat"].values
    keep = (lon >= box[0]) & (lon <= box[1]) & (lat >= box[2]) & (lat <= box[3])
    nodes = np.flatnonzero(keep)
    nv = ds["nv"].values - 1  # (three, nele), 0-based
    elements = np.flatnonzero(keep[nv].all(axis=0))
    # old node number -> new node number
    renumber = np.full(keep.size, -1)
    renumber[nodes] = np.arange(nodes.size)

    sub = ds.isel(time=slice(0, FVCOM_TIMES), siglay=slice(0, FVCOM_LAYERS))
    out = {}
    for name in ["lon", "lat", "x", "y", "h"]:
        out[name] = sub[name].isel(node=nodes)
    for name in ["lonc", "latc", "xc", "yc", "h_center"]:
        out[name] = sub[name].isel(nele=elements)
    out["siglay"] = sub["siglay"].isel(node=nodes)
    out["nv"] = xr.DataArray(
        renumber[nv[:, elements]] + 1, dims=("three", "nele"), attrs=ds["nv"].attrs
    ).astype("int32")
    for name in FVCOM_NODE_VARS:
        out[name] = sub[name].isel(node=nodes)
    for name in FVCOM_ELEMENT_VARS:
        out[name] = sub[name].isel(nele=elements)
    out["time"] = sub["time"]
    out["Itime"] = sub["Itime"]
    out["Itime2"] = sub["Itime2"]
    result = xr.Dataset(out)
    result.attrs = {"title": "ncplot test data (corner of an FVCOM run)", "source_file": os.path.basename(source)}
    for var in result.variables.values():
        var.encoding = {}
        var.attrs.pop("grid", None)
        var.attrs.pop("coordinates", None)
    result.to_netcdf(target, encoding=_encoding(result), unlimited_dims=["time"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cmip6-dir", default=CMIP6_DIR)
    parser.add_argument("--fvcom-file", default=FVCOM_FILE)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for target, source in CMIP6_FILES.items():
        make_cmip6(os.path.join(args.cmip6_dir, source), os.path.join(args.out, target))
    make_fvcom(args.fvcom_file, os.path.join(args.out, "fvcom_small.nc"))
    total = 0
    for name in sorted(os.listdir(args.out)):
        if name.endswith(".nc"):
            size = os.path.getsize(os.path.join(args.out, name))
            total += size
            print(f"{name:35s} {size / 1024:8.1f} kB")
    print(f"{'total':35s} {total / 1024:8.1f} kB")


if __name__ == "__main__":
    main()
