import numpy as np
import xarray as xr

from ncplot.utils import normalise_lon


def _make_ds(lons):
    lons = np.asarray(lons, dtype="float32")
    ds = xr.Dataset(
        {"x": (("lat", "lon"), np.tile(lons, (2, 1)))},
        coords={"lat": [0.0, 1.0], "lon": lons},
    )
    ds["lon"].attrs["units"] = "degrees_east"
    return ds


class TestNormaliseLon:
    def test_global_0_360_is_converted(self):
        ds = _make_ds(np.arange(0.5, 360, 1.0))
        out = normalise_lon(ds, "lon")

        assert out.lon.values[0] == -179.5
        assert out.lon.values[-1] == 179.5
        assert np.all(np.diff(out.lon.values) > 0)
        assert out.lon.attrs["units"] == "degrees_east"
        # data must travel with its longitude
        assert np.array_equal(out.x.isel(lat=0).values % 360, out.lon.values % 360)

    def test_minus180_180_is_unchanged(self):
        ds = _make_ds(np.arange(-179.5, 180, 1.0))
        assert normalise_lon(ds, "lon").identical(ds)

    def test_region_above_180_is_converted(self):
        out = normalise_lon(_make_ds(np.arange(200.5, 260, 1.0)), "lon")
        assert out.lon.values.min() == -159.5
        assert out.lon.values.max() == -100.5

    def test_region_straddling_180_is_unchanged(self):
        ds = _make_ds(np.arange(150.5, 230, 1.0))
        assert normalise_lon(ds, "lon").identical(ds)

    def test_no_lon_is_unchanged(self):
        ds = _make_ds(np.arange(0.5, 360, 1.0))
        assert normalise_lon(ds, None).identical(ds)
