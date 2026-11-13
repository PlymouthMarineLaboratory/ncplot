"""``view`` on every grid layout of the CMIP6 test data and on FVCOM's mesh:
the plot is made by ncplot's hvplot code, and drawn as a Bokeh figure."""

import holoviews as hv
import numpy as np
import pytest
import xarray as xr

import ncplot
import ncplot.plot
from ncplot.loading import open_one
from ncplot.regrid import native_range

from conftest import (
    ALL_FILES, CMIP6_FILES, CURVILINEAR_FILES, FVCOM_FILE, REGULAR_FILES, file_id, variable_of,
)

VIRIDIS_FIRST, RDBU_FIRST = "#440154", "#053061"


@pytest.fixture(autouse=True)
def notebook(monkeypatch):
    """As in a notebook, where view returns the plot."""
    monkeypatch.setattr(ncplot.plot, "in_notebook", lambda out=None: True)


def draw(plot):
    return hv.renderer("bokeh").get_plot(plot)


def image_of(plot):
    return next(np.asarray(r.data_source.data["image"][0]) for r in plot.state.renderers if "image" in r.data_source.data)


def mapper_of(plot):
    return next(r.glyph.color_mapper for r in plot.state.renderers if hasattr(r.glyph, "color_mapper"))


@pytest.mark.parametrize("path", ALL_FILES, ids=file_id)
def test_every_layout_is_plotted(path):
    result = ncplot.view(path, vars=variable_of(path), coast=False)
    assert isinstance(result, hv.DynamicMap)
    assert hv.render(result, backend="bokeh") is not None
    expected = ["siglay", "time"] if path == FVCOM_FILE else ["time"]
    assert sorted(d.name for d in result.kdims) == expected


@pytest.mark.parametrize("path", ALL_FILES, ids=file_id)
def test_every_layout_is_plotted_without_datashader(path, monkeypatch):
    """datashader is optional, and was needed even by the plots of regular grids."""
    monkeypatch.setattr(ncplot.plot, "as_raster", lambda: False)
    result = ncplot.view(path, vars=variable_of(path), coast=False)
    assert draw(result).state is not None


@pytest.mark.parametrize("path", CURVILINEAR_FILES, ids=file_id)
def test_a_curvilinear_grid_is_drawn_whole_on_a_regular_one(path):
    plot = draw(ncplot.view(path, vars="tos", coast=False))
    image = image_of(plot)
    # a map of the globe, with its sea and its land (missing)
    assert 0.3 < np.isfinite(image).mean() < 0.95
    assert plot.state.x_range.start < -170 and plot.state.x_range.end > 170
    assert plot.state.y_range.start < -60 and plot.state.y_range.end > 60
    # the scale is of all the times, symmetric about 0 as the values span it
    low, high = native_range(open_one(path), "tos")
    assert low < 0 < high
    colours = mapper_of(plot)
    assert colours.low == pytest.approx(-colours.high) and colours.high >= max(-low, high) - 1e-3
    assert colours.palette[0] == RDBU_FIRST


def test_fvcom_is_drawn_with_a_slider_for_time_and_one_for_its_layers():
    ds = open_one(FVCOM_FILE)
    for name in ("temp", "u"):
        # one variable on the nodes, one on the elements
        result = ncplot.view(FVCOM_FILE, vars=name, coast=False)
        assert sorted(d.name for d in result.kdims) == ["siglay", "time"]
        plot = draw(result)
        assert np.isfinite(image_of(plot)).any()
        low, high = native_range(ds, name)
        colours = mapper_of(plot)
        if low > 0:
            assert (colours.low, colours.high) == (pytest.approx(low), pytest.approx(high))
            assert colours.palette[0] == VIRIDIS_FIRST
        else:
            assert colours.low == pytest.approx(-colours.high) and colours.palette[0] == RDBU_FIRST
        # the mesh's own corner of the world
        assert plot.state.x_range.start == pytest.approx(-5.0, abs=0.1)
        assert plot.state.y_range.start == pytest.approx(50.0, abs=0.1)


def test_the_sliders_step_through_the_times_and_layers():
    ds = open_one(FVCOM_FILE)
    result = ncplot.view(FVCOM_FILE, vars="temp", coast=False)
    plot = draw(result)
    names = [d.name for d in result.kdims]
    first = image_of(plot).copy()
    for step in (0, 1):
        for layer in (1, 2, 3):
            key = tuple(layer if name == "siglay" else result.kdims[names.index("time")].values[step] for name in names)
            plot.update(key)
            expected = ds["temp"].isel(time=step, siglay=layer - 1).values
            shown = image_of(plot)
            # each frame is that time and layer of the mesh
            assert np.nanmin(shown) >= expected.min() - 1e-4 and np.nanmax(shown) <= expected.max() + 1e-4
            if (step, layer) != (0, 1):
                assert not np.array_equal(np.nan_to_num(first), np.nan_to_num(shown))


def test_a_given_colour_scale_is_used():
    plot = draw(ncplot.view(FVCOM_FILE, vars="temp", coast=False, clim=(5, 15)))
    colours = mapper_of(plot)
    assert (colours.low, colours.high) == (5, 15)


@pytest.mark.parametrize("path", REGULAR_FILES, ids=file_id)
def test_regular_grids_are_plotted_as_they_were(path):
    plot = draw(ncplot.view(path, vars="tos", coast=False))
    assert image_of(plot).size > 0


def test_datasets_dataarrays_lists_and_patterns_are_plotted():
    path = CURVILINEAR_FILES[0]
    for given in (xr.open_dataset(path), xr.open_dataset(path)["tos"], [path], path.replace("cmip6_ij", "cmip6_i?")):
        result = ncplot.view(given, vars=None if isinstance(given, xr.DataArray) else "tos", coast=False)
        assert isinstance(result, hv.DynamicMap)
        image = image_of(draw(result))
        assert 0.3 < np.isfinite(image).mean() < 0.95


def test_what_is_not_on_a_grid_is_plotted_as_before():
    ds = xr.Dataset(
        {"sst": ("time", np.arange(10.0))}, coords={"time": np.arange("2000-01", "2000-11", dtype="datetime64[M]")}
    )
    result = ncplot.view(ds)
    assert result is not None
    assert hv.render(result, backend="bokeh") is not None


def test_coastlines_are_drawn_over_a_regridded_map(coastlines_available):
    for path in (CURVILINEAR_FILES[0], FVCOM_FILE):
        plot = draw(ncplot.view(path, vars=variable_of(path), coast=True))
        assert image_of(plot).size > 0
        # the coastline is a second layer on the map
        assert len(plot.state.renderers) > 1


def test_the_html_file_is_made(tmp_path):
    out = tmp_path / "map.html"
    assert ncplot.view(FVCOM_FILE, vars="temp", out=str(out), coast=False) is None
    assert out.stat().st_size > 10_000
