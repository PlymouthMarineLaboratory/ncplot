"""Shared set-up for the tests: the small model files in ``tests/data`` (made by
``make_test_data.py`` from CMIP6 ocean output and a corner of an FVCOM run)."""

import glob
import os

import pytest

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# one file for each layout of the CMIP6 horizontal grids
CMIP6_FILES = sorted(glob.glob(os.path.join(DATA, "cmip6_*.nc")))
# those on regular grids (1-D longitudes and latitudes), which need no regridding
REGULAR_FILES = [f for f in CMIP6_FILES if f.endswith("_1d.nc")]
# those whose longitudes and latitudes are 2-D (a curvilinear grid)
CURVILINEAR_FILES = [f for f in CMIP6_FILES if f not in REGULAR_FILES]
FVCOM_FILE = os.path.join(DATA, "fvcom_small.nc")
ALL_FILES = CMIP6_FILES + [FVCOM_FILE]


def file_id(path):
    return os.path.basename(path).replace(".nc", "")


def variable_of(path):
    """The variable to plot from a test file."""
    return "temp" if path == FVCOM_FILE else "tos"


@pytest.fixture(scope="session")
def coastlines_available():
    """Skip the test if coastlines cannot be drawn: they need geoviews and
    cartopy, and cartopy downloads Natural Earth's coastline the first time
    (so it needs the internet)."""
    try:
        import geoviews  # noqa: F401
        from cartopy.io import shapereader
    except ImportError as error:
        pytest.skip(f"coastlines need geoviews and cartopy: {error}")
    try:
        shapereader.natural_earth(resolution="110m", category="physical", name="coastline")
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"the Natural Earth coastline could not be had: {error}")
