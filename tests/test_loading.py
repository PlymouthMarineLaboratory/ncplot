"""Opening files: the oddities of real model output."""

import numpy as np

from ncplot.loading import open_one

from conftest import CMIP6_FILES, FVCOM_FILE


def test_fvcom_times_are_decoded_though_one_time_variable_cannot_be():
    """FVCOM's Itime2 is in "msec since 00:00:00", which xarray cannot decode:
    that must not stop the file's real time axis being decoded."""
    ds = open_one(FVCOM_FILE)
    assert np.issubdtype(ds["time"].dtype, np.datetime64)
    assert str(ds["time"].values[0])[:10] == "2012-01-02"
    assert ds["Itime2"].dtype.kind in "iu"


def test_non_standard_calendar_times_are_kept():
    """Some CMIP6 files have no-leap or 360-day calendars: cftime dates."""
    kinds = {open_one(path)["time"].dtype.kind for path in CMIP6_FILES}
    assert kinds <= {"M", "O"}
    assert "O" in kinds
