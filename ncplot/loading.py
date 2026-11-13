"""Opening a NetCDF file lazily and forgivingly."""

import warnings

import xarray as xr


def decode_what_decodes(ds):
    """``ds`` (opened with ``decode_times=False``) with each time variable that
    can be decoded decoded, and those that cannot (FVCOM writes ``Itime2`` in
    ``msec since 00:00:00``, which has no date) left as they were."""
    from xarray.conventions import decode_cf_variable

    decoded = {}
    for name, variable in ds.variables.items():
        if " since " not in str(variable.attrs.get("units", "")):
            continue
        try:
            decoded[name] = decode_cf_variable(name, variable, decode_times=True)
        except Exception:  # noqa: BLE001 - whatever xarray or cftime cannot decode stays as it is
            continue
    coords = {name: variable for name, variable in decoded.items() if name in ds.coords}
    data = {name: variable for name, variable in decoded.items() if name not in ds.coords}
    return ds.assign_coords(coords).assign(data)


def open_one(path, **kwargs):
    """``xr.open_dataset(path)``, falling back to decoding the times one
    variable at a time when xarray cannot decode all of them."""
    with warnings.catch_warnings():
        # a reference date like FVCOM's "msec since 00:00:00" is ambiguous, and told so
        warnings.filterwarnings("ignore", category=xr.SerializationWarning)
        try:
            return xr.open_dataset(path, **kwargs)
        except ValueError as error:
            if "decode" not in str(error).lower() and "time" not in str(error).lower():
                raise
        return decode_what_decodes(xr.open_dataset(path, decode_times=False, **kwargs))
