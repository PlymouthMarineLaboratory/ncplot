import numpy as np
import xarray as xr

from ncplot.plot import _find_time_coord, _get_color_limits, _load_input_as_dataset


class TestFinal:
    def test_combines_files_by_time(self, tmp_path):
        ds1 = xr.Dataset(
            {"temp": ("valid_time", np.array([1.0, 2.0]))},
            coords={"valid_time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")},
        )
        ds1["valid_time"].attrs["standard_name"] = "time"

        ds2 = xr.Dataset(
            {"temp": ("valid_time", np.array([3.0, 4.0]))},
            coords={"valid_time": np.array(["2020-01-03", "2020-01-04"], dtype="datetime64[ns]")},
        )
        ds2["valid_time"].attrs["standard_name"] = "time"

        file1 = tmp_path / "part1.nc"
        file2 = tmp_path / "part2.nc"
        ds1.to_netcdf(file1)
        ds2.to_netcdf(file2)

        combined = _load_input_as_dataset([str(file1), str(file2)])

        assert _find_time_coord(ds1) == "valid_time"
        assert "time" in combined.coords
        assert combined.sizes["time"] == 4
        assert combined["temp"].values.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_combines_files_by_variables(self, tmp_path):
        ds1 = xr.Dataset(
            {"temp": ("time", np.array([1.0, 2.0]))},
            coords={"time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")},
        )
        ds2 = xr.Dataset(
            {"salinity": ("time", np.array([5.0, 6.0]))},
            coords={"time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")},
        )

        file1 = tmp_path / "temp.nc"
        file2 = tmp_path / "salinity.nc"
        ds1.to_netcdf(file1)
        ds2.to_netcdf(file2)

        combined = _load_input_as_dataset([str(file1), str(file2)])

        assert set(combined.data_vars) == {"temp", "salinity"}
        assert combined.sizes["time"] == 2

    def test_color_limits_use_entire_dataset(self):
        ds = xr.Dataset(
            {"temp": ("time", np.array([-2.0, 1.0, 5.0]))},
            coords={"time": np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")},
        )

        assert _get_color_limits(ds, "temp", {}) == (-5.0, 5.0)