How to use ncplot 
---------------------------

Using ncplot is very easy. It lets you plot the contents of a NetCDF automatically. We will illustrate this by plotting sea surface temperature from the National Oceanic and Atmospheric Administration's COBE2 dataset. To plot a file in Python, preferably a Jupyter notebook, do the following::

    from ncplot import view
    view("https://psl.noaa.gov/thredds/dodsC/Datasets/COBE2/sst.mon.ltm.1981-2010.nc", vars = "sst", coast = True)

In this case, we use the coast argument to ensure the world's coastlines are displayed.

.. raw:: html
   :file: plot.html

There is also built in support for xarray datasets::

    import ncplot.xarray
    import xarray as xr 
    ds = xr.open_dataset("https://psl.noaa.gov/thredds/dodsC/Datasets/COBE2/sst.mon.ltm.1981-2010.nc")
    ds["sst"].ncplot.view(vars = "sst", coast = True)

.. raw:: html
   :file: xarray.html

ncplot operates as a command line tool, letting you view the contents of a NetCDF file on a website. All you need to do is provide the file you want to look at::

    $ ncplot foo.nc


Model grids
~~~~~~~~~~~

Regular longitude-latitude grids are plotted as they are. Curvilinear grids (2-D longitudes and latitudes, such as the native
ocean grids of CMIP6 models, with their seam and tripolar north fold), unstructured grids, and the triangular mesh of FVCOM output
are put on a regular longitude-latitude grid first, and plotted in the same way, with the same sliders. FVCOM's sigma layers get a slider
of their own. Each time and level is put on the regular grid only when it is shown. For a large file name the variable you want, as
with several variables (the default is all of them) ncplot has to read every time step of each to set its colour scale::

    from ncplot import view

    view("fvcom_output.nc", vars="temp")

Coastlines need geoviews and cartopy, and datashader draws large regular grids quickly: ``pip install "ncplot[complete]"``.
