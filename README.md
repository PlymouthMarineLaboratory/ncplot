<p align="center">
  <img src="docs-site/assets/img/ncplot_wordmark.svg" alt="NCPlot" width="380">
</p>

> [!TIP]
> 📖 **New here?** The NCPlot website — installing, a quickstart, the API reference, and more — lives at **[plymouthmarinelaboratory.github.io/ncplot](https://plymouthmarinelaboratory.github.io/ncplot/)**.

[![pypi](https://img.shields.io/pypi/v/ncplot.svg)](https://pypi.python.org/pypi/ncplot/)
[![Documentation Status](https://readthedocs.org/projects/ncplot/badge/?version=latest)](https://ncplot.readthedocs.io/en/latest/?badge=latest)


# NCPlot

Easy interactive viewing of NetCDF files in Python.

An easy to use Python (3.9 and above) package for quickly plotting the contents of NetCDF files or xarray datasets in Python or on the command line. 

This package is designed to work with [CF](https://cfconventions.org/) compliant NetCDF data.

## Installation

To install from conda:
```sh
conda install -c conda-forge ncplot

```

To install from pypi:
```sh
pip install ncplot 
```

Install the development version using using pip:
```sh
pip install git+https://github.com/pmlmodelling/ncplot.git
```



## How to use in Python



The package is made up of a simple and easy to use function: view. To visualize everything in a file:

```sh
from ncplot import view

view("example.nc")

```

or to visualize a specific variable or list of variables:

```sh
from ncplot import view

view("example.nc", vars)

```

where vars is either a string representing a variable or a list of variables.


If you want to plot xarray datasets dataarrays, first import the xarray accessor, then plot in a similar way to xarray's built in plotting method:


```sh
import ncplot.xarray
import xarray as xr

ds = xr.open_dataset("example.nc") 
ds.ncplot.view()

```




## Model grids

Regular longitude-latitude grids are plotted as they are. Other grids are put on a regular longitude-latitude
grid first, so that they are plotted in the same way, with the same sliders:

* **Curvilinear grids** with 2-D longitudes and latitudes, such as the native ocean grids of CMIP6 models, including
  those with a seam where the longitudes jump by 360 degrees and a tripolar north fold.
* **Unstructured grids**, whose cells run along one dimension with a longitude and latitude each.
* **[FVCOM](https://github.com/FVCOM-GitHub/FVCOM) output**: the triangular mesh, with variables on its nodes or its elements, and
  sigma layers as an extra slider.

Each time step and level is put on the regular grid only when it is shown. For a large file name the variable you want,
e.g. `view("fvcom_output.nc", vars="temp")`: with several variables (the default is all of them) ncplot has to read every
time step of each to set its colour scale.

Coastlines (`view("example.nc", coast=True)`) need geoviews and cartopy, and datashader draws large regular grids
quickly: `pip install "ncplot[complete]"`.

## How to use on the command line

Command line usage is as simple as:


```sh


ncplot example.nc

```












# Reference and tutorials

A full API reference, and a how-to guide are available at the [NCPlot website](https://plymouthmarinelaboratory.github.io/ncplot/) and on [readthedocs](https://ncplot.readthedocs.io/en/latest/).








